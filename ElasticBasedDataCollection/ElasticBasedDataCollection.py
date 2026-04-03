import urllib3
import warnings
import json
import uuid
import datetime
import pandas as pd
import re
import os
import yaml
from elasticsearch import Elasticsearch, ElasticsearchWarning

# ==========================================
# 1. Environment & Constants Configuration
# ==========================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.simplefilter('ignore', category=ElasticsearchWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='elasticsearch')
warnings.filterwarnings("ignore", message=".*verify_certs=False.*")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Defaults (overridden by config.yaml at runtime)
CSV_DIR_NAME = 'ProcmonCsvLog'
FINAL_LOG_DIR = 'FinalLog'
INTERNAL_IPS = ["127.0.0.1", "0.0.0.0"]
CSV_HOSTNAME = "unknown"
CSV_PRINCIPAL = "NT AUTHORITY\\SYSTEM"
TIMEZONE_OFFSET_HOURS = 8

# Fixed UUID namespace
NAMESPACE_EDR = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')

# Enforced output field order (top-level)
OUTPUT_ORDER = [
    "action",
    "actorID",
    "hostname",
    "id",
    "object",
    "objectID",
    "pid",
    "ppid",
    "principal",
    "properties",
    "tid",
    "timestamp"
]

# CSV Schema (for field alignment)
CSV_SCHEMA = {
    ("READ", "FILE"): ["acuity_level", "file_path", "image_path", "size"],
    ("ADD", "REGISTRY"): ["acuity_level", "data", "image_path", "key", "type", "value"],
    ("CREATE", "THREAD"): ["acuity_level", "image_path", "src_pid", "src_tid", "stack_base", "stack_limit",
                           "start_address", "subprocess_tag", "tgt_pid", "tgt_tid", "user_stack_base",
                           "user_stack_limit"],
    ("TERMINATE", "THREAD"): ["acuity_level", "image_path", "src_pid", "src_tid", "stack_base", "stack_limit",
                              "start_address", "subprocess_tag", "tgt_pid", "tgt_tid", "user_stack_base",
                              "user_stack_limit"],
    ("LOAD", "MODULE"): ["acuity_level", "base_address", "image_path", "module_path"],
    ("MODIFY", "FILE"): ["acuity_level", "file_path", "image_path", "info_class"],
    ("REMOVE", "REGISTRY"): ["acuity_level", "data", "image_path", "key", "value"]
}
CSV_NUMERIC_PROPERTIES = {"acuity_level", "size", "offset", "src_pid", "src_tid", "tgt_pid", "tgt_tid"}

# ==========================================
# 2. Global State Tables
# ==========================================
PROCESS_MAP = {}  # Map<pid_str, creation_timestamp_str>
FILE_MAP = {}  # Map<file_path, creation_timestamp_str>
REG_KEY_MAP = {}  # Map<key_path, delete_count_int>
FLOW_SESSION_MAP = {}  # Map<(ext_ip, ext_port, int_port, proto), pid_str>

DEFAULT_TIME = "2026-01-01T00:00:00.000+08:00"


# ==========================================
# 3. Utility Functions
# ==========================================

def generate_deterministic_uuid(seed_string):
    if seed_string is None:
        seed_string = "NULL"
    return str(uuid.uuid5(NAMESPACE_EDR, str(seed_string)))


def get_int_value(value, default=-1):
    try:
        if pd.isna(value) or str(value).strip() == "": return default
        return int(value)
    except Exception:
        return default


def get_str_value(value, default="NULL"):
    if value is None or (isinstance(value, float) and pd.isna(value)) or str(value) == "":
        return default
    return str(value)


def get_numeric_str_value(value, default="-1"):
    if value is None or (isinstance(value, float) and pd.isna(value)) or str(value) == "":
        return default
    return str(value)


def parse_iso_timestamp(ts_str):
    try:
        if not ts_str: return datetime.datetime.now().astimezone()
        if ts_str.endswith('Z'):
            dt = datetime.datetime.fromisoformat(ts_str[:-1])
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        else:
            dt = datetime.datetime.fromisoformat(ts_str)
        tz = datetime.timezone(datetime.timedelta(hours=TIMEZONE_OFFSET_HOURS))
        return dt.astimezone(tz)
    except Exception:
        return datetime.datetime.now().astimezone()


def parse_time_str(time_str):
    try:
        s = str(time_str).strip()
        if not s: return datetime.time(0, 0, 0)
        parts = s.split()
        if len(parts) != 2: return datetime.time(0, 0, 0)
        t_part, meridian = parts
        h_str, m_str, s_full = t_part.split(':')

        if '.' in s_full:
            s_str, ms_str = s_full.split('.')
            ms_str = ms_str[:6]
        else:
            s_str = s_full
            ms_str = "0"

        h, m, s, us = int(h_str), int(m_str), int(s_str), int(ms_str.ljust(6, '0'))

        if meridian.upper() == 'PM' and h != 12:
            h += 12
        elif meridian.upper() == 'AM' and h == 12:
            h = 0

        return datetime.time(h, m, s, us)
    except Exception:
        return datetime.time(0, 0, 0)


def load_config(config_path=None):
    """Load and validate configuration from a YAML file."""
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.yaml")
    if not os.path.exists(config_path):
        print(f"[ERROR] Configuration file '{config_path}' not found.")
        print("  Please copy 'config.yaml.example' to 'config.yaml' and edit it.")
        return None
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        print("[ERROR] Configuration file is empty or invalid.")
        return None
    return cfg


# ==========================================
# 4. Core Logic: Stateful ID Generation
# ==========================================

def get_flow_tuple(props):
    src_ip = get_str_value(props.get('src_ip'))
    src_port = get_str_value(props.get('src_port'))
    dest_ip = get_str_value(props.get('dest_ip'))
    dest_port = get_str_value(props.get('dest_port'))
    proto = get_str_value(props.get('l4protocol'))

    if src_ip in INTERNAL_IPS:
        return dest_ip, dest_port, src_port, proto
    else:
        return src_ip, src_port, dest_port, proto


def process_log_ids_stateful(log_entry):
    action = log_entry['action']
    obj_type = log_entry['object']

    pid = str(log_entry['pid'])
    ppid = str(log_entry.get('ppid', -1))
    timestamp = log_entry['timestamp']
    props = log_entry['properties']

    # --- 1. Preprocessing: FLOW PID correction ---
    if obj_type == "FLOW":
        ext_ip, ext_port, int_port, proto = get_flow_tuple(props)
        flow_key = (ext_ip, ext_port, int_port, proto)

        if action == "START" and pid != "-1" and pid != "0":
            FLOW_SESSION_MAP[flow_key] = pid

        elif action == "MESSAGE":
            if pid == "-1" or pid == "0":
                if flow_key in FLOW_SESSION_MAP:
                    pid = str(FLOW_SESSION_MAP[flow_key])
                    log_entry['pid'] = int(pid)
            else:
                FLOW_SESSION_MAP[flow_key] = pid

    # --- 2. Generate ActorID (initiator) ---
    actor_pid = ppid if (obj_type == "PROCESS" and action == "CREATE") else pid
    actor_creation_time = PROCESS_MAP.get(actor_pid, DEFAULT_TIME)

    actor_seed = f"{actor_pid}{actor_creation_time}"
    actor_uuid = generate_deterministic_uuid(actor_seed)

    # --- 3. Generate ObjectID (target) & update state ---
    object_seed = ""

    if obj_type == "PROCESS":
        target_pid = pid

        if action == "CREATE":
            PROCESS_MAP[target_pid] = timestamp
            object_seed = f"{target_pid}{timestamp}"

        elif action == "OPEN":
            real_target_pid = get_str_value(props.get('target_pid') or props.get('tgt_pid'))
            if real_target_pid == "NULL" or real_target_pid == "-1":
                real_target_pid = target_pid

            target_time = PROCESS_MAP.get(real_target_pid, DEFAULT_TIME)
            object_seed = f"{real_target_pid}{target_time}"

            if 'target_pid' in props:
                del props['target_pid']

        elif action == "TERMINATE":
            target_time = PROCESS_MAP.get(target_pid, DEFAULT_TIME)
            object_seed = f"{target_pid}{target_time}"
            if target_pid in PROCESS_MAP:
                del PROCESS_MAP[target_pid]
        else:
            target_time = PROCESS_MAP.get(target_pid, DEFAULT_TIME)
            object_seed = f"{target_pid}{target_time}"

    elif obj_type == "FILE":
        file_path = get_str_value(props.get('file_path') or props.get('path'))

        if action == "CREATE":
            FILE_MAP[file_path] = timestamp
            object_seed = f"{file_path}{timestamp}"
        elif action == "DELETE":
            f_time = FILE_MAP.get(file_path, DEFAULT_TIME)
            object_seed = f"{file_path}{f_time}"
            if file_path in FILE_MAP: del FILE_MAP[file_path]
        elif action == "RENAME":
            f_time = FILE_MAP.get(file_path, DEFAULT_TIME)
            object_seed = f"{file_path}{f_time}"
        else:
            f_time = FILE_MAP.get(file_path, DEFAULT_TIME)
            object_seed = f"{file_path}{f_time}"

    elif obj_type == "FLOW":
        ext_ip, ext_port, int_port, proto = get_flow_tuple(props)

        if action == "OPEN":
            object_seed = f"{int_port}"
        else:
            object_seed = f"{ext_ip}{ext_port}{int_port}{proto}"

    elif obj_type == "REGISTRY":
        key = get_str_value(props.get('key'))
        count = REG_KEY_MAP.get(key, 0)
        object_seed = f"{key}{count}"

        if action == "REMOVE":
            REG_KEY_MAP[key] = count + 1

    elif obj_type == "MODULE":
        mod_path = get_str_value(props.get('module_path'))
        object_seed = f"{mod_path}"

    elif obj_type == "SERVICE":
        svc_name = get_str_value(props.get('name'))
        object_seed = f"{svc_name}"

    elif obj_type == "TASK":
        t_name = get_str_value(props.get('task_name'))
        u_name = get_str_value(props.get('user_name'))
        object_seed = f"{t_name}{u_name}"

    elif obj_type == "THREAD":
        if action == "REMOTE_CREATE":
            t_pid = get_str_value(props.get('tgt_pid', pid))
            t_tid = get_str_value(props.get('tgt_tid', '-1'))
            object_seed = f"{t_pid}{t_tid}"

            tgt_proc_time = PROCESS_MAP.get(t_pid, DEFAULT_TIME)
            tgt_proc_seed = f"{t_pid}{tgt_proc_time}"
            props['tgt_pid_uuid'] = generate_deterministic_uuid(tgt_proc_seed)
        else:
            t_tid = get_str_value(props.get('tgt_tid', '-1'))
            if t_tid == '-1': t_tid = str(log_entry['tid'])
            object_seed = f"{pid}{t_tid}"

    elif obj_type == "USER_SESSION":
        user = get_str_value(props.get('user'))
        object_seed = f"{user}"

    elif obj_type == "SHELL":
        object_seed = f"{actor_uuid}"

    if not object_seed:
        object_seed = f"{obj_type}{timestamp}"

    object_uuid = generate_deterministic_uuid(object_seed)

    event_seed = f"{timestamp}{action}{obj_type}{actor_uuid}{object_uuid}"
    event_uuid = generate_deterministic_uuid(event_seed)

    log_entry['id'] = event_uuid
    log_entry['actorID'] = actor_uuid
    log_entry['objectID'] = object_uuid

    return log_entry


# ==========================================
# 5. ES Log Transformation
# ==========================================
def get_es_principal(row):
    domain = row.get('user.domain', 'NT AUTHORITY')
    name = row.get('user.name', 'SYSTEM')
    if domain and name: return f"{domain}\\{name}"
    return name or "NT AUTHORITY\\SYSTEM"


def transform_es_log(behavior_type, row):
    object_type, action = behavior_type.split(': ')

    raw_pid = row.get('process.pid') or row.get('winlog.process.pid')
    if object_type == "THREAD" and action == "REMOTE_CREATE":
        raw_pid = row.get('winlog.event_data.SourceProcessId') or raw_pid
    elif object_type == "FLOW":
        raw_pid = row.get('winlog.event_data.ProcessId') or raw_pid

    pid = get_int_value(raw_pid, -1)
    ppid = get_int_value(row.get('process.parent.pid'), -1)
    tid = get_int_value(row.get('process.thread.id') or row.get('winlog.process.thread.id'), -1)

    raw_image = (row.get('process.executable') or row.get('winlog.event_data.Image') or
                 row.get('winlog.event_data.ImagePath') or row.get('winlog.event_data.ProcessName'))
    if object_type == "THREAD" and action == "REMOTE_CREATE":
        raw_image = row.get('winlog.event_data.SourceImage') or raw_image

    image_path_val = get_str_value(raw_image)
    if image_path_val.startswith('"') and image_path_val.endswith('"'):
        image_path_val = image_path_val[1:-1]

    props = {}

    if object_type == "PROCESS":
        props["acuity_level"] = "-1"
        props["command_line"] = get_str_value(row.get('process.command_line'))
        props["image_path"] = image_path_val
        props["parent_image_path"] = get_str_value(row.get('process.parent.executable'))
        if action == "OPEN":
            props["target_pid"] = get_numeric_str_value(row.get('winlog.event_data.TargetProcessId'))
        if action in ["CREATE", "TERMINATE"]:
            props["sid"] = get_str_value(row.get('user.id'))
            props["user"] = get_es_principal(row)

    elif object_type == "FLOW":
        src_ip = get_str_value(row.get('source.ip') or row.get('winlog.event_data.SourceAddress'))
        dest_ip = get_str_value(row.get('destination.ip'))

        direction = "null"
        if src_ip in INTERNAL_IPS:
            direction = "outbound"
        elif dest_ip in INTERNAL_IPS:
            direction = "inbound"

        props = {
            "acuity_level": "-1",
            "dest_ip": dest_ip,
            "dest_port": get_numeric_str_value(row.get('destination.port')),
            "direction": direction,
            "image_path": image_path_val,
        }

        proto_val = row.get('network.iana_number') or row.get('winlog.event_data.Protocol')
        if not proto_val:
            t = str(row.get('network.transport', '')).lower()
            proto_val = "6" if t == 'tcp' else "17" if t == 'udp' else "1" if 'icmp' in t else "-1"

        props["l4protocol"] = get_numeric_str_value(proto_val)

        if action == "MESSAGE":
            props["size"] = get_numeric_str_value(row.get('network.bytes'))

        props["src_ip"] = src_ip
        props["src_port"] = get_numeric_str_value(row.get('source.port') or row.get('winlog.event_data.SourcePort'))

    elif object_type == "FILE":
        if action == "RENAME":
            original_path = row.get('file.Ext.original.path')
            file_path = get_str_value(original_path if original_path else row.get('file.path'))
            new_path = get_str_value(row.get('file.path'))

            props = {
                "acuity_level": "-1",
                "file_path": file_path,
                "image_path": image_path_val,
                "new_path": new_path
            }
        else:
            props["acuity_level"] = "-1"
            props["file_path"] = get_str_value(row.get('file.path'))
            props["image_path"] = image_path_val
            if action == "WRITE":
                props["size"] = get_numeric_str_value(row.get('file.size'))
            if action == "DELETE":
                props["info_class"] = "FileDispositionInformation"

    elif object_type == "REGISTRY":
        props["acuity_level"] = "-1"
        val_data = row.get('winlog.event_data.Details') or row.get('registry.data.strings')
        props["data"] = str(val_data) if val_data is not None else "NULL"
        props["image_path"] = image_path_val
        full_key_path = get_str_value(row.get('winlog.event_data.TargetObject') or row.get('registry.path'))
        props["key"] = full_key_path
        props["type"] = get_str_value(row.get('registry.data.type'))
        val_name = row.get('registry.value')
        if not val_name and full_key_path and '\\' in full_key_path:
            val_name = full_key_path.split('\\')[-1]
        props["value"] = get_str_value(val_name)

    elif object_type == "SHELL":
        props["acuity_level"] = "-1"
        severity = row.get('log.level', 'Informational').capitalize()
        host_name = row.get('winlog.computer_name', row.get('host.hostname', ''))
        user_name = row.get('winlog.user.name') or row.get('user.name', '')
        provider = row.get('winlog.provider_name', 'Microsoft-Windows-PowerShell')
        event_id = row.get('winlog.event_id', '')
        props[
            "context_info"] = f"Severity = {severity}\nHost Name = {host_name}\nProvider = {provider}\nEvent ID = {event_id}\nUser = {user_name}\nDetails = See payload for script content"
        props["image_path"] = image_path_val
        props["payload"] = get_str_value(row.get('powershell.file.script_block_text'))

    elif object_type == "TASK":
        if action == "START":
            props = {
                "acuity_level": "-1",
                "image_path": image_path_val,
                "path": "NULL",
                "task_name": get_str_value(row.get('winlog.event_data.TaskName')),
                "task_pid": "-1",
                "task_process_uuid": "NULL"
            }
        else:
            props["acuity_level"] = "-1"
            props["image_path"] = image_path_val
            props["task_name"] = get_str_value(row.get('winlog.event_data.TaskName'))
            user_val = get_str_value(row.get('winlog.event_data.SubjectUserSid'), "")
            if not user_val:
                d = row.get('winlog.event_data.SubjectDomainName', '')
                n = row.get('winlog.event_data.SubjectUserName', '')
                user_val = f"{d}\\{n}" if d and n else n
            props["user_name"] = get_str_value(user_val if user_val else None)

    elif object_type == "THREAD":
        props["acuity_level"] = "-1"
        props["image_path"] = image_path_val
        if action == "REMOTE_CREATE":
            props["src_pid"] = str(pid)
            props["src_tid"] = str(tid)
            props["stack_base"] = "-1"
            props["stack_limit"] = "-1"
            props["start_address"] = get_str_value(row.get('winlog.event_data.StartAddress'))
            props["subprocess_tag"] = "-1"
            props["tgt_pid"] = get_numeric_str_value(row.get('winlog.event_data.TargetProcessId'))
            props["tgt_pid_uuid"] = "NULL"
            props["tgt_tid"] = get_numeric_str_value(row.get('winlog.event_data.NewThreadId'))
            props["user_stack_base"] = "-1"
            props["user_stack_limit"] = "-1"
        else:
            props["src_pid"] = get_numeric_str_value(row.get('process.pid'))
            props["tgt_pid"] = get_numeric_str_value(row.get('winlog.event_data.TargetProcessId'))
            props["start_address"] = get_str_value(row.get('winlog.event_data.StartAddress'))

    elif object_type == "USER_SESSION":
        user_val = get_es_principal(row)
        logon_id = get_str_value(
            row.get('winlog.event_data.TargetLogonId') or row.get('winlog.event_data.LogonId') or row.get(
                'winlog.event_data.SubjectLogonId'))
        props["acuity_level"] = "-1"
        props["image_path"] = image_path_val
        props["user"] = user_val
        props["logon_id"] = logon_id
        if action == "GRANT":
            privs = row.get('winlog.event_data.PrivilegeList')
            props["privileges"] = "\n\t\t\t".join(privs) if isinstance(privs, list) else get_str_value(privs)
        elif action in ["REMOTE", "RDP", "LOGIN", "INTERACTIVE", "UNLOCK"]:
            props["requesting_domain"] = get_str_value(row.get('winlog.event_data.SubjectDomainName'))
            props["requesting_logon_id"] = get_str_value(row.get('winlog.event_data.SubjectLogonId'))
            if action in ["REMOTE", "RDP"]:
                props["src_ip"] = get_str_value(row.get('winlog.event_data.IpAddress') or row.get('source.ip'))
                props["src_port"] = get_numeric_str_value(row.get('winlog.event_data.IpPort') or row.get('source.port'))
            if action == "RDP":
                props["requesting_user"] = get_str_value(row.get('winlog.event_data.SubjectUserName'))
    else:
        props["acuity_level"] = "-1"
        props["image_path"] = image_path_val
        if object_type == "SERVICE":
            props["name"] = row.get('service.name') or row.get('winlog.event_data.ServiceName') or "NULL"
            props["service_type"] = row.get('winlog.event_data.ServiceType', 'user mode service')
            props["start_type"] = row.get('winlog.event_data.StartType', 'auto start')

    ts_obj = parse_iso_timestamp(row.get('@timestamp'))

    log_entry = {
        "action": action,
        "object": object_type,
        "pid": pid,
        "ppid": ppid,
        "principal": get_es_principal(row),
        "properties": props,
        "tid": tid,
        "timestamp": ts_obj.isoformat(timespec='milliseconds'),
        "hostname": row.get('host.hostname', 'unknown')
    }
    return log_entry


# ==========================================
# 6. CSV Log Transformation
# ==========================================
def get_csv_action_object(row):
    op = str(row.get('Operation', ''))
    if op in ["ReadFile", "QueryDirectory"]: return "READ", "FILE"
    if op in ["WriteFile", "SetFileInformation", "SetEndOfFile", "RegSetValue",
              "SetEndOfFileInformationFile"]: return "MODIFY", "FILE"
    if "Process Create" in op: return "CREATE", "PROCESS"
    if "Thread Create" in op: return "CREATE", "THREAD"
    if op == "RegCreateKey": return "ADD", "REGISTRY"
    if "Process Exit" in op: return "TERMINATE", "PROCESS"
    if "Thread Exit" in op: return "TERMINATE", "THREAD"
    if op == "RegDeleteKey": return "REMOVE", "REGISTRY"
    if op == "Load Image": return "LOAD", "MODULE"
    return None, None


def parse_csv_details(op, detail_str):
    props = {}
    if not isinstance(detail_str, str): return props
    if "Read" in op or "Write" in op or "Set" in op:
        m_offset = re.search(r'Offset: ([\d,]+)', detail_str)
        m_len = re.search(r'Length: ([\d,]+)', detail_str)
        m_info = re.search(r'InfoClass: (\w+)', detail_str)
        if m_offset: props['offset'] = m_offset.group(1).replace(',', '')
        if m_len: props['size'] = m_len.group(1).replace(',', '')
        if m_info: props['info_class'] = m_info.group(1)
        if op == "SetEndOfFileInformationFile": props['info_class'] = "FileEndOfFileInformation"
    elif op == "Load Image":
        m_base = re.search(r'Image Base: (0x[\da-fA-F]+)', detail_str)
        if m_base: props['base_address'] = m_base.group(1)
    elif "Thread" in op:
        m_tid = re.search(r'Thread ID: (\d+)', detail_str)
        if m_tid: props['tgt_tid'] = m_tid.group(1)
    return props


def process_csv_logs(df, start_date):
    """
    Process CSV logs.
    Pre-sorts the DataFrame by time to prevent minor row-ordering jitter
    from breaking the day-crossing detection algorithm.
    """
    # Precaution: ensure 'Time of Day' is string type
    df['Time of Day'] = df['Time of Day'].astype(str)

    # Sort by time string to resolve sub-second row ordering jitter
    try:
        df = df.sort_values(by='Time of Day')
    except Exception:
        pass  # If sorting fails, process in original order

    csv_logs = []
    current_date = start_date
    last_time_obj = None

    for _, row in df.iterrows():
        try:
            action, obj_type = get_csv_action_object(row)
            if not action: continue

            curr_time_obj = parse_time_str(row.get('Time of Day', '00:00:00'))

            # Day-crossing detection: if current time < previous, assume next day
            if last_time_obj and curr_time_obj < last_time_obj:
                current_date += datetime.timedelta(days=1)
            last_time_obj = curr_time_obj

            tz = datetime.timezone(datetime.timedelta(hours=TIMEZONE_OFFSET_HOURS))
            full_dt = datetime.datetime.combine(current_date, curr_time_obj).replace(tzinfo=tz)
            timestamp_str = full_dt.isoformat(timespec='milliseconds')

            pid = get_int_value(row.get('PID'))
            ppid = get_int_value(row.get('Parent PID'))
            tid = get_int_value(row.get('TID'))

            raw_props = parse_csv_details(str(row.get('Operation', '')), row.get('Detail', ''))
            raw_props['image_path'] = get_str_value(row.get('Image Path', ''))

            if obj_type == "FILE":
                raw_props['file_path'] = str(row.get('Path', ''))
            elif obj_type == "REGISTRY":
                raw_props['key'] = str(row.get('Path', ''))
                if '\\' in raw_props['key']: raw_props['value'] = raw_props['key'].split('\\')[-1]
                if action == "REMOVE": raw_props['data'] = "-"
            elif obj_type == "MODULE":
                raw_props['module_path'] = str(row.get('Path', ''))
            elif obj_type == "THREAD":
                raw_props['src_pid'] = str(pid)
                raw_props['src_tid'] = str(tid)
                raw_props['tgt_pid'] = str(pid)

            final_props = {}
            target_schema = CSV_SCHEMA.get((action, obj_type), [])
            if not target_schema:
                final_props = raw_props
            else:
                for key in target_schema:
                    val = raw_props.get(key)
                    if val is None or str(val) == "":
                        final_props[key] = "-1" if key in CSV_NUMERIC_PROPERTIES else "NULL"
                    else:
                        final_props[key] = str(val)

            log_entry = {
                "action": action,
                "object": obj_type,
                "pid": pid,
                "ppid": ppid,
                "principal": CSV_PRINCIPAL,
                "properties": final_props,
                "tid": tid,
                "timestamp": timestamp_str,
                "hostname": CSV_HOSTNAME
            }
            csv_logs.append(log_entry)

        except Exception as e:
            continue
    return csv_logs


# ==========================================
# 7. File Scanning & Selection
# ==========================================
def scan_and_select_csv():
    """
    Returns: (full_file_path, start_date, end_date, suffix_string)
    """
    if not os.path.exists(CSV_DIR_NAME):
        print(f"[ERROR] Directory '{CSV_DIR_NAME}' does not exist. Please create it and place CSV files inside.")
        return None, None, None, None

    # Regex to match single or dual date filename formats
    # Format 1: ProcmonLogfile-2026-02-02-R2.csv
    # Format 2: ProcmonLogfile-2026-02-02-2026-02-04-R2.csv
    pattern = re.compile(r"^ProcmonLogfile-(\d{4}-\d{2}-\d{2})(?:-(\d{4}-\d{2}-\d{2}))?-R.*\.csv$", re.IGNORECASE)
    valid_files = []

    for fname in os.listdir(CSV_DIR_NAME):
        match = pattern.match(fname)
        if match:
            date_str_1 = match.group(1)
            date_str_2 = match.group(2)
            try:
                start_date = datetime.date.fromisoformat(date_str_1)
                if date_str_2:
                    end_date = datetime.date.fromisoformat(date_str_2)
                else:
                    end_date = start_date

                suffix = fname.replace(".csv", "").replace(f"ProcmonLogfile-{date_str_1}", "")
                if date_str_2:
                    suffix = suffix.replace(f"-{date_str_2}", "")
                suffix = suffix.lstrip("-")

                valid_files.append({
                    "name": fname,
                    "start_date": start_date,
                    "end_date": end_date,
                    "path": os.path.join(CSV_DIR_NAME, fname),
                    "suffix": suffix
                })
            except ValueError:
                continue

    if not valid_files:
        print(f"[ERROR] No files matching the expected format found in '{CSV_DIR_NAME}'.")
        return None, None, None, None

    valid_files.sort(key=lambda x: x['name'], reverse=True)

    print("\nDiscovered the following CSV log files:")
    for idx, f in enumerate(valid_files):
        range_str = f"{f['start_date']}"
        if f['start_date'] != f['end_date']:
            range_str += f" to {f['end_date']}"
        print(f"[{idx}] {f['name']}  (time range: {range_str})")

    while True:
        try:
            choice = input("\nSelect a file index to process (e.g. 0): ").strip()
            idx = int(choice)
            if 0 <= idx < len(valid_files):
                selected = valid_files[idx]
                print(f"Selected: {selected['name']}")
                return selected['path'], selected['start_date'], selected['end_date'], selected['suffix']
            else:
                print("Invalid index, please try again.")
        except ValueError:
            print("Please enter a valid numeric index.")


# ==========================================
# 8. Main Pipeline
# ==========================================
def main():
    global CSV_DIR_NAME, FINAL_LOG_DIR, INTERNAL_IPS, CSV_HOSTNAME, CSV_PRINCIPAL, TIMEZONE_OFFSET_HOURS

    cfg = load_config()
    if not cfg:
        return

    es_cfg = cfg.get('elasticsearch', {})
    host_cfg = cfg.get('host', {})
    paths_cfg = cfg.get('paths', {})

    CSV_DIR_NAME = paths_cfg.get('csv_dir', CSV_DIR_NAME)
    FINAL_LOG_DIR = paths_cfg.get('output_dir', FINAL_LOG_DIR)

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(CSV_DIR_NAME):
        CSV_DIR_NAME = os.path.join(_script_dir, CSV_DIR_NAME)
    if not os.path.isabs(FINAL_LOG_DIR):
        FINAL_LOG_DIR = os.path.join(_script_dir, FINAL_LOG_DIR)

    INTERNAL_IPS = host_cfg.get('internal_ips', INTERNAL_IPS)
    CSV_HOSTNAME = host_cfg.get('hostname', CSV_HOSTNAME)
    CSV_PRINCIPAL = host_cfg.get('principal', CSV_PRINCIPAL)
    TIMEZONE_OFFSET_HOURS = cfg.get('timezone_offset_hours', TIMEZONE_OFFSET_HOURS)

    # 8.1 Read CSV and determine time range
    csv_file_path, file_start_date, file_end_date, file_suffix = scan_and_select_csv()
    if not csv_file_path:
        return

    try:
        print(f"Reading {csv_file_path} to calculate precise time range...")
        df = pd.read_csv(csv_file_path)
        if df.empty:
            print("[ERROR] CSV file is empty")
            return

        start_time_obj = parse_time_str(df.iloc[0]['Time of Day'])
        end_time_obj = parse_time_str(df.iloc[-1]['Time of Day'])

        cst_tz = datetime.timezone(datetime.timedelta(hours=TIMEZONE_OFFSET_HOURS))

        # Initial calculation
        start_dt_cst = datetime.datetime.combine(file_start_date, start_time_obj).replace(tzinfo=cst_tz)
        end_dt_cst = datetime.datetime.combine(file_end_date, end_time_obj).replace(tzinfo=cst_tz)

        # Branch: single-day file vs. multi-day range file
        if file_start_date == file_end_date:
            # Single-day mode: detect midnight crossing via time regression
            if end_time_obj < start_time_obj:
                end_dt_cst += datetime.timedelta(days=1)
                print("[INFO] Single-date file detected midnight crossing, end time auto +1 day")
        else:
            # Multi-day mode: trust the end date from the filename.
            # Note: if the filename end date is 02-04 and CSV data only goes to 02-04 10:00,
            # end_dt_cst will correctly reflect 02-04 10:00.
            # If the filename says 02-04 but CSV content only reaches 02-03 23:00,
            # ES will query an extra day, but no data will be lost.
            print(f"[INFO] Using end date range from filename: {file_start_date} -> {file_end_date}")

        start_dt_utc = start_dt_cst.astimezone(datetime.timezone.utc)
        end_dt_utc = end_dt_cst.astimezone(datetime.timezone.utc)

        start_query_ts = start_dt_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_query_ts = end_dt_utc.strftime('%Y-%m-%dT%H:%M:%SZ')

        print(f"ES query range (UTC): {start_query_ts} to {end_query_ts}")

    except Exception as e:
        print(f"[ERROR] Failed to process CSV time range: {e}")
        return

    # 8.2 Fetch ES logs
    print("Querying ES logs (full fetch)...")
    es = Elasticsearch(
        es_cfg.get('url', 'https://localhost:9200'),
        basic_auth=(es_cfg.get('username', 'elastic'), es_cfg.get('password', '')),
        verify_certs=es_cfg.get('verify_certs', False),
        ssl_show_warn=False
    )

    es_logs = []
    queries = {
        "FILE: CREATE": 'FROM logs-endpoint.events.file-* | WHERE event.type == "creation"',
        "FILE: DELETE": 'FROM logs-endpoint.events.file-* | WHERE event.type == "deletion"',
        "FILE: RENAME": 'FROM logs-endpoint.events.file-* | WHERE event.action == "rename"',
        "FILE: WRITE": 'FROM logs-endpoint.events.file-* | WHERE event.action == "write" OR event.action == "overwrite"',
        "FLOW: MESSAGE": 'FROM .ds-logs-network_traffic.flow-* | WHERE event.type == "connection"',
        "FLOW: OPEN": 'FROM logs-system.security-* | WHERE winlog.event_id == "5154"',
        "FLOW: START": 'FROM logs-endpoint.events.network-* | WHERE event.type == "start"',
        "PROCESS: CREATE": 'FROM logs-endpoint.events.process-* | WHERE event.type == "start"',
        "PROCESS: OPEN": 'FROM .ds-logs-windows.sysmon_operational* | WHERE event.action == "ProcessAccess"',
        "PROCESS: TERMINATE": 'FROM logs-endpoint.events.process-* | WHERE event.type == "end"',
        "REGISTRY: EDIT": 'FROM .ds-logs-endpoint.events.registry-default* | WHERE event.action == "modification"',
        "SERVICE: CREATE": 'FROM logs-system.security-*, logs-system.system-* | WHERE winlog.event_id == "4697" OR winlog.event_id == "7045"',
        "SHELL: COMMAND": 'FROM logs-windows.powershell_operational-* | WHERE winlog.event_id == "4104"',
        "TASK: CREATE": 'FROM logs-system.security-* | WHERE winlog.event_id == "4698"',
        "TASK: DELETE": 'FROM logs-system.security-* | WHERE winlog.event_id == "4699"',
        "TASK: MODIFY": 'FROM logs-system.security-* | WHERE winlog.event_id == "4702"',
        "TASK: START": 'FROM winlogbeat-* | WHERE winlog.event_id == "100"',
        "THREAD: REMOTE_CREATE": 'FROM .ds-winlogbeat* | WHERE event.code == "8"',
        "USER_SESSION: LOGIN": 'FROM logs-system.security-* | WHERE winlog.event_id == "4624"',
        "USER_SESSION: LOGOUT": 'FROM logs-system.security-* | WHERE winlog.event_id == "4634" OR winlog.event_id == "4647"',
        "USER_SESSION: GRANT": 'FROM logs-system.security-* | WHERE winlog.event_id == "4672"',
        "USER_SESSION: INTERACTIVE": 'FROM logs-system.security-* | WHERE winlog.event_id == "4624" AND winlog.event_data.LogonType == "2"',
        "USER_SESSION: RDP": 'FROM logs-system.security-* | WHERE winlog.event_id == "4624" AND winlog.event_data.LogonType == "10"',
        "USER_SESSION: REMOTE": 'FROM logs-system.security-* | WHERE winlog.event_id == "4624" AND winlog.event_data.LogonType == "3"',
        "USER_SESSION: UNLOCK": 'FROM logs-system.security-* | WHERE winlog.event_id == "4801"',
    }

    common_filter = f'| WHERE @timestamp >= "{start_query_ts}" AND @timestamp < "{end_query_ts}" AND host.name == "{CSV_HOSTNAME}"'

    for action_name, esql in queries.items():
        try:
            full_query = f"{esql} {common_filter}"
            resp = es.esql.query(query=full_query, format="json")
            if resp.body.get('values'):
                cols = [c['name'] for c in resp.body['columns']]
                for val in resp.body['values']:
                    es_logs.append(transform_es_log(action_name, dict(zip(cols, val))))
        except Exception as e:
            print(f"[ERROR] Query {action_name}: {e}")

    # 8.3 Process CSV logs (with the correct start date)
    csv_logs = process_csv_logs(df, file_start_date)
    print(f"ES log count: {len(es_logs)}, CSV log count: {len(csv_logs)}")

    # 8.4 Merge and sort
    all_logs = es_logs + csv_logs
    print("Sorting all logs by timestamp (to ensure correct state machine logic)...")
    all_logs.sort(key=lambda x: x['timestamp'])

    # 8.5 Stateful stream processing
    print("Running stateful ID generation and PID correction...")
    final_logs = []
    for log in all_logs:
        processed_log = process_log_ids_stateful(log)
        final_logs.append(processed_log)

    # Filter out logs with PID == -1
    print("Filtering out invalid PID logs...")
    final_logs = [log for log in final_logs if str(log.get('pid')) != '-1']

    # 8.6 Output results
    output_filename = f"FinalLog-{file_suffix}.json"
    if not os.path.exists(FINAL_LOG_DIR):
        try:
            os.makedirs(FINAL_LOG_DIR)
        except OSError as e:
            print(f"[ERROR] Failed to create output directory: {e}")
            return

    output_path = os.path.join(FINAL_LOG_DIR, output_filename)
    print(f"\nWriting processed results to file: {output_path} ...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for log in final_logs:
                ordered_log = {k: log.get(k) for k in OUTPUT_ORDER if k in log}
                f.write(json.dumps(ordered_log, ensure_ascii=False) + '\n')
        print(f"Write successful! Total logs written: {len(final_logs)}.")
    except Exception as e:
        print(f"[ERROR] Failed to write file: {e}")


if __name__ == "__main__":
    main()