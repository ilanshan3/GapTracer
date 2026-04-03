#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GapTracerEvaluation: Attack tracing evaluation with ML-based sequence judgment.

Uses a graph-based traceback engine to identify malicious processes from
unified behavior logs and RPC trace files. Start-point classification and
RPC window filtering use SeqClean (BERT encoder-decoder) + SeqJudge
(Random Forest) for sequence-level maliciousness classification.

Pipeline:
  1. Parse unified behavior logs (JSON Lines) and RPC trace files
  2. Filter suspicious RPC initiators by frequency and whitelist
  3. Identify initial malicious PIDs via ML sequence judgment
  4. Expand threat graph via bidirectional traversal (parent/child/file/RPC/registry)
  5. Discover malicious IPs from the threat cluster's network connections
  6. Lateral expansion: find additional processes connecting to discovered IPs
  7. Second-round graph traversal for newly discovered seeds
  8. Extract malicious entity UUIDs (image/file/network) from the full threat graph
  9. Compare extracted UUIDs against ground truth for Precision/Recall/F1/Jaccard
"""

import os
import sys
import json
import re
import bisect
import pickle
import hashlib
import zipfile
import warnings
from collections import defaultdict
from datetime import datetime, timedelta

import joblib
import torch
from dateutil import parser as dateutil_parser
from transformers import BertConfig, BertModel, PreTrainedModel
from transformers import logging as hf_logging
from transformers.modeling_outputs import Seq2SeqLMOutput

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()


# ==========================================
# 1. Feature Configuration
# ==========================================
WHITELIST_STRINGS = {
    "svchost", "testconsole", "conhost", "wmiprvse", "dllhost",
    "searchprotocolhost", "securityhealthservice", "securityhealthsystray",
    "runtimebroker", "gpupdate", "elastic-endpoint", "msedge", "trustedinstaller",
    "conhost.exe", "msedge.exe", "searchprotocolhost.exe", "zone.identifier",
    "dllhost.exe", "smartscreen.exe", "explorer.exe", "searchindexer.exe",
    "searchfilterhost.exe"
}

SYSTEM_PROCESS_BOUNDARY = {"dllhost.exe", "wmiprvse.exe"}

# IPs excluded from malicious IP discovery.
# 192.168.18.50 is the monitored host in the provided datasets.
# If you collect your own data, replace it with your host's IP.
TRIVIAL_IPS = {"127.0.0.1", "::1", "0.0.0.0", "255.255.255.255", "", "192.168.18.50"}


# ==========================================
# 2. Global State Tables
# ==========================================
name_lower_to_canonical = {}


# ==========================================
# 3. ML Configuration
# ==========================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_DATA_DIR = os.path.join(_PROJECT_ROOT, "ElasticBasedDataCollection", "ObRPC-onDataset")
_DATASET_GDRIVE_FILE_ID = "1T7DSQBqlQ1M673U1BOnsk8R30n6nupOx"
_SEQ_CLEAN_ROOT = os.path.join(_PROJECT_ROOT, "SeqCleanModel")
_SEQ_JUDGE_ROOT = os.path.join(_PROJECT_ROOT, "SeqJudgeModel")

CLEAN_MODEL_DIR = os.path.join(_SEQ_CLEAN_ROOT, "final_model")
VOCAB_PATH = os.path.join(_SEQ_CLEAN_ROOT, "vocab.json")
JUDGE_MODEL_PATH = os.path.join(_SEQ_JUDGE_ROOT, "rf_ngram_model.pkl")
VECTORIZER_PATH = os.path.join(_SEQ_JUDGE_ROOT, "ngram_vectorizer.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LENGTH = 4000
MAX_NEW_TOKENS = 500

RPC_TIME_WINDOW = 0.1
SUBGRAPH_TWS = 0.1


# ==========================================
# 4. Dataset Auto-Download
# ==========================================
def ensure_dataset():
    """Download and extract ObRPC-onDataset from Google Drive if not present locally."""
    if os.path.isdir(_DATA_DIR):
        return
    parent_dir = os.path.dirname(_DATA_DIR)
    os.makedirs(parent_dir, exist_ok=True)
    zip_path = os.path.join(parent_dir, "ObRPC-onDataset.zip")

    print(f"  Dataset not found at {_DATA_DIR}")
    print(f"  Downloading from Google Drive (file ID: {_DATASET_GDRIVE_FILE_ID}) ...")

    try:
        import gdown
    except ImportError:
        sys.exit(
            "ERROR: 'gdown' is required to download the dataset.\n"
            "       Install it with: pip install gdown"
        )

    url = f"https://drive.google.com/uc?id={_DATASET_GDRIVE_FILE_ID}"
    gdown.download(url, zip_path, quiet=False)

    if not os.path.isfile(zip_path):
        sys.exit(f"ERROR: Download failed — {zip_path} not found.")

    print("  Extracting dataset ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(parent_dir)
    os.remove(zip_path)

    if not os.path.isdir(_DATA_DIR):
        extracted = [d for d in os.listdir(parent_dir)
                     if os.path.isdir(os.path.join(parent_dir, d)) and d != "ObRPC-onDataset"]
        for d in extracted:
            candidate = os.path.join(parent_dir, d)
            if any(entry.startswith("Data-Attack") for entry in os.listdir(candidate)):
                os.rename(candidate, _DATA_DIR)
                break

    if not os.path.isdir(_DATA_DIR):
        sys.exit(f"ERROR: Extraction succeeded but {_DATA_DIR} was not created.")

    print(f"  Dataset ready at {_DATA_DIR}")


# ==========================================
# 5. Utility Functions
# ==========================================
def reset_trackers():
    global name_lower_to_canonical
    name_lower_to_canonical = {}


def normalize(text):
    return str(text).lower().strip() if text else ""


def is_whitelisted(target_str):
    if not target_str:
        return False
    target_lower = str(target_str).lower()
    for wl in WHITELIST_STRINGS:
        if wl in target_lower:
            return True
    return False


# ==========================================
# 6. Process Name Resolution
# ==========================================
def get_canonical_name(name):
    global name_lower_to_canonical
    if not name or name.lower() == "unknown":
        return "Unknown"
    clean_name = name.split('\\')[-1]
    if clean_name.lower().endswith('.exe'):
        clean_name = clean_name[:-4]
    lower_name = clean_name.lower()
    existing = name_lower_to_canonical.get(lower_name)
    if not existing:
        name_lower_to_canonical[lower_name] = clean_name
    elif sum(1 for c in clean_name if c.isupper()) > sum(1 for c in existing if c.isupper()):
        name_lower_to_canonical[lower_name] = clean_name
    return name_lower_to_canonical[lower_name]


def resolve_proc_name(pid, timestamp, default_name, pid_history):
    if pid not in pid_history:
        return default_name
    history = pid_history[pid]
    timestamps = [x[0] for x in history]
    idx = bisect.bisect_right(timestamps, timestamp)
    return history[idx - 1][1] if idx > 0 else history[0][1]


def resolve_image_path(pid, timestamp, pid_history_path):
    """Same time-alignment logic as resolve_proc_name, returns full image path."""
    if not pid_history_path or pid not in pid_history_path:
        return ""
    history = pid_history_path[pid]
    if not history:
        return ""
    timestamps = [x[0] for x in history]
    idx = bisect.bisect_right(timestamps, timestamp)
    return history[idx - 1][1] if idx > 0 else history[0][1]


def extract_proc_name(path, default="unknown"):
    """Extract clean process name from a file path (case-insensitive, without .exe)."""
    if not path or path == "NULL":
        return default.lower()
    clean = path.split('\\')[-1].lower()
    if clean.endswith('.exe'):
        clean = clean[:-4]
    return clean


# ==========================================
# 7. Log Parsing Engine
# ==========================================
def parse_json_logs(json_log_path):
    events, min_datetime = [], None
    pid_history = defaultdict(list)
    pid_history_path = defaultdict(list)
    if not os.path.exists(json_log_path):
        return events, min_datetime, pid_history, pid_history_path
    with open(json_log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                log = json.loads(line)
                ts = dateutil_parser.parse(log['timestamp'])
                log['parsed_timestamp'] = ts.timestamp()
                events.append(log)
                if min_datetime is None or ts < min_datetime:
                    min_datetime = ts
                pid = log.get('pid', -1)
                image_path = log.get('properties', {}).get('image_path')
                if pid != -1 and image_path and image_path != "NULL":
                    proc_name = image_path.split('\\')[-1]
                    pid_history[pid].append((log['parsed_timestamp'], proc_name))
                    pid_history_path[pid].append((log['parsed_timestamp'], image_path))
                    get_canonical_name(proc_name)
            except Exception:
                continue
    for pid in pid_history:
        pid_history[pid].sort(key=lambda x: x[0])
    for pid in pid_history_path:
        pid_history_path[pid].sort(key=lambda x: x[0])
    events.sort(key=lambda x: x['parsed_timestamp'])
    return events, min_datetime, pid_history, pid_history_path


def parse_rpc_logs(rpc_log_path, start_datetime, pid_history):
    raw_rpc_events = []
    current_date = start_datetime.date() if start_datetime else datetime.now().date()
    tz_info = start_datetime.tzinfo if start_datetime else None
    if not os.path.exists(rpc_log_path):
        return raw_rpc_events

    pattern = re.compile(
        r"(?P<time>\d{2}:\d{2}:\d{2}\.\d{3}) \| "
        r"(?P<src_proc>[\w\.\-]+)\((?P<src_pid>\d+)\) -> "
        r"(?P<dst_proc>[\w\.\-]+)\((?P<dst_pid>\d+)\)"
    )
    last_time = None
    with open(rpc_log_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                parsed_time = dateutil_parser.parse(match.group('time')).time()
                if last_time and parsed_time < last_time:
                    current_date += timedelta(days=1)
                last_time = parsed_time
                event_ts = datetime.combine(current_date, parsed_time, tzinfo=tz_info).timestamp()

                src_pid, dst_pid = int(match.group('src_pid')), int(match.group('dst_pid'))
                raw_src, raw_dst = match.group('src_proc'), match.group('dst_proc')
                if raw_src.lower() == 'unknown':
                    raw_src = resolve_proc_name(src_pid, event_ts, raw_src, pid_history)
                if raw_dst.lower() == 'unknown':
                    raw_dst = resolve_proc_name(dst_pid, event_ts, raw_dst, pid_history)
                raw_rpc_events.append({
                    "timestamp": event_ts,
                    "src_proc": get_canonical_name(raw_src),
                    "src_pid": src_pid,
                    "dst_proc": get_canonical_name(raw_dst),
                    "dst_pid": dst_pid,
                    "raw": line.strip()
                })
    return raw_rpc_events


def load_or_parse_logs(json_path, rpc_path):
    if not os.path.exists(json_path) or not os.path.exists(rpc_path):
        return None, None, None, None, None

    unique_str = (f"{os.path.abspath(json_path)}_{os.path.getmtime(json_path)}_"
                  f"{os.path.abspath(rpc_path)}_{os.path.getmtime(rpc_path)}")
    cache_hash = hashlib.md5(unique_str.encode('utf-8')).hexdigest()
    cache_dir = os.path.dirname(os.path.abspath(json_path))
    cache_file = os.path.join(cache_dir, f".edr_cache_seq15_{cache_hash}.pkl")

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    json_events, start_dt, pid_hist, pid_hist_path = parse_json_logs(json_path)
    raw_rpcs = parse_rpc_logs(rpc_path, start_dt, pid_hist)

    if json_events and raw_rpcs:
        with open(cache_file, 'wb') as f:
            pickle.dump((json_events, start_dt, pid_hist, pid_hist_path, raw_rpcs), f)

    return json_events, start_dt, pid_hist, pid_hist_path, raw_rpcs


def filter_rpc_events(rpc_events, max_count=100):
    rpc_counts = defaultdict(int)
    for rpc in rpc_events:
        rpc_counts[(normalize(rpc['src_proc']), rpc['src_pid'])] += 1
    filtered_events, entity_stats = [], {}
    for rpc in rpc_events:
        proc_name = normalize(rpc['src_proc'])
        key = (proc_name, rpc['src_pid'])
        count = rpc_counts[key]
        if count < max_count and not is_whitelisted(proc_name):
            filtered_events.append(rpc)
            if key not in entity_stats:
                entity_stats[key] = {"first_ts": rpc['timestamp'], "count": count}
            elif rpc['timestamp'] < entity_stats[key]["first_ts"]:
                entity_stats[key]["first_ts"] = rpc['timestamp']
    return filtered_events, entity_stats


# ==========================================
# 8. Unified Event Stream & Sequence Extraction
# ==========================================
def build_unified_event_stream(json_events, rpc_events, pid_history):
    """
    Merge JSON and RPC events into a single time-sorted stream,
    with pid_history used to auto-fill missing process names.
    """
    unified = []

    for e in rpc_events:
        unified.append({
            'Type': 'RPC',
            'Timestamp': e['timestamp'],
            'PID': e['src_pid'],
            'ProcName': e['src_proc'].lower(),
            'RPID': e['dst_pid'],
            'RProcName': e['dst_proc'].lower(),
            'Raw': e
        })

    for e in json_events:
        obj_type = e.get('object')
        action = e.get('action')
        props = e.get('properties', {})
        ts = e['parsed_timestamp']

        if obj_type == 'PROCESS' and action == 'CREATE':
            ppid = e.get('ppid', -1)
            p_path = props.get('parent_image_path', '')
            cpid = e.get('pid', -1)
            c_path = props.get('image_path', '')

            if ppid != -1 and cpid != -1:
                p_name = extract_proc_name(p_path)
                if p_name == "unknown":
                    raw_name = resolve_proc_name(ppid, ts, "unknown", pid_history)
                    p_name = extract_proc_name(raw_name)

                c_name = extract_proc_name(c_path)
                if c_name == "unknown":
                    raw_name = resolve_proc_name(cpid, ts, "unknown", pid_history)
                    c_name = extract_proc_name(raw_name)

                unified.append({
                    'Type': 'CreateProcess',
                    'Timestamp': ts,
                    'PID': ppid,
                    'ProcName': p_name,
                    'CPID': cpid,
                    'CProcName': c_name,
                    'Raw': e
                })
        else:
            pid = e.get('pid', -1)
            path = props.get('image_path', '')
            if pid != -1:
                p_name = extract_proc_name(path)
                if p_name == "unknown":
                    raw_name = resolve_proc_name(pid, ts, "unknown", pid_history)
                    p_name = extract_proc_name(raw_name)

                unified.append({
                    'Type': 'Normal',
                    'Timestamp': ts,
                    'PID': pid,
                    'ProcName': p_name,
                    'Raw': e
                })

    unified.sort(key=lambda x: x['Timestamp'])
    return unified


def extract_and_filter_subgraph_onthefly(unified_events, start_pid, start_proc_name, tws=0.1):
    """
    Single-pass temporal traversal: strict (PID, proc_name) contamination tracking
    with time-window noise filtering.
    """
    start_node = (start_pid, start_proc_name.lower())
    traced_nodes = {start_node}

    pid_time_map = {}
    filtered_events = []

    for E in unified_events:
        src_node = (E['PID'], E['ProcName'])

        if src_node not in traced_nodes:
            continue

        is_noise = False

        if E['Type'] == 'RPC':
            dst_node = (E['RPID'], E['RProcName'])
            traced_nodes.add(dst_node)

            if src_node not in pid_time_map:
                pid_time_map[dst_node] = E['Timestamp']
            else:
                if E['Timestamp'] >= (pid_time_map[src_node] + tws):
                    is_noise = True
                    pid_time_map[dst_node] = pid_time_map[src_node]
                else:
                    pid_time_map[dst_node] = E['Timestamp']

        elif E['Type'] == 'CreateProcess':
            dst_node = (E['CPID'], E['CProcName'])
            traced_nodes.add(dst_node)

            if src_node in pid_time_map:
                if E['Timestamp'] >= (pid_time_map[src_node] + tws):
                    is_noise = True
                    pid_time_map[dst_node] = pid_time_map[src_node]

        else:
            if src_node in pid_time_map:
                if E['Timestamp'] >= (pid_time_map[src_node] + tws):
                    is_noise = True

        if not is_noise:
            filtered_events.append(E)

    return filtered_events


SEQUENCE_VOCAB = frozenset({
    "ChildProcess", "CombinedFile", "NetworkConnection", "ProgramsFile", "RpcCall",
    "SystemFile", "UserFile", "create", "delete", "instant_process", "ip", "read",
    "system_process", "user_process", "write",
})


def abstract_subject_v15(principal, image_path):
    """Classify a process as system_process / user_process / instant_process."""
    principal = str(principal).upper()
    image_path = str(image_path).upper()
    if "AUTHORITY\\SYSTEM" in principal or "NETWORK SERVICE" in principal or "LOCAL SERVICE" in principal:
        return "system_process"
    if "C:\\WINDOWS" in image_path:
        return "system_process"
    if "PROGRAM FILES" in image_path or "PROGRAMDATA" in image_path:
        return "user_process"
    if "C:\\USERS" in image_path:
        return "user_process"
    if image_path.endswith(".DLL") or image_path.endswith(".SYS"):
        return "system_process"
    if "POWERSHELL" in image_path or "CMD" in image_path or "TASKHOSTW" in image_path:
        return "instant_process"
    return "user_process"


def abstract_object_file_v15(file_path):
    """Classify a file path into SystemFile / UserFile / ProgramsFile / CombinedFile."""
    fp = str(file_path).upper()
    if fp.endswith(".DLL") or fp.endswith(".SYS") or fp.endswith(".SO"):
        return "SystemFile"
    if "\\DEVICE\\HARDDISKVOLUME" in fp or "\\$" in fp or "SYSTEM VOLUME INFORMATION" in fp or "C:\\WINDOWS" in fp:
        return "SystemFile"
    if "C:\\USERS" in fp:
        return "UserFile"
    if "PROGRAM FILES" in fp or "PROGRAMDATA" in fp:
        return "ProgramsFile"
    return "CombinedFile"


def _append_vocab_line(lemmatized_sequence, subject, action, obj):
    if subject in SEQUENCE_VOCAB and action in SEQUENCE_VOCAB and obj in SEQUENCE_VOCAB:
        lemmatized_sequence.append(f"{subject} {action} {obj}")


def lemmatize_events(filtered_events, pid_history_path=None):
    lemmatized_sequence = []
    pid_history_path = pid_history_path or {}
    for evt in filtered_events:
        if evt['Type'] == 'RPC':
            raw = evt['Raw']
            ts = raw.get('timestamp')
            src_pid = raw.get('src_pid')
            dst_pid = raw.get('dst_pid')
            src_path = resolve_image_path(src_pid, ts, pid_history_path) if pid_history_path else ""
            dst_path = resolve_image_path(dst_pid, ts, pid_history_path) if pid_history_path else ""
            if not src_path:
                src_path = raw.get('src_proc', '')
            if not dst_path:
                dst_path = raw.get('dst_proc', '')
            subject = abstract_subject_v15("NT AUTHORITY\\SYSTEM", src_path)
            obj = abstract_subject_v15("NT AUTHORITY\\SYSTEM", dst_path)
            _append_vocab_line(lemmatized_sequence, subject, "RpcCall", obj)
            continue

        raw = evt['Raw']
        obj_type = raw.get('object', '')
        if obj_type not in ["FILE", "PROCESS", "FLOW", "MODULE"]:
            continue

        action_raw = raw.get('action', '')
        props = raw.get('properties', {})
        principal = raw.get('principal', '')
        action, subject, obj = "", "", ""

        if obj_type in ["FILE", "MODULE"]:
            image_path = props.get('image_path', '')
            subject = abstract_subject_v15(principal, image_path)
            file_path = props.get('file_path') or props.get('module_path', '')
            obj = abstract_object_file_v15(file_path)
            if action_raw == "CREATE":
                action = "create"
            elif action_raw == "DELETE":
                action = "delete"
            elif action_raw in ["READ", "LOAD"]:
                action = "read"
            elif action_raw in ["MODIFY", "WRITE", "RENAME"]:
                action = "write"
            else:
                continue

        elif obj_type == "FLOW":
            image_path = props.get('image_path', '')
            subject = abstract_subject_v15(principal, image_path)
            if action_raw in ["START", "MESSAGE"]:
                action = "NetworkConnection"
                obj = "ip"
            else:
                continue

        elif obj_type == "PROCESS":
            if action_raw == "CREATE":
                action = "ChildProcess"
                parent_path = props.get('parent_image_path', '')
                subject = abstract_subject_v15(principal, parent_path)
                tgt_path = props.get('image_path', '')
                obj = abstract_subject_v15(principal, tgt_path)
            else:
                continue

        if action and subject and obj:
            _append_vocab_line(lemmatized_sequence, subject, action, obj)

    return lemmatized_sequence


# ==========================================
# 9. Graph Traceback Engine
# ==========================================
class TracebackEngine:
    def __init__(self, json_events, rpc_events):
        self.rpc_events = rpc_events
        self.proc_create_map = defaultdict(list)
        self.file_create_map = defaultdict(list)
        self.actor_events = defaultdict(list)
        self.network_events = []
        self.registry_events = []

        for evt in json_events:
            act, obj, props = evt.get('action'), evt.get('object'), evt.get('properties', {})
            if obj == 'PROCESS' and act == 'CREATE':
                pid, ppid = evt.get('pid'), evt.get('ppid')
                if pid:
                    self.proc_create_map[pid].append(evt)
                if ppid:
                    self.actor_events[ppid].append(evt)
            else:
                actor_pid = evt.get('pid')
                if actor_pid:
                    self.actor_events[actor_pid].append(evt)
                if obj == 'FILE' and act == 'CREATE':
                    self.file_create_map[normalize(props.get('file_path'))].append(evt)

            if 'src_ip' in props or 'dest_ip' in props:
                self.network_events.append(evt)
            if obj == 'REGISTRY':
                self.registry_events.append(evt)

    def _get_closest_event(self, elist, ttime):
        valid = [e for e in elist if e['parsed_timestamp'] <= ttime]
        return max(valid, key=lambda x: x['parsed_timestamp']) if valid else None

    def _get_rpc_callers_filtered(self, target_time, visited, dynamic_mal_paths, window=0.1):
        callers = set()
        for rpc in self.rpc_events:
            if target_time - window <= rpc['timestamp'] <= target_time:
                caller = rpc['src_pid']
                proc_name = rpc['src_proc']

                if caller in visited:
                    continue
                if is_whitelisted(proc_name):
                    visited.add(caller)
                    continue

                aevts = self.actor_events.get(caller, [])
                cevt = self._get_closest_event(aevts, rpc['timestamp'])
                path = ""
                if cevt:
                    is_proc_create = (cevt.get('object') == 'PROCESS' and cevt.get('action') == 'CREATE')
                    path = normalize(
                        cevt.get('properties', {}).get('parent_image_path')
                        if is_proc_create
                        else cevt.get('properties', {}).get('image_path')
                    )

                if is_whitelisted(path):
                    visited.add(caller)
                    continue

                if path and path in dynamic_mal_paths:
                    callers.add(caller)
                else:
                    visited.add(caller)

        return callers

    def resolve_threat_graph(self, initial_pids, initial_paths,
                              pre_visited=None, pre_paths=None):
        master_queue = list(initial_pids)
        visited_pids = set(pre_visited or set()) | set(initial_pids)
        dynamic_malicious_paths = set(pre_paths or set()) | set(initial_paths)

        def safe_add_new_path(p_str, collector_set):
            if p_str and not is_whitelisted(p_str):
                collector_set.add(p_str)

        while master_queue:
            current_pid = master_queue.pop(0)
            new_paths_this_iter = set()

            for cevt in self.proc_create_map.get(current_pid, []):
                cts = cevt['parsed_timestamp']
                props = cevt.get('properties', {})
                ppid, pimage = cevt.get('ppid'), normalize(props.get('parent_image_path'))
                safe_add_new_path(pimage, new_paths_this_iter)

                if any(sb in pimage for sb in SYSTEM_PROCESS_BOUNDARY):
                    callers = self._get_rpc_callers_filtered(cts, visited_pids, dynamic_malicious_paths)
                    for c in callers:
                        visited_pids.add(c)
                        master_queue.append(c)
                elif ppid and ppid not in visited_pids and not is_whitelisted(pimage):
                    visited_pids.add(ppid)
                    master_queue.append(ppid)

                cimage = normalize(props.get('image_path'))
                safe_add_new_path(cimage, new_paths_this_iter)

                fevt = self._get_closest_event(self.file_create_map[cimage], cts)
                if fevt:
                    fpid = fevt.get('pid')
                    fimage = normalize(fevt.get('properties', {}).get('image_path'))
                    safe_add_new_path(fimage, new_paths_this_iter)

                    if any(sb in fimage for sb in SYSTEM_PROCESS_BOUNDARY):
                        callers = self._get_rpc_callers_filtered(fevt['parsed_timestamp'], visited_pids, dynamic_malicious_paths)
                        for c in callers:
                            visited_pids.add(c)
                            master_queue.append(c)
                    elif fpid and fpid not in visited_pids and not is_whitelisted(fimage):
                        visited_pids.add(fpid)
                        master_queue.append(fpid)

            for evt in self.actor_events.get(current_pid, []):
                obj, act, props = evt.get('object'), evt.get('action'), evt.get('properties', {})
                if obj == 'PROCESS' and act == 'CREATE':
                    child_pid, child_image = evt.get('pid'), normalize(props.get('image_path'))
                    safe_add_new_path(child_image, new_paths_this_iter)
                    if child_pid and child_pid not in visited_pids and not is_whitelisted(child_image):
                        visited_pids.add(child_pid)
                        master_queue.append(child_pid)
                elif obj == 'FILE' and act == 'CREATE':
                    fpath = normalize(props.get('file_path'))
                    safe_add_new_path(fpath, new_paths_this_iter)
                elif obj == 'SERVICE':
                    svc_path = normalize(props.get('image_path') or props.get('service_path', ''))
                    safe_add_new_path(svc_path, new_paths_this_iter)

            effective_new_paths = {p for p in new_paths_this_iter if len(p) > 5 and p not in dynamic_malicious_paths}
            if effective_new_paths:
                for evt in self.registry_events:
                    r_props = evt.get('properties', {})
                    val_norm = normalize(
                        f"{r_props.get('value', '')} | {r_props.get('target_name', '')} | {r_props.get('key', '')}")
                    if not val_norm or is_whitelisted(val_norm):
                        continue
                    r_image_path = normalize(r_props.get('image_path'))
                    if is_whitelisted(r_image_path):
                        continue
                    for mp in effective_new_paths:
                        if mp in val_norm:
                            r_pid = evt.get('pid')
                            if r_pid and r_pid not in visited_pids and r_pid not in master_queue:
                                master_queue.append(r_pid)
                                visited_pids.add(r_pid)
                                break
            dynamic_malicious_paths.update(effective_new_paths)

        return visited_pids, dynamic_malicious_paths


# ==========================================
# 10. UUID Comparison Metrics
# ==========================================
def load_uuid_set(path: str) -> set:
    """Load UUIDs from a file, one per line."""
    if not path or not os.path.isfile(path):
        return set()
    out = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            u = line.strip()
            if u:
                out.add(u)
    return out


def uuid_retrieval_metrics(ground_truth: set, predicted: set) -> dict:
    """
    Compute retrieval metrics using baseline output as ground truth.
    - Precision = |GT & Pred| / |Pred|
    - Recall    = |GT & Pred| / |GT|
    - F1
    - Accuracy (Jaccard) = |GT & Pred| / |GT | Pred|
    """
    tp = len(ground_truth & predicted)
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)
    union = tp + fp + fn

    if not ground_truth and not predicted:
        precision = recall = f1 = accuracy = 1.0
    else:
        precision = tp / len(predicted) if predicted else (0.0 if ground_truth else 1.0)
        recall = tp / len(ground_truth) if ground_truth else 1.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        accuracy = tp / union if union > 0 else 1.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "n_gt": len(ground_truth), "n_pred": len(predicted),
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
    }


def compute_scenario_metrics(baseline_path, ground_truth, predicted):
    return uuid_retrieval_metrics(ground_truth, predicted)


# ==========================================
# 11. ML Pipeline — Clean + Judge
# ==========================================
class CustomEncoderDecoderModel(PreTrainedModel):
    def __init__(self, encoder, decoder, vocab_size):
        config = decoder.config
        config.is_decoder = True
        config.add_cross_attention = True
        config.vocab_size = vocab_size
        config.pad_token_id = 0
        config.bos_token_id = 1
        config.eos_token_id = 2
        super().__init__(config)
        self.encoder = encoder
        self.decoder = decoder
        self.lm_head = torch.nn.Linear(decoder.config.hidden_size, vocab_size)
        self.config = config

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                encoder_hidden_states=None, encoder_attention_mask=None, **kwargs):
        if encoder_hidden_states is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            encoder_hidden_states = encoder_outputs.last_hidden_state
            encoder_attention_mask = attention_mask

        decoder_input = decoder_input_ids if decoder_input_ids is not None else kwargs.get("labels", None)

        decoder_outputs = self.decoder(
            input_ids=decoder_input,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            **kwargs,
        )
        sequence_output = decoder_outputs.last_hidden_state
        logits = self.lm_head(sequence_output)

        return Seq2SeqLMOutput(logits=logits, past_key_values=None)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        max_length = self.encoder.config.max_position_embeddings
        if input_ids.size(1) > max_length:
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]

        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        batch_size = input_ids.shape[0]
        decoder_input_ids = torch.full(
            (batch_size, 1), self.config.bos_token_id, dtype=torch.long, device=input_ids.device
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_outputs.last_hidden_state,
            "encoder_attention_mask": attention_mask,
        }


def tokenize_sequence(seq, vocab, max_length):
    tokens = seq.split()
    token_ids = [vocab.get(token, vocab["[UNK]"]) for token in tokens]
    token_ids = [vocab["[CLS]"]] + token_ids[: max_length - 2] + [vocab["[SEP]"]]
    padding = [vocab["[PAD]"]] * (max_length - len(token_ids))
    attention_mask = [1] * len(token_ids) + [0] * (max_length - len(token_ids))
    return token_ids + padding, attention_mask


class SeqCleanJudgePipeline:
    """Single sequence: Clean(generate) -> Judge(RF); malicious iff predict==1."""

    def __init__(self):
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        vocab_size = len(self.vocab)

        encoder_config = BertConfig(
            vocab_size=vocab_size, hidden_size=128, num_hidden_layers=2,
            num_attention_heads=2, max_position_embeddings=5000,
        )
        decoder_config = BertConfig(
            vocab_size=vocab_size, hidden_size=128, num_hidden_layers=2,
            num_attention_heads=2, is_decoder=True, add_cross_attention=True,
            max_position_embeddings=5000,
        )

        self.clean_model = CustomEncoderDecoderModel(
            encoder=BertModel(encoder_config), decoder=BertModel(decoder_config), vocab_size=vocab_size
        )

        safetensors_path = os.path.join(CLEAN_MODEL_DIR, "model.safetensors")
        bin_path = os.path.join(CLEAN_MODEL_DIR, "pytorch_model.bin")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            self.clean_model.load_state_dict(load_file(safetensors_path, device=DEVICE.type))
        elif os.path.exists(bin_path):
            self.clean_model.load_state_dict(torch.load(bin_path, map_location=DEVICE))
        else:
            raise FileNotFoundError(f"SeqClean model weights not found: {CLEAN_MODEL_DIR}")

        self.clean_model.to(DEVICE)
        self.clean_model.eval()

        self.judge_model = joblib.load(JUDGE_MODEL_PATH)
        self.vectorizer = joblib.load(VECTORIZER_PATH)

    def is_malicious(self, raw_text: str) -> bool:
        text = raw_text if raw_text is not None else ""
        if not text.strip():
            text = "[PAD]"
        input_ids, attention_mask = tokenize_sequence(text, self.vocab, MAX_SEQ_LENGTH)
        input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
        attention_mask_t = torch.tensor([attention_mask], dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            generated_ids = self.clean_model.generate(
                input_ids=input_ids_t, attention_mask=attention_mask_t,
                max_new_tokens=MAX_NEW_TOKENS, num_beams=2, early_stopping=True,
            )

        pred_ids = generated_ids[0].tolist()
        pad_id, cls_id, sep_id = self.vocab["[PAD]"], self.vocab["[CLS]"], self.vocab["[SEP]"]
        tokens = [
            self.inv_vocab.get(tid, "[UNK]")
            for tid in pred_ids
            if tid not in (pad_id, cls_id, sep_id)
        ]
        cleaned_text = " ".join(tokens).strip() or "[PAD]"

        X_vec = self.vectorizer.transform([cleaned_text])
        pred = self.judge_model.predict(X_vec)[0]
        return int(pred) == 1


def try_load_pipeline():
    """Load ML models (SeqClean + SeqJudge). Raises on failure."""
    try:
        return SeqCleanJudgePipeline()
    except Exception as e:
        raise RuntimeError(
            f"SeqClean + SeqJudge model loading failed: {e}\n"
            f"Ensure model files exist under {_SEQ_CLEAN_ROOT} and {_SEQ_JUDGE_ROOT}."
        ) from e


def sequence_text_for_pid(unified_stream, pid_hist_path, pid, proc_name_hint: str) -> str:
    p_name = extract_proc_name(proc_name_hint)
    filtered_subgraph = extract_and_filter_subgraph_onthefly(
        unified_events=unified_stream, start_pid=pid, start_proc_name=p_name, tws=SUBGRAPH_TWS,
    )
    lemmatized = lemmatize_events(filtered_subgraph, pid_hist_path)
    return "\n".join(lemmatized)


# ==========================================
# 12. Eval Traceback Engine
# ==========================================
class TracebackEngineEval(TracebackEngine):
    def __init__(self, json_events, rpc_events, unified_stream, pid_hist_path,
                 pipeline: "SeqCleanJudgePipeline"):
        super().__init__(json_events, rpc_events)
        self._unified_stream = unified_stream
        self._pid_hist_path = pid_hist_path
        self._pipeline = pipeline

    def _get_rpc_callers_filtered(self, target_time, visited, dynamic_mal_paths, window=RPC_TIME_WINDOW):
        callers = set()
        for rpc in self.rpc_events:
            if target_time - window <= rpc["timestamp"] <= target_time:
                caller = rpc["src_pid"]
                proc_name = rpc["src_proc"]

                if caller in visited:
                    continue
                if is_whitelisted(proc_name):
                    visited.add(caller)
                    continue

                aevts = self.actor_events.get(caller, [])
                cevt = self._get_closest_event(aevts, rpc["timestamp"])
                path = ""
                if cevt:
                    is_proc_create = cevt.get("object") == "PROCESS" and cevt.get("action") == "CREATE"
                    path = normalize(
                        cevt.get("properties", {}).get("parent_image_path")
                        if is_proc_create
                        else cevt.get("properties", {}).get("image_path")
                    )

                if is_whitelisted(path):
                    visited.add(caller)
                    continue

                text = sequence_text_for_pid(
                    self._unified_stream, self._pid_hist_path, caller, proc_name
                )
                if self._pipeline.is_malicious(text):
                    callers.add(caller)
                else:
                    visited.add(caller)

        return callers


# ==========================================
# 13. Scenario Processing & Main
# ==========================================
def process_scenario(scenario_name, json_log_path, rpc_log_path):
    reset_trackers()

    loaded = load_or_parse_logs(json_log_path, rpc_log_path)
    if not loaded or loaded[0] is None:
        return None
    json_events, start_dt, pid_hist, pid_hist_path, raw_rpcs = loaded
    if not json_events:
        return None

    pipeline = try_load_pipeline()
    filtered_rpcs, unique_initiators = filter_rpc_events(raw_rpcs)
    unified_stream = build_unified_event_stream(json_events, raw_rpcs, pid_hist)

    engine = TracebackEngineEval(
        json_events, filtered_rpcs, unified_stream, pid_hist_path, pipeline,
    )

    initial_malicious_pids = set()
    initial_malicious_paths = set()

    for (proc, pid), info in unique_initiators.items():
        aevts = engine.actor_events.get(pid, [])
        cevt = engine._get_closest_event(aevts, info["first_ts"])
        path = ""
        if cevt:
            props = cevt.get("properties", {})
            is_proc_create = cevt.get("object") == "PROCESS" and cevt.get("action") == "CREATE"
            path = normalize(
                props.get("parent_image_path") if is_proc_create else props.get("image_path")
            )

        text = sequence_text_for_pid(unified_stream, pid_hist_path, pid, proc)
        if pipeline.is_malicious(text):
            initial_malicious_pids.add(pid)
            if path:
                initial_malicious_paths.add(path)

    if not initial_malicious_pids:
        return None

    all_suspicious_pids, dynamic_mal_paths = engine.resolve_threat_graph(
        list(initial_malicious_pids), initial_malicious_paths
    )

    discovered_ips = set()
    for evt in engine.network_events:
        pid = evt.get('pid')
        if pid not in all_suspicious_pids:
            continue
        props = evt.get('properties', {})
        for ip_field in ('src_ip', 'dest_ip'):
            ip = props.get(ip_field)
            if ip and ip not in TRIVIAL_IPS:
                discovered_ips.add(ip)

    if discovered_ips:
        new_seeds = []
        new_seed_paths = set()
        for evt in engine.network_events:
            props = evt.get('properties', {})
            sip, dip = props.get('src_ip'), props.get('dest_ip')
            if sip not in discovered_ips and dip not in discovered_ips:
                continue
            pid = evt.get('pid')
            if not pid or pid in all_suspicious_pids:
                continue
            image_path = normalize(props.get('image_path'))
            if not image_path:
                cevt = engine._get_closest_event(
                    engine.actor_events.get(pid, []), evt['parsed_timestamp'])
                if cevt:
                    image_path = normalize(cevt.get('properties', {}).get('image_path'))
            if is_whitelisted(image_path):
                continue
            all_suspicious_pids.add(pid)
            new_seeds.append(pid)
            if image_path:
                new_seed_paths.add(image_path)

        if new_seeds:
            extra_pids, extra_paths = engine.resolve_threat_graph(
                new_seeds, new_seed_paths,
                pre_visited=all_suspicious_pids, pre_paths=dynamic_mal_paths,
            )
            all_suspicious_pids |= extra_pids
            dynamic_mal_paths |= extra_paths

    found_ids = set()
    malicious_ip_set = discovered_ips

    for evt in json_events:
        action = evt.get("action")
        obj_type = evt.get("object")
        actor_id = evt.get("actorID")
        object_id = evt.get("objectID")
        props = evt.get("properties", {})

        actor_pid = evt.get("ppid") if (obj_type == "PROCESS" and action == "CREATE") else evt.get("pid")
        if actor_pid not in all_suspicious_pids:
            continue

        if obj_type == "PROCESS" and action == "CREATE":
            norm_parent = normalize(props.get("parent_image_path", ""))
            if norm_parent and norm_parent in dynamic_mal_paths:
                if actor_id and str(actor_id).strip().upper() != "NULL" and actor_id not in found_ids:
                    found_ids.add(actor_id)

            norm_child = normalize(props.get("image_path", ""))
            if norm_child and norm_child in dynamic_mal_paths:
                if object_id and str(object_id).strip().upper() != "NULL" and object_id not in found_ids:
                    found_ids.add(object_id)
        else:
            norm_image = normalize(props.get("image_path", ""))
            if norm_image and norm_image in dynamic_mal_paths:
                if actor_id and str(actor_id).strip().upper() != "NULL" and actor_id not in found_ids:
                    found_ids.add(actor_id)

        norm_file = normalize(props.get("file_path", ""))
        if norm_file and norm_file in dynamic_mal_paths:
            if object_id and str(object_id).strip().upper() != "NULL" and object_id not in found_ids:
                found_ids.add(object_id)

        src_ip = props.get("src_ip")
        dest_ip = props.get("dest_ip")
        hit_ip = None
        if src_ip in malicious_ip_set:
            hit_ip = src_ip
        elif dest_ip in malicious_ip_set:
            hit_ip = dest_ip
        if hit_ip and object_id and str(object_id).strip().upper() != "NULL" and object_id not in found_ids:
            found_ids.add(object_id)

    base_name = scenario_name.split("-")[0]
    output_filename = f"{base_name}-MaliciousUUIDs-Eval.txt"
    output_path = os.path.join(os.path.dirname(json_log_path), output_filename)
    baseline_filename = f"{base_name}-MaliciousUUIDs.txt"
    baseline_path = os.path.join(os.path.dirname(json_log_path), baseline_filename)

    with open(output_path, "w", encoding="utf-8") as f_out:
        for muuid in found_ids:
            f_out.write(f"{muuid}\n")

    ground_truth = load_uuid_set(baseline_path)
    return compute_scenario_metrics(baseline_path, ground_truth, found_ids)


def main():
    ensure_dataset()

    SCENARIOS = [
        ("Data-Attack1", "Data-Attack1/Attack1.txt", "Data-Attack1/rpc_trace.txt"),
        ("Data-Attack2", "Data-Attack2/Attack2.txt", "Data-Attack2/rpc_trace.txt"),
        ("Data-Attack3", "Data-Attack3/Attack3.txt", "Data-Attack3/rpc_trace.txt"),
        ("Data-Attack4", "Data-Attack4/Attack4.txt", "Data-Attack4/rpc_trace.txt"),
        ("Data-Attack5", "Data-Attack5/Attack5.txt", "Data-Attack5/rpc_trace.txt"),
        ("Data-Attack6", "Data-Attack6/Attack6.txt", "Data-Attack6/rpc_trace.txt"),
    ]

    total = len(SCENARIOS)
    all_metrics = []
    bar_len = 30

    for idx, (scenario_name, json_rel, rpc_rel) in enumerate(SCENARIOS):
        filled = int(bar_len * idx / total)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\r  [{bar}] {idx}/{total} Processing {scenario_name}...", end="", flush=True)

        json_path = os.path.join(_DATA_DIR, json_rel)
        rpc_path = os.path.join(_DATA_DIR, rpc_rel)
        m = process_scenario(scenario_name, json_path, rpc_path)

        if m is not None:
            all_metrics.append(m)

    print(f"\r  [{'#' * bar_len}] {total}/{total} Done.{' ' * 30}")

    if all_metrics:
        n = len(all_metrics)
        avg_p = sum(x["precision"] for x in all_metrics) / n
        avg_r = sum(x["recall"] for x in all_metrics) / n
        avg_f1 = sum(x["f1"] for x in all_metrics) / n
        avg_acc = sum(x["accuracy"] for x in all_metrics) / n
        print(f"  Macro-average ({n} scenarios): Precision={avg_p:.4f}  Recall={avg_r:.4f}  F1={avg_f1:.4f}  Accuracy(Jaccard)={avg_acc:.4f}")


if __name__ == "__main__":
    main()
