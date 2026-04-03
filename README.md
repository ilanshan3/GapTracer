# ElasticBasedCollection

This repository is the source code for the paper "GapTracer: Unraveling RPC Obfuscations in Provenance Graphs for Attack Source Tracing".

## Project Structure

```
├── ElasticBasedDataCollection/
│   ├── ElasticBasedDataCollection.py   # Data collection & merging pipeline
│   ├── config.yaml.example             # Configuration template (copy to config.yaml)
│   ├── ProcmonCsvLog/                  # (gitignored) Place Procmon CSV exports here
│   ├── FinalLog/                       # (gitignored) Merged output JSON Lines
│   └── ObRPC-onDataset/               # Pre-collected attack scenario datasets
│       ├── Data-Attack1/
│       ├── Data-Attack2/
│       ├── Data-Attack3/
│       ├── Data-Attack4/
│       ├── Data-Attack5/
│       └── Data-Attack6/
├── GapTracerEvaluation/
│   └── GapTracerEvaluation.py          # Attack tracing evaluation script
├── SeqCleanModel/                      # BERT encoder-decoder for sequence cleaning
│   ├── ModelTrain.py                   # Training script
│   ├── vocab.json
│   └── final_model/
├── SeqJudgeModel/                      # Random Forest for sequence judgment
│   ├── ModelTrain.py                   # Training script
│   ├── rf_ngram_model.pkl
│   └── ngram_vectorizer.pkl
├── ObfuscationCase/                   # Annotated figures from the paper (Fig. 1)
│   ├── Fig.1-Part1.png
│   ├── Fig.2-Part2.png
│   └── Detail-Information.csv
├── requirements.txt
└── LICENSE
```

## Overview

### ElasticBasedDataCollection

Collects and merges behavioral logs from two sources:

1. Reads a Procmon CSV log file (with interactive file selection)
2. Queries Elasticsearch for matching IDS events in the same time range
3. Normalizes both sources into a unified log schema
4. Merges and sorts all logs chronologically
5. Runs a stateful ID generation pass (deterministic UUIDs for actors, objects, and events)
6. Outputs a single JSON Lines file

### GapTracerEvaluation

Evaluates GapTracer's attack tracing on the pre-collected `ObRPC-onDataset`:

1. Parses unified behavior logs (JSON Lines) and RPC trace files
2. Filters suspicious RPC initiators by frequency and whitelist
3. Identifies initial malicious PIDs via ML sequence judgment (SeqClean + SeqJudge)
4. Expands threat graph via bidirectional traversal (parent/child/file/RPC/registry)
5. Dynamically discovers malicious IPs from the threat cluster's network connections
6. Performs network lateral expansion to find additional processes connecting to discovered IPs
7. Runs second-round graph traversal for newly discovered seeds
8. Extracts malicious entity UUIDs (image/file/network) from the full threat graph
9. Compares extracted UUIDs against ground truth for Precision/Recall/F1/Jaccard

### ObfuscationCase

Provides enlarged, annotated versions of **Fig. 1** from the paper for closer inspection of the RPC obfuscation scenario. The original figure is split into two parts, with each process node numbered for reference.

| File | Description |
|------|-------------|
| `Fig.1-Part1.png` | Enlarged left portion of Fig. 1 with numbered process nodes |
| `Fig.2-Part2.png` | Enlarged right portion of Fig. 1 with numbered process nodes |
| `Detail-Information.csv` | Mapping from each number label to its PID and Process Name |

### SeqCleanModel & SeqJudgeModel

Two ML models used by GapTracerEvaluation for sequence-level maliciousness classification:

- **SeqCleanModel** — BERT encoder-decoder that denoises behavioral event sequences
- **SeqJudgeModel** — Random Forest classifier (with N-gram features) that judges cleaned sequences as benign or malicious

Pre-trained weights are included. To retrain, see each model's `ModelTrain.py` for usage.

## Prerequisites

- Python 3.8+

The following are only required if you want to collect your own datasets (see [Data Collection](#data-collection-optional)):

- Elasticsearch 8.x with the following data sources:
  - Elastic Endpoint (file, process, network, registry events)
  - Sysmon (via Winlogbeat)
  - Windows Security logs
  - PowerShell operational logs
  - Network traffic flow logs
- [Process Monitor](https://learn.microsoft.com/en-us/sysinternals/downloads/procmon) CSV exports

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run attack tracing evaluation against the pre-collected datasets:

```bash
python GapTracerEvaluation/GapTracerEvaluation.py
```

This processes all 6 attack scenarios in `ElasticBasedDataCollection/ObRPC-onDataset/` and reports Precision/Recall/F1/Jaccard metrics. Pre-trained SeqClean + SeqJudge model weights are included under `SeqCleanModel/` and `SeqJudgeModel/`.

### Data Collection (Optional)

If you want to collect your own datasets using Elastic IDS + Procmon:

3. Update `TRIVIAL_IPS` in `GapTracerEvaluation/GapTracerEvaluation.py` — replace `192.168.18.50` (the monitored host IP in the provided datasets) with your own host's IP so it is excluded from malicious IP discovery.

4. Create your configuration file:

```bash
cp ElasticBasedDataCollection/config.yaml.example ElasticBasedDataCollection/config.yaml
```

5. Edit `ElasticBasedDataCollection/config.yaml` with your Elasticsearch credentials and environment settings.

6. Place Procmon CSV files in `ElasticBasedDataCollection/ProcmonCsvLog/`.

   Filename format: `ProcmonLogfile-YYYY-MM-DD-Rx.csv` or `ProcmonLogfile-YYYY-MM-DD-YYYY-MM-DD-Rx.csv`

7. Run:

```bash
python ElasticBasedDataCollection/ElasticBasedDataCollection.py
```

8. Output will be written to `ElasticBasedDataCollection/FinalLog/` as JSON Lines.

## Dataset Structure

Each attack scenario folder (`Data-Attack*/`) contains:

| File | Description |
|------|-------------|
| `AttackN.txt` | Unified behavior log (JSON Lines), produced by `ElasticBasedDataCollection.py` |
| `rpc_trace.txt` | RPC call trace captured via ETW |
| `Data-MaliciousUUIDs.txt` | Ground-truth malicious entity UUIDs |

## Output Schema

Each line in the output JSON file contains:

| Field | Description |
|-------|-------------|
| `action` | Event action (CREATE, TERMINATE, READ, WRITE, etc.) |
| `actorID` | Deterministic UUID of the initiating entity |
| `hostname` | Source hostname |
| `id` | Unique event UUID |
| `object` | Object type (PROCESS, FILE, FLOW, REGISTRY, etc.) |
| `objectID` | Deterministic UUID of the target entity |
| `pid` | Process ID |
| `ppid` | Parent Process ID |
| `principal` | User principal (DOMAIN\Username) |
| `properties` | Event-specific properties |
| `tid` | Thread ID |
| `timestamp` | ISO 8601 timestamp |

## Configuration

`ElasticBasedDataCollection/config.yaml` is used only by the data collection script. See `config.yaml.example` in the same directory for a full template with comments.

Key sections:
- **elasticsearch** — Connection URL, credentials, TLS settings
- **host** — Target hostname, internal IP list, default principal
- **paths** — Input/output directory paths (relative to the script's location)
- **timezone_offset_hours** — UTC offset for timestamp conversion

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
