#!/usr/bin/env bash
set -euo pipefail

# collect_trs_topics.sh
#
# Walk all .trs files (excluding *.conc.ort.trs), extract <Topic> elements,
# and output a CSV with unique topics and their counts.
#
# Usage:
#   bash collect_trs_topics.sh /path/to/fonetisk-ortografisk-transkripsjoner-til-nedlasting [output_dir]
#
# Output:
#   output_dir/topics_summary.csv with columns:
#     topic_id, topic_desc, files_count, occurrences
#
# Notes:
#  - Handles ISO-8859-1 TRS and removes external DTD lines to parse safely.
#  - Deduplicates topics within a single file for files_count.
#  - occurrences counts every <Topic> element occurrence across files.

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/trs_dir [output_dir]" >&2
  exit 1
fi

TRS_DIR="${1%/}"
OUTDIR="${2:-dataset_stats}"
mkdir -p "$OUTDIR"

if [[ ! -d "$TRS_DIR" ]]; then
  echo "Error: TRS directory not found: $TRS_DIR" >&2
  exit 1
fi

export TRS_DIR OUTDIR

python3 - << 'PYCODE'
import os, re, csv, sys
import xml.etree.ElementTree as ET
from collections import defaultdict

TRS_DIR = os.environ["TRS_DIR"]
OUTDIR = os.environ["OUTDIR"]

def norm(s): 
    return (s or "").strip()

def parse_topics_from_trs(path):
    """Return list of (topic_id, topic_desc) from a TRS file."""
    with open(path, 'rb') as f:
        raw = f.read()
    text = None
    # Try common encodings for NDC TRS
    for enc in ('latin-1', 'utf-8', 'utf-8-sig'):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise UnicodeError(f"Cannot decode file {path}")
    # Strip external DTD to avoid network or parser failures
    text = re.sub(r'<!DOCTYPE[^>]*>', '', text, flags=re.IGNORECASE)
    root = ET.fromstring(text)

    topics = []
    topics_elem = root.find('Topics')
    if topics_elem is not None:
        for tp in topics_elem.findall('Topic'):
            tid  = norm(tp.attrib.get('id', ''))
            desc = norm(tp.attrib.get('desc', ''))
            topics.append((tid, desc))
    return topics

# Collect list of TRS files, excluding *.conc.ort.trs
trs_files = []
for rr, dd, ff in os.walk(TRS_DIR):
    for fn in ff:
        if not fn.lower().endswith('.trs'):
            continue
        if fn.lower().endswith('.conc.ort.trs'):
            continue
        trs_files.append(os.path.join(rr, fn))
trs_files.sort()

if not trs_files:
    print(f"[ERROR] No .trs files found under {TRS_DIR} (excluding *.conc.ort.trs)", file=sys.stderr)
    sys.exit(2)

# Global aggregation
topic_occurrences = defaultdict(int)     # (id, desc) -> total occurrences
topic_files = defaultdict(set)           # (id, desc) -> set of file paths (for dedup files_count)
warnings = []

for p in trs_files:
    try:
        topics = parse_topics_from_trs(p)
    except Exception as e:
        warnings.append(f"Failed to parse {p}: {e}")
        continue

    # Count all occurrences (every Topic element)
    for t in topics:
        topic_occurrences[t] += 1
    # Deduplicate per file for files_count
    for t in set(topics):
        topic_files[t].add(p)

# Prepare rows
rows = []
# Sort by descending files_count, then occurrences, then description
def sort_key(item):
    (tid, desc), _ = item
    return ( -len(topic_files[(tid, desc)]), -topic_occurrences[(tid, desc)], desc.lower() )

for (tid, desc), _ in sorted(topic_occurrences.items(), key=sort_key):
    rows.append({
        'topic_id': tid,
        'topic_desc': desc,
        'files_count': len(topic_files[(tid, desc)]),
        'occurrences': topic_occurrences[(tid, desc)],
    })

# Write CSV
out_path = os.path.join(OUTDIR, 'topics_summary.csv')
with open(out_path, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['topic_id','topic_desc','files_count','occurrences'])
    w.writeheader()
    for r in rows:
        w.writerow(r)

# Console summary
print(f"TRS files scanned: {len(trs_files)}")
print(f"Unique topics:     {len(rows)}")
print(f"Output:            {out_path}")

if warnings:
    print("\n[WARNINGS]")
    for w in warnings[:50]:
        print(" - " + w)
    if len(warnings) > 50:
        print(f" ... {len(warnings)-50} more")
PYCODE

echo "Done."