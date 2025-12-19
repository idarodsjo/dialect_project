#!/usr/bin/env python3
"""
parse_trs.py — Extract speaker intervals from Transcriber .trs files.

Outputs:
  - CSV of all intervals
  - Audacity label files per speaker
  - Optional RTTM per speaker
Usage:
  python parse_trs.py /path/to/file.trs --audio /path/to/file.wav --out outdir
"""

import argparse
import csv
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

def safe_float(s):
    try:
        return float(s)
    except Exception:
        return None

def normalize_speaker_list(attr_val):
    if not attr_val:
        return []
    # speakers attribute may be "spk1 spk2"
    return re.split(r"\s+", attr_val.strip())

def merge_intervals(intervals, merge_gap=0.05):
    """
    intervals: list of (start, end)
    Returns merged, sorted list.
    """
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e + merge_gap:  # merge if touching/very close
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged

def extract_text_snippet(elem, max_len=60):
    # get text visible inside the <Turn> ignoring tags
    text = "".join(elem.itertext()).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_len:
        return text[:max_len] + "…"
    return text

def parse_trs(trs_path, overlap_policy="assign_to_all", merge_gap=0.05):
    """
    overlap_policy: 'assign_to_all' | 'skip_overlaps'
    Returns:
      speakers_meta: dict id -> { 'name': str, ... }
      intervals: dict id -> list of (start, end, text_snippet)
    """
    # Transcriber files often are ISO-8859-1
    with open(trs_path, 'r', encoding='iso-8859-1', errors='ignore') as f:
        tree = ET.parse(f)

    root = tree.getroot()

    # Speakers metadata
    speakers_meta = {}
    spk_section = root.find('Speakers')
    if spk_section is not None:
        for spk in spk_section.findall('Speaker'):
            spk_id = spk.attrib.get('id')
            if spk_id:
                speakers_meta[spk_id] = {
                    'name': spk.attrib.get('name') or spk_id,
                    'dialect': spk.attrib.get('dialect', ''),
                    'accent': spk.attrib.get('accent', ''),
                    'scope': spk.attrib.get('scope', ''),
                }

    # Collect raw intervals
    intervals = {spk_id: [] for spk_id in speakers_meta.keys()}

    # Navigate Episode/Section/Turn
    for episode in root.findall('Episode'):
        for section in episode.findall('Section'):
            for turn in section.findall('Turn'):
                start = safe_float(turn.attrib.get('startTime'))
                end = safe_float(turn.attrib.get('endTime'))
                if start is None or end is None or end <= start:
                    continue

                spk_attr = turn.attrib.get('speaker', '').strip()
                spk_ids = normalize_speaker_list(spk_attr)

                # Some turns might be comments/noise w/o speaker
                if not spk_ids:
                    continue

                text_snip = extract_text_snippet(turn)

                if len(spk_ids) == 1:
                    intervals.setdefault(spk_ids[0], []).append((start, end, text_snip))
                else:
                    # Overlapping or multi-speaker turns.
                    if overlap_policy == "assign_to_all":
                        for sid in spk_ids:
                            intervals.setdefault(sid, []).append((start, end, text_snip))
                    elif overlap_policy == "skip_overlaps":
                        # Ignore ambiguous multi-speaker intervals
                        pass
                    else:
                        # default fallback: assign_to_all
                        for sid in spk_ids:
                            intervals.setdefault(sid, []).append((start, end, text_snip))

    # Merge per speaker
    merged = {}
    for sid, spans in intervals.items():
        base = [(s, e) for (s, e, _) in spans]
        merged_spans = merge_intervals(base, merge_gap=merge_gap)
        # Keep a representative snippet for each merged span (optional)
        merged[sid] = [(s, e) for (s, e) in merged_spans]

    return speakers_meta, intervals, merged

def write_csv(csv_path, intervals_dict):
    # Flatten to CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['speaker_id', 'start', 'end', 'duration', 'text_snippet'])
        for sid, spans in intervals_dict.items():
            for (s, e, snip) in spans:
                w.writerow([sid, f"{s:.3f}", f"{e:.3f}", f"{(e-s):.3f}", snip])

def write_audacity_labels(dir_path, merged_intervals, speakers_meta):
    os.makedirs(dir_path, exist_ok=True)
    for sid, spans in merged_intervals.items():
        label_path = Path(dir_path) / f"{speakers_meta.get(sid,{}).get('name',sid)}_{sid}.txt"
        with open(label_path, 'w', encoding='utf-8') as f:
            for s, e in spans:
                f.write(f"{s:.6f}\t{e:.6f}\t{sid}\n")

def write_rttm(dir_path, merged_intervals, audio_basename, speakers_meta):
    """
    RTTM fields: TYPE, FILE, CHAN, TB, DUR, ORTH, STYPE, NAME, CONF, SLAT
    We'll write SPEAKER lines: SPEAKER file 1 start duration <NA> <NA> speaker <NA> <NA>
    """
    os.makedirs(dir_path, exist_ok=True)
    for sid, spans in merged_intervals.items():
        out = Path(dir_path) / f"{audio_basename}.{sid}.rttm"
        with open(out, 'w', encoding='utf-8') as f:
            for s, e in spans:
                dur = e - s
                f.write(f"SPEAKER {audio_basename} 1 {s:.3f} {dur:.3f} <NA> <NA> {sid} <NA> <NA>\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trs", help="Path to .trs file")
    ap.add_argument("--audio", help="Path to matching audio file (for RTTM basename)")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--overlap", default="assign_to_all",
                    choices=["assign_to_all", "skip_overlaps"],
                    help="How to treat multi-speaker turns")
    ap.add_argument("--merge_gap", type=float, default=0.05,
                    help="Merge intervals closer than this (seconds)")
    ap.add_argument("--rttm", action="store_true", help="Write RTTM per speaker")
    args = ap.parse_args()

    speakers_meta, intervals, merged = parse_trs(
        args.trs,
        overlap_policy=args.overlap,
        merge_gap=args.merge_gap
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) CSV with raw (unmerged) intervals
    write_csv(out_dir / "all_intervals.csv", intervals)

    # 2) Audacity labels per speaker (merged)
    write_audacity_labels(out_dir / "labels", merged, speakers_meta)

    # 3) Optional RTTM
    if args.rttm:
        audio_base = Path(args.audio).stem if args.audio else Path(args.trs).stem
        write_rttm(out_dir / "rttm", merged, audio_base, speakers_meta)

    # 4) Also print a small summary
    print("Speakers found:")
    for sid, meta in speakers_meta.items():
        n = len(merged.get(sid, []))
        total = sum(e - s for (s, e) in merged.get(sid, []))
        print(f"  - {sid} ({meta['name']}): {n} merged spans, {total:.1f}s")

if __name__ == "__main__":
    main()