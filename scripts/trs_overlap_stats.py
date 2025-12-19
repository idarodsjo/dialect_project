#!/usr/bin/env python3
"""
trs_overlap_stats.py

Parse Transcriber .trs files to compute speaker overlap statistics and optionally
export per-speaker time intervals after applying a "minor-words" threshold rule.

Usage (examples):

  python trs_overlap_stats.py \
      --trs-dir /path/to/trs_files \
      --out-dir ./reports \
      --minor-words-threshold 2 \
      --max-files 100 \
      --export-segments

Outputs:
  - reports/summary.csv : one row per TRS file with overlap stats
  - reports/details/<basename>.segments.csv (if --export-segments): segments by speaker
  - reports/details/<basename>.meta.json: metadata per file (speakers, duration, counts)

Notes:
  * The script treats a Turn with multiple speakers (e.g., speaker="spk1 spk2") as overlap.
  * If <Who nb="…"> sub-spans exist, we count words per Who and:
      - If the total words spoken by all but the main speaker(s) <= minor-words-threshold,
        we assign the entire turn to the main speaker(s) (breaking the overlap).
      - Otherwise the full turn duration remains as overlapping among the listed speakers.
  * If a multi-speaker Turn lacks <Who> tags, it always counts as overlap.
  * Word counting ignores markup and counts tokens that contain at least one letter.

Compatible with Python 3.8+.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

# ---------- Utilities ----------

def read_text_robust(path: Path) -> str:
    """Read file trying UTF-8 then fallback to Latin-1 without crashing on DTD."""
    data = path.read_bytes()
    for enc in ("utf-8", "iso-8859-1", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")

# Use a Unicode range that covers Latin and extended letters (sufficient for Norwegian dialect tokens)
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-žȘșȚțÆæŒœƒẞß]+", re.UNICODE)


def tokenize_and_count(text: str) -> int:
    """Count word-like tokens (contain letters). Remove markup markers like '*' or '#' prefixes."""
    cleaned = re.sub(r"\[[^\]]*\]", " ", text)  # remove [noise]
    cleaned = re.sub(r"\([^)]*\)", " ", cleaned)  # remove (comments)
    cleaned = cleaned.replace("\u00A0", " ")
    tokens = re.split(r"\s+", cleaned.strip()) if cleaned.strip() else []
    count = 0
    for tok in tokens:
        tok2 = tok.lstrip("*#~-\u266A\u266B\u266C")
        if WORD_RE.search(tok2):
            count += 1
    return count


@dataclass
class TurnInfo:
    start: float
    end: float
    speakers: List[str]  # as listed in @speaker, may be []
    has_who: bool
    who_word_counts: Dict[str, int]  # speaker_id -> words (if who tags present)
    duration: float


@dataclass
class FileStats:
    file: str
    n_speakers: int
    speakers: List[str]
    total_turns: int
    total_duration: float
    overlap_turns_original: int
    overlap_sec_original: float
    overlap_turns_after_threshold: int
    overlap_sec_after_threshold: float
    unresolved_multi_no_who_turns: int
    unresolved_multi_no_who_sec: float
    assigned_due_to_threshold_turns: int
    words_threshold: int


# ---------- Core parsing ----------

def parse_trs_turns(xml_text: str) -> Tuple[Dict[str, str], List[TurnInfo]]:
    """Parse .trs XML into speakers mapping and turn list.

    Returns (speaker_id->name, [TurnInfo...])
    """
    xml_text_wo_dtd = re.sub(r"<!DOCTYPE[^>]*>", "", xml_text, flags=re.IGNORECASE)
    root = ET.fromstring(xml_text_wo_dtd)

    spk_map: Dict[str, str] = {}
    speakers_el = root.find("Speakers")
    if speakers_el is not None:
        for spk in speakers_el.findall("Speaker"):
            sid = spk.attrib.get("id")
            name = spk.attrib.get("name", sid)
            if sid:
                spk_map[sid] = name

    turns: List[TurnInfo] = []
    for section in root.findall(".//Section"):
        for turn in section.findall("Turn"):
            start = float(turn.attrib.get("startTime", "0") or 0.0)
            end = float(turn.attrib.get("endTime", "0") or 0.0)
            duration = max(0.0, end - start)
            speaker_attr = turn.attrib.get("speaker", "").strip()
            speakers = [s for s in speaker_attr.split() if s] if speaker_attr else []

            who_word_counts: Dict[str, int] = defaultdict(int)
            has_who = False

            nb_to_speaker: Dict[str, str] = {}
            if speakers:
                for idx, sid in enumerate(speakers, start=1):
                    nb_to_speaker[str(idx)] = sid

            current_nb: Optional[str] = None
            buffer_text_by_nb: Dict[str, List[str]] = defaultdict(list)

            if turn.text and turn.text.strip():
                buffer_text_by_nb[""] .append(turn.text)

            for child in list(turn):
                tag = child.tag
                if tag == "Who":
                    has_who = True
                    current_nb = child.attrib.get("nb")
                    if child.text:
                        buffer_text_by_nb[current_nb].append(child.text)
                    if child.tail:
                        buffer_text_by_nb[current_nb].append(child.tail)
                else:
                    if child.tail:
                        nb_key = current_nb if current_nb is not None else ""
                        buffer_text_by_nb[nb_key].append(child.tail)

            for nb_key, chunks in buffer_text_by_nb.items():
                if not chunks:
                    continue
                word_count = tokenize_and_count(" ".join(chunks))
                if nb_key and nb_key in nb_to_speaker:
                    who_word_counts[nb_to_speaker[nb_key]] += word_count
                else:
                    if len(speakers) == 1:
                        who_word_counts[speakers[0]] += word_count

            turns.append(
                TurnInfo(
                    start=start,
                    end=end,
                    speakers=speakers,
                    has_who=has_who,
                    who_word_counts=dict(who_word_counts),
                    duration=duration,
                )
            )

    return spk_map, turns


# ---------- Stats computation ----------

def compute_overlap_stats(spk_map: Dict[str, str], turns: List[TurnInfo], minor_words_threshold: int) -> Tuple[FileStats, Dict[str, List[Tuple[float, float]]]]:
    total_duration = 0.0
    total_turns = 0
    overlap_turns_original = 0
    overlap_sec_original = 0.0

    overlap_turns_after = 0
    overlap_sec_after = 0.0

    unresolved_multi_no_who_turns = 0
    unresolved_multi_no_who_sec = 0.0

    assigned_due_to_threshold_turns = 0

    segments_by_speaker: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    for t in turns:
        total_duration += t.duration
        total_turns += 1

        if len(t.speakers) <= 1:
            if t.speakers:
                segments_by_speaker[t.speakers[0]].append((t.start, t.end))
            continue

        overlap_turns_original += 1
        overlap_sec_original += t.duration

        if not t.has_who or not t.who_word_counts:
            unresolved_multi_no_who_turns += 1
            unresolved_multi_no_who_sec += t.duration
            overlap_turns_after += 1
            overlap_sec_after += t.duration
            continue

        counts = {sid: t.who_word_counts.get(sid, 0) for sid in t.speakers}
        max_words = max(counts.values())
        main_speakers = [sid for sid, wc in counts.items() if wc == max_words and wc > 0]
        other_words = sum(wc for sid, wc in counts.items() if sid not in main_speakers)

        if other_words <= minor_words_threshold and len(main_speakers) >= 1:
            if len(main_speakers) == 1:
                segments_by_speaker[main_speakers[0]].append((t.start, t.end))
                assigned_due_to_threshold_turns += 1
            else:
                overlap_turns_after += 1
                overlap_sec_after += t.duration
            continue
        else:
            overlap_turns_after += 1
            overlap_sec_after += t.duration

    stats = FileStats(
        file="",
        n_speakers=len(spk_map),
        speakers=list(spk_map.keys()),
        total_turns=total_turns,
        total_duration=total_duration,
        overlap_turns_original=overlap_turns_original,
        overlap_sec_original=overlap_sec_original,
        overlap_turns_after_threshold=overlap_turns_after,
        overlap_sec_after_threshold=overlap_sec_after,
        unresolved_multi_no_who_turns=unresolved_multi_no_who_turns,
        unresolved_multi_no_who_sec=unresolved_multi_no_who_sec,
        assigned_due_to_threshold_turns=assigned_due_to_threshold_turns,
        words_threshold=minor_words_threshold,
    )
    return stats, segments_by_speaker


def merge_adjacent_segments(segments: List[Tuple[float, float]], tol: float = 0.01) -> List[Tuple[float, float]]:
    if not segments:
        return []
    segments = sorted(segments)
    merged = [segments[0]]
    for s, e in segments[1:]:
        ps, pe = merged[-1]
        if s <= pe + tol:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


# ---------- CLI ----------

def process_directory(trs_dir: Path, out_dir: Path, minor_words_threshold: int, max_files: Optional[int], export_segments: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    details_dir = out_dir / "details"
    if export_segments:
        details_dir.mkdir(parents=True, exist_ok=True)

    
    trs_files = sorted([
        p for p in trs_dir.rglob("*.trs")
        if p.suffix == ".trs" and len(p.suffixes) == 1
    ])

    if max_files is not None:
        trs_files = trs_files[:max_files]

    summary_rows: List[Dict[str, object]] = []

    for idx, trs_path in enumerate(trs_files, start=1):
        try:
            xml_text = read_text_robust(trs_path)
            spk_map, turns = parse_trs_turns(xml_text)
            stats, segments = compute_overlap_stats(spk_map, turns, minor_words_threshold)
            stats.file = trs_path.name

            per_spk_sec = {}
            for sid, segs in segments.items():
                merged = merge_adjacent_segments(segs)
                segments[sid] = merged
                per_spk_sec[sid] = sum(e - s for s, e in merged)

            row = {
                "file": trs_path.name,
                "n_speakers_declared": stats.n_speakers,
                "total_turns": stats.total_turns,
                "total_duration_sec": round(stats.total_duration, 3),
                "overlap_turns_original": stats.overlap_turns_original,
                "overlap_sec_original": round(stats.overlap_sec_original, 3),
                "overlap_pct_original": round((stats.overlap_sec_original / stats.total_duration * 100) if stats.total_duration else 0.0, 2),
                "overlap_turns_after_threshold": stats.overlap_turns_after_threshold,
                "overlap_sec_after_threshold": round(stats.overlap_sec_after_threshold, 3),
                "overlap_pct_after_threshold": round((stats.overlap_sec_after_threshold / stats.total_duration * 100) if stats.total_duration else 0.0, 2),
                "unresolved_multi_no_who_turns": stats.unresolved_multi_no_who_turns,
                "unresolved_multi_no_who_sec": round(stats.unresolved_multi_no_who_sec, 3),
                "assigned_due_to_threshold_turns": stats.assigned_due_to_threshold_turns,
                "words_threshold": stats.words_threshold,
            }

            for sid in sorted(per_spk_sec.keys()):
                row[f"sec_{sid}"] = round(per_spk_sec[sid], 3)

            summary_rows.append(row)

            if export_segments:
                base = trs_path.stem
                meta = {
                    "file": trs_path.name,
                    "speakers": {sid: spk_map.get(sid, sid) for sid in spk_map},
                    "total_turns": stats.total_turns,
                    "total_duration_sec": stats.total_duration,
                    "words_threshold": minor_words_threshold,
                }
                (details_dir / f"{base}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

                seg_csv = details_dir / f"{base}.segments.csv"
                with seg_csv.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["speaker_id", "start_sec", "end_sec", "duration_sec"])
                    for sid in sorted(segments.keys()):
                        for s, e in segments[sid]:
                            writer.writerow([sid, f"{s:.3f}", f"{e:.3f}", f"{(e - s):.3f}"])

            print(f"[{idx}/{len(trs_files)}] Processed {trs_path.name}")

        except Exception as ex:
            print(f"Error processing {trs_path}: {ex}")

    summary_csv = out_dir / f"summary{minor_words_threshold}.csv"
    if summary_rows:
        fieldnames = []
        for row in summary_rows:
            for k in row.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        print(f"Wrote {summary_csv}")
    else:
        print("No TRS files processed; no summary written.")


def main():
    ap = argparse.ArgumentParser(description="Compute overlap stats from .trs files and export segments.")
    ap.add_argument("--trs-dir", type=str, required=True, help="Directory containing .trs files (recursively scanned)")
    ap.add_argument("--out-dir", type=str, default="./reports", help="Directory to write outputs")
    ap.add_argument("--minor-words-threshold", type=int, default=5, help="Allow up to this many words by non-main speakers to assign the turn to the main speaker")
    ap.add_argument("--max-files", type=int, default=645, help="Process at most N TRS files")
    ap.add_argument("--export-segments", action="store_true", help="Export per-speaker segments CSV for each file after thresholding")

    args = ap.parse_args()

    trs_dir = Path(args.trs_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not trs_dir.exists():
        raise SystemExit(f"TRS directory not found: {trs_dir}")

    process_directory(
        
        trs_dir=trs_dir,
        out_dir=out_dir,
        minor_words_threshold=args.minor_words_threshold,
        max_files=args.max_files,
        export_segments=args.export_segments,
    )


if __name__ == "__main__":
    main()