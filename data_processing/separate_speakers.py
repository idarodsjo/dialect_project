"""
Creates speaker-separated segments and concatenated waveforms using transcription files and metadata.
By default, all overlapping segments are removed. See project topic.
The output directory contains one subdirectory per speaker, which in turn contains one waveform per conversation. 
    - The original waveform file names are preserved, with a speaker ID suffix added.
CSV files are creates to have oversight of segments and then used for concatenation and train/val/test splits. 

Run:
  python separate_speakers.py

Outputs under OUT_ROOT:
  - segments.csv (raw segments incl. overlaps and non-metadata speakers)
  - segments_clean.csv (filtered, merged)
  - speakers.csv (per speaker totals + metadata)
  - splits_by_speaker.csv (train/val/test groups)
  - speaker_wavs/{speaker_tid}/{conversation_id}__{speaker_tid}.wav
"""

import os, re, csv, math, json, shutil, subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
import pandas as pd
from pathlib import Path
import wave

# define paths for NDC dataset
AUDIO_ROOT = "/home/idatro/preprocessed_NorDia/norske-lydfiler-fra-ndc-6"
TRS_ROOT   = "/home/idatro/talebase/data/speech_raw/NDC/fonetisk-ortografisk-transkripsjoner-til-nedlasting"
META_TSV   = "/home/idatro/talebase/data/speech_raw/NDC/NDC_norge_metadata.tsv"
OUT_ROOT   = "ndc_speaker_outputs"
WAV_OUT_DIR = os.path.join(OUT_ROOT, "speaker_wavs")
os.makedirs(WAV_OUT_DIR, exist_ok=True)


# utils
# !!!! USE FUNC BELOW TO NORMALIZE FILENAMES (NEED TO CORREPSOND FOR WAV AND TRS) !!!!
def norm_base_from_audio(fn: str) -> str:
    """
    Map a WAV filename to the TRS 'audio_filename' base.
    Example: 'aaseral_01_um-02uk.wav' -> 'aaseral_01um-02uk'
    """
    stem = os.path.splitext(fn)[0]
    # split first underscore: prefix + rest
    m = re.match(r"^([^-_]+)_(.+)$", stem)
    if not m:
        return stem
    prefix, tail = m.group(1), m.group(2)
    # tokens after the first '_' are separated by '-', normalize each by removing non-alnum
    parts = tail.split("-")
    parts = [re.sub(r"[^0-9A-Za-z]", "", p) for p in parts]  # remove underscores etc.
    return prefix + "_" + "-".join(parts)


def _strip_trs_suffixes(cid: str) -> list[str]:
    """
    Generate candidates by progressively stripping suspected trailing suffixes.
    e.g., 'bud_01um-ma' -> ['bud_01um-ma', 'bud_01um']
          'aaseral_03gm-eo_800' -> ['aaseral_03gm-eo_800', 'aaseral_03gm-eo', 'aaseral_03gm']
    """
    cands = [cid]
    cur = cid
    # Repeatedly strip trailing patterns like '-foo', '-foo_bar', or '_digits'
    while True:
        new = re.sub(r'[-_][A-Za-z]+(?:_[0-9]+)?$', '', cur)
        if new == cur:
            break
        cands.append(new)
        cur = new
    return cands

def _tokenize_tail(base: str) -> list[str]:
    """
    Return normalized speaker tokens from the part after the first underscore.
    'aaseral_01um-02uk' -> ['01um','02uk']
    """
    if "_" not in base:
        return []
    tail = base.split("_", 1)[1]
    parts = tail.split("-")
    return [re.sub(r"[^0-9A-Za-z]", "", p) for p in parts if p]

def resolve_wav_for_trs(conv_id: str, spkmap: dict, allowed_tids: set, audio_index: dict) -> str | None:
    """
    1) Try exact and stripped-suffix matches.
    2) If no hit, reconstruct using speakers listed in TRS:
       - determine place prefix from any speaker name (e.g., 'bud' in 'bud_01um')
       - collect local tokens of allowed speakers in TRS (e.g., ['01um','02uk'])
       - choose the WAV whose tokens contain (most of) those locals.
    """
    # 1) direct / stripped matches
    for cand in [conv_id] + _strip_trs_suffixes(conv_id):
        if cand in audio_index:
            return audio_index[cand]

    # 2) reconstruct from TRS speakers
    # derive prefix/place
    names = [v for v in spkmap.values() if v]
    prefix = None
    for nm in names:
        if "_" in nm:
            prefix = nm.split("_", 1)[0]
            break
    if not prefix:
        return None

    # allowed local tokens present in this TRS
    locals_in_trs = []
    for nm in names:
        # keep only speakers that are in metadata
        if nm in allowed_tids and "_" in nm:
            locals_in_trs.append(re.sub(r"[^0-9A-Za-z]", "", nm.split("_",1)[1]))

    # candidate WAV keys starting with 'prefix_'
    candidates = [k for k in audio_index.keys() if k.startswith(prefix + "_")]
    if not candidates:
        return None

    def score(k: str) -> tuple[int, int]:
        toks = set(_tokenize_tail(k))
        # score by (#matched tokens, total tokens)
        matched = sum(1 for t in locals_in_trs if t in toks)
        return (matched, len(toks))

    # pick the candidate with the highest (#matched, then more tokens)
    best = None
    best_sc = (-1, -1)
    for k in candidates:
        sc = score(k)
        if sc > best_sc:
            best_sc = sc; best = k

    # require at least one token match
    if best and best_sc[0] > 0:
        return audio_index[best]
    return None


def read_metadata_allowed_tids(meta_tsv):
    meta = pd.read_csv(meta_tsv, sep="\t")
    if "tid" not in meta.columns:
        raise ValueError("Metadata TSV must contain a 'tid' column")
    return set(meta["tid"].astype(str).tolist()), meta

def parse_trs_segments(trs_path, allowed_tids):
    """
    Parse TRS with <Sync> anchors. Returns:
    - segments: list of dicts with (conv_id, wav_base, start, end, speaker_tid, overlap_flag, text)
    - spkmap: dict spkid -> speaker_tid
    """
    tree = ET.parse(trs_path)
    root = tree.getroot()

    # conversation id from Trans@audio_filename
    trans = root
    conv_id = trans.attrib.get("audio_filename")  # e.g., 'aaseral_01um-02uk'
    if not conv_id:
        # fallback to trs filename stem
        conv_id = Path(trs_path).stem

    # Speakers map: spk id -> name (name already equals speaker_tid for native speakers)
    spkmap = {}
    for sp in root.findall(".//Speakers/Speaker"):
        spkid = sp.attrib.get("id")
        name  = (sp.attrib.get("name") or "").strip()
        spkmap[spkid] = name  # name is e.g., 'aaseral_01um' or 'sl'

    # Collect raw turn segments (split by Sync)
    raw = []
    for ti, turn in enumerate(root.findall(".//Turn")):
        t_start = float(turn.attrib.get("startTime", "nan"))
        t_end   = float(turn.attrib.get("endTime", "nan"))
        spk_attr = (turn.attrib.get("speaker") or "").strip()
        spk_ids = [x for x in re.split(r"[ \t;,+]+", spk_attr) if x]
        multi_speaker = (len(spk_ids) != 1)

        # Map to speaker_tid(s)
        speaker_tids = [spkmap.get(s, "") for s in spk_ids] if spk_ids else []
        # For text, ignore <Event> tags; harvest raw text nodes (simple approach)
        def text_from(elem):
            texts = []
            for e in elem.iter():
                if e.tag in ("Event", "Comment"):  # skip
                    continue
                if e.text:
                    texts.append(e.text)
            return " ".join([t.strip() for t in texts if t and t.strip()])

        # Collect Sync anchors
        syncs = [float(s.attrib.get("time")) for s in turn.findall("./Sync")]
        syncs = sorted(syncs)
        if not syncs:
            syncs = [t_start]
        # Build [t_i, t_{i+1}) ... last ends at t_end
        edges = syncs + [t_end]

        # Create sub-segments
        for i in range(len(edges)-1):
            s0, s1 = edges[i], edges[i+1]
            if not (math.isfinite(s0) and math.isfinite(s1)) or (s1 - s0) <= 0:
                continue
            txt = text_from(turn).strip()  # coarse: same text attached to all subs; ok for metadata
            if multi_speaker:
                # mark overlap; we don't keep it
                raw.append(dict(conv_id=conv_id, start=s0, end=s1, speaker_tid=None, overlap=True, text=txt))
            else:
                stid = speaker_tids[0] if speaker_tids else None
                raw.append(dict(conv_id=conv_id, start=s0, end=s1, speaker_tid=stid, overlap=False, text=txt))

    # Temporal overlap detection across different speakers
    # Sort and sweep; if two segments overlap in time and speakers differ -> mark overlap
    raw_sorted = sorted(raw, key=lambda r: (r["start"], r["end"]))
    for i in range(len(raw_sorted)-1):
        a, b = raw_sorted[i], raw_sorted[i+1]
        if a["end"] > b["start"] + 1e-6:  # overlap in time
            # Mark both as overlap unless they are the same speaker (we still keep same-speaker continuity)
            if a["speaker_tid"] != b["speaker_tid"]:
                a["overlap"] = True
                b["overlap"] = True

    # Filter to allowed speakers only (drop interviewer, etc.)
    segments = []
    for r in raw_sorted:
        if r["speaker_tid"] is None:
            segments.append({**r, "keep_reason": "multi_or_none"})
            continue
        if r["speaker_tid"] not in allowed_tids:
            segments.append({**r, "keep_reason": "not_in_metadata"})
            continue
        segments.append({**r, "keep_reason": "ok"})
    return segments, spkmap


def ffmpeg_extract(in_wav, start, end, out_wav):
    """Pure-Python PCM extractor (assumes 16 kHz mono PCM WAV)."""
    with wave.open(in_wav, 'rb') as r, wave.open(out_wav, 'wb') as w:
        nch = r.getnchannels()
        sw  = r.getsampwidth()
        sr  = r.getframerate()
        n   = r.getnframes()
        if nch != 1 or sr != 16000:
            raise ValueError(f"Expected mono/16k WAV. Got nch={nch}, sr={sr} for {in_wav}")
        w.setnchannels(nch); w.setsampwidth(sw); w.setframerate(sr)
        i0 = max(0, int(round(start * sr)))
        i1 = min(n, int(round(end   * sr)))
        if i1 <= i0:
            return
        r.setpos(i0)
        remaining = i1 - i0
        chunk = 4096
        while remaining > 0:
            to_read = min(chunk, remaining)
            frames = r.readframes(to_read)
            if not frames:
                break
            w.writeframes(frames)
            remaining -= to_read



def ffmpeg_concat(wav_list, out_wav):
    """Pure-Python concatenation (assumes all WAVs share params)."""
    if len(wav_list) == 1:
        shutil.copyfile(wav_list[0], out_wav); return
    with wave.open(out_wav, 'wb') as w_out:
        # set params from first file
        with wave.open(wav_list[0], 'rb') as r0:
            nch = r0.getnchannels()
            sw  = r0.getsampwidth()
            sr  = r0.getframerate()
            w_out.setnchannels(nch); w_out.setsampwidth(sw); w_out.setframerate(sr)
        # append frames
        for wpath in wav_list:
            with wave.open(wpath, 'rb') as r:
                if r.getnchannels()!=nch or r.getsampwidth()!=sw or r.getframerate()!=sr:
                    raise ValueError(f"Param mismatch in {wpath}")
                while True:
                    buf = r.readframes(4096)
                    if not buf: break
                    w_out.writeframes(buf)


def main():
    allowed_tids, meta = read_metadata_allowed_tids(META_TSV)

    # Pass 1: build raw segment manifest
    rows = []
    audio_index = {}  # conversation_id -> absolute wav path
    for root, _, files in os.walk(AUDIO_ROOT):
        for fn in files:
            if not fn.lower().endswith(".wav"):
                continue
            wav_path = os.path.join(root, fn)
            # index by both raw stem and normalized base so either lookup works
            stem = os.path.splitext(fn)[0]                 # e.g., 'aaseral_01_um-02uk'
            norm = norm_base_from_audio(fn)               # e.g., 'aaseral_01um-02uk'
            audio_index[stem] = wav_path
            audio_index[norm] = wav_path

    for root, _, files in os.walk(TRS_ROOT):
        for fn in files:
            if not fn.lower().endswith(".trs"):
                continue
            if fn.lower().endswith(".conc.ort.trs"):
                # Skip the orthographic variant – we only want the plain .trs
                continue
            trs_path = os.path.join(root, fn)
            segments, spkmap = parse_trs_segments(trs_path, allowed_tids)
            # Attach audio path by conversation id
            conv_id = segments[0]["conv_id"] if segments else Path(trs_path).stem
            wav_path = resolve_wav_for_trs(conv_id, spkmap, allowed_tids, audio_index)
                # If you have alternative naming, handle here
            for r in segments:
                rows.append({
                    "conversation_id": r["conv_id"],
                    "wav_path": wav_path,
                    "trs_path": trs_path,
                    "start_sec": round(r["start"], 3), "end_sec": round(r["end"], 3),
                    "duration_sec": round(max(0.0, r["end"]-r["start"]), 3),
                    "speaker_tid": r["speaker_tid"],
                    "overlap_flag": bool(r["overlap"]),
                    "keep_reason": r["keep_reason"],
                    "text": r["text"],
                })

    df = pd.DataFrame(rows)
    out_dir = Path(OUT_ROOT); out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir/"segments.csv", index=False)

    n_miss = df["wav_path"].isna().sum()
    if n_miss:
        print(f"[WARNING] No matching WAV for {n_miss} segments across " f"{df[df['wav_path'].isna()]['conversation_id'].nunique()} conversations.")
        print("  Examples:", df.loc[df["wav_path"].isna(), "conversation_id"].drop_duplicates().head(10).tolist())
    # Pass 2: clean
    #   - keep only: in-metadata + not overlap + has wav path
    #   - drop short (< 0.3s)
    df_clean = df.copy()
    df_clean = df_clean[
        (df_clean["speaker_tid"].isin(allowed_tids)) &
        (~df_clean["overlap_flag"].astype(bool)) &
        (df_clean["wav_path"].notna()) &
        (df_clean["duration_sec"] >= 0.3)
    ].copy()

    if df_clean.empty:
        print("[WARN] df_clean is empty after filtering; skipping audio writing.")
        df_clean.to_csv(out_dir/"segments_clean.csv", index=False)
        spk = pd.DataFrame(columns=["speaker_tid","total_dur_sec","n_segments"])
        spk.to_csv(out_dir/"speakers.csv", index=False)
        return


    # Merge tiny gaps (≤ 0.2s) within (speaker, conversation)
    df_clean = df_clean.sort_values(["speaker_tid", "conversation_id", "start_sec"])
    merged = []
    for (spk, conv), g in df_clean.groupby(["speaker_tid", "conversation_id"], sort=False):
        cur = None
        for r in g.itertuples(index=False):
            if cur is None:
                cur = r._asdict(); continue
            gap = r.start_sec - cur["end_sec"]
            if 0 <= gap <= 0.2:
                cur["end_sec"] = r.end_sec
                cur["duration_sec"] = round(cur["end_sec"] - cur["start_sec"], 3)
                cur["text"] = (cur.get("text","") + " " + (r.text or "")).strip()
            else:
                merged.append(cur); cur = r._asdict()
        if cur: merged.append(cur)
    df_clean = pd.DataFrame(merged)
    df_clean.to_csv(out_dir/"segments_clean.csv", index=False)

    # Pass 3: write per speaker per conversation WAVs
    tmp_dir = out_dir/"_tmp"; tmp_dir.mkdir(exist_ok=True)
    for (spk, conv), g in df_clean.groupby(["speaker_tid", "conversation_id"], sort=False):
        g = g.sort_values("start_sec")
        seg_wavs = []
        for i, row in enumerate(g.itertuples(index=False)):
            seg_path = tmp_dir / f"{conv}__{spk}__{i:06d}.wav"
            ffmpeg_extract(row.wav_path, row.start_sec, row.end_sec, str(seg_path))
            seg_wavs.append(str(seg_path))
        out_folder = Path(WAV_OUT_DIR) / spk
        out_folder.mkdir(parents=True, exist_ok=True)
        out_wav = out_folder / f"{conv}__{spk}.wav"
        ffmpeg_concat(seg_wavs, str(out_wav))
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Pass 4: speakers table + leakage-free splits
    spk = df_clean.groupby("speaker_tid")["duration_sec"].agg(["sum","count"]).reset_index()
    spk = spk.rename(columns={"sum":"total_dur_sec","count":"n_segments"})
    cols = ["tid","place","area","region","country","sex","age","agegroup","birth"]
    spk = spk.merge(meta[cols].rename(columns={"tid":"speaker_tid"}), on="speaker_tid", how="left")
    spk.to_csv(out_dir/"speakers.csv", index=False)

if __name__ == "__main__":
    main()
