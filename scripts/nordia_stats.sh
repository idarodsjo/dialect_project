
export METADATA_SPEAKER_COL="tid"
export METADATA_GENDER_COL="sex"
export METADATA_MUNI_COL="place"
export METADATA_COUNTY_COL="area"

export MAP_MUNI_COL="old_muni"
export MAP_COUNTY_COL="old_county"
export MAP_CARD_COL="cardinal_four"

export ENABLE_SUFFIX_GENDER_HEURISTIC=0


#!/usr/bin/env bash
set -euo pipefail

# nordia_stats.sh
#
# Computes stats for NorDia/NDC:
#   - Speaker stats ONLY from metadata (TSV).
#   - Validates 4 speakers per place (default place=metadata 'place'; switch via SPEAKER_PLACE_LEVEL=county).
#   - TRS parsing for segment stats.
#   - Per-recording region (county) and dialect (cardinal) from TRS speakers -> metadata (tid) -> county -> mapping.
#   - Per-dialect (cardinal_four) stats now include: recordings, unique speakers, segments,
#     mean_segment_duration_s, total_duration_s.
#
# Directory layout:
#   NorDia/
#     |- fonetisk-ortografisk-transkripsjoner-til-nedlasting/   (TRS files)
#     |- NDC_norge_metadata.tsv                                  (TSV metadata; one row per speaker)
#
# Usage:
#   bash nordia_stats.sh /path/to/NorDia [output_dir]
#
# Optional env overrides (metadata headers vary across releases):
#   METADATA_TID_COL       (default autodetect: tid, id, speaker, speaker_id, spk, navn)
#   METADATA_GENDER_COL    (default autodetect: sex, gender, kjønn)
#   METADATA_PLACE_COL     (default autodetect: place, kommune, municipality)
#   METADATA_COUNTY_COL    (default autodetect: county, fylke, area, region)
#
# Optional:
#   SPEAKER_PLACE_LEVEL=municipality|county   (default: municipality -> uses metadata 'place')

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/NorDia [output_dir]" >&2
  exit 1
fi

ROOT="${1%/}"
OUTDIR="${2:-dataset_stats}"

TRS_DIR="$ROOT/fonetisk-ortografisk-transkripsjoner-til-nedlasting"
METADATA="$ROOT/NDC_norge_metadata.tsv"

if [[ ! -d "$ROOT" ]]; then
  echo "Error: ROOT dir not found: $ROOT" >&2
  exit 1
fi
if [[ ! -d "$TRS_DIR" ]]; then
  echo "Error: TRS dir not found: $TRS_DIR" >&2
  exit 1
fi
if [[ ! -f "$METADATA" ]]; then
  echo "Error: Metadata TSV not found: $METADATA" >&2
  exit 1
fi

mkdir -p "$OUTDIR"

export TRS_DIR METADATA OUTDIR
export METADATA_TID_COL=${METADATA_TID_COL:-}
export METADATA_GENDER_COL=${METADATA_GENDER_COL:-}
export METADATA_PLACE_COL=${METADATA_PLACE_COL:-}
export METADATA_COUNTY_COL=${METADATA_COUNTY_COL:-}
export SPEAKER_PLACE_LEVEL=${SPEAKER_PLACE_LEVEL:-municipality}

python3 - << 'PYCODE'
import os, sys, csv, re
from collections import defaultdict, Counter
from statistics import mean

TRS_DIR = os.environ["TRS_DIR"]
METADATA = os.environ["METADATA"]
OUTDIR = os.environ["OUTDIR"]
PLACE_LEVEL = os.environ.get("SPEAKER_PLACE_LEVEL","municipality").strip().lower()
if PLACE_LEVEL not in ("municipality","county"):
    print(f"[WARN] SPEAKER_PLACE_LEVEL={PLACE_LEVEL} invalid; defaulting to 'municipality'", file=sys.stderr)
    PLACE_LEVEL = "municipality"

# ---- Helpers ----
def norm(s): return (s or "").strip()
def norm_key(s): return re.sub(r'\s+', ' ', norm(s)).lower()

# County -> cardinal mapping (provided by user)
county_to_cardinal_raw = {
    "Agder": "east",
    "Innlandet": "east",
    "Møre og Romsdal": "mid",
    "Nordland": "north",
    "Oslo": "east",
    "Rogaland": "west",
    "Troms": "north",
    "Finnmark": "north",
    "Trøndelag": "mid",
    "Vestfold": "east",
    "Telemark": "east",
    "Vestland": "west",
    "Østfold": "east",
    "Buskerud": "east",
    "Akershus": "east",
    "Oppland": "east",
    "Hedmark": "east",
    "Sogn og Fjordane": "west",
    "Hordaland": "west",
    "Sør-Trøndelag": "mid",
    "Nord-Trøndelag": "mid",
    "Aust-Agder": "east",
    "Vest-Agder": "east",
    "Telemark": "east",
    "Vestfold og Telemark": "east",
    "Viken": "east"
}
county_to_cardinal = {norm_key(k): v for k, v in county_to_cardinal_raw.items()}

def map_county_to_cardinal(county_name: str):
    ck = norm_key(county_name)
    if not ck:
        return ''
    if ck in county_to_cardinal:
        return county_to_cardinal[ck]
    # Split merged/compound labels (e.g., "Troms og Finnmark", "Sogn og Fjordane")
    parts = re.split(r'\s+og\s+|[\/\-,]', ck)
    parts = [p.strip() for p in parts if p.strip()]
    for p in parts:
        if p in county_to_cardinal:
            return county_to_cardinal[p]
    # substring fallback
    for known in county_to_cardinal:
        if known in ck:
            return county_to_cardinal[known]
    return ''

# ---- Read metadata (TSV) ----
def read_metadata_tsv(path):
    for enc in ('utf-8', 'utf-8-sig', 'latin-1'):
        try:
            with open(path, 'r', encoding=enc, newline='') as f:
                reader = csv.DictReader(f, delimiter='\t')
                headers = reader.fieldnames or []
                rows = list(reader)
            return headers, rows
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode metadata TSV: {path}")

meta_headers, meta_rows = read_metadata_tsv(METADATA)

# Autodetect columns or use env overrides
OVR_TID   = os.environ.get('METADATA_TID_COL') or None
OVR_GEN   = os.environ.get('METADATA_GENDER_COL') or None
OVR_PLACE = os.environ.get('METADATA_PLACE_COL') or None
OVR_CTNY  = os.environ.get('METADATA_COUNTY_COL') or None

def find_col(headers, pats):
    for rx in pats:
        p = re.compile(rx, re.I)
        for h in headers:
            if p.search(h):
                return h
    return None

tid_col    = OVR_TID   or find_col(meta_headers, [r'\btid\b', r'\bid\b', r'\bspeaker(_id)?\b', r'\bspk\b', r'\bnavn\b'])
gender_col = OVR_GEN   or find_col(meta_headers, [r'\bsex\b', r'\bgender\b', r'kjønn'])
place_col  = OVR_PLACE or find_col(meta_headers, [r'\bplace\b', r'\bkommune\b', r'\bmunicip(al|ality)\b'])
# Prefer county/fylke/area; if none, fallback to region as best-effort:
county_col = OVR_CTNY  or find_col(meta_headers, [r'\bcounty\b', r'\bfylke\b', r'\barea\b', r'\bregion\b'])

missing = [n for n,c in [('tid',tid_col),('gender',gender_col),('place',place_col),('county/area/region',county_col)] if c is None]
if missing:
    print(f"[WARN] Could not autodetect metadata column(s): {', '.join(missing)}", file=sys.stderr)

# ---- Build metadata indices ----
tid2attrs = {}
place2speakers = defaultdict(set)
all_speakers = set()
speaker2gender = {}

def norm_gender(g):
    gk = norm_key(g)
    if gk in ('f', 'female', 'k', 'kvinne', 'kv'):
        return 'female'
    if gk in ('m', 'male', 'mann'):
        return 'male'
    return 'unknown'

for r in meta_rows:
    tid = norm(r.get(tid_col,'')) if tid_col else ''
    if not tid:
        continue
    tid_key = norm_key(tid)
    place = norm(r.get(place_col,'')) if place_col else ''
    county = norm(r.get(county_col,'')) if county_col else ''
    gender = norm_gender(r.get(gender_col,'')) if gender_col else 'unknown'

    tid2attrs[tid_key] = {'place': place, 'county': county, 'gender': gender}
    all_speakers.add(tid_key)
    speaker2gender[tid_key] = gender

    if PLACE_LEVEL == 'municipality':
        place_key = place
    else:
        place_key = county
    if place_key:
        place2speakers[place_key].add(tid_key)

# Fill genders defaults if any missing
for spk in all_speakers:
    if spk not in speaker2gender:
        speaker2gender[spk] = 'unknown'
gender_counts = Counter(speaker2gender.values())

# ---- Parse TRS files and derive per-recording speakers/segments ----
import xml.etree.ElementTree as ET

def parse_trs_file(path):
    with open(path, 'rb') as f:
        raw = f.read()
    text = None
    for enc in ('latin-1','utf-8','utf-8-sig'):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise UnicodeError("decode failed")
    # avoid external DTD loading
    text = re.sub(r'<!DOCTYPE[^>]*>', '', text, flags=re.IGNORECASE)
    root = ET.fromstring(text)

    audio_filename = root.attrib.get('audio_filename', '')  # e.g., "aaseral_01um-02uk"
    # Collect speakers from <Speakers><Speaker name="...">
    speakers = []
    spk_elem = root.find('Speakers')
    if spk_elem is not None:
        for sp in spk_elem.findall('Speaker'):
            nm = sp.attrib.get('name') or sp.attrib.get('id') or ''
            nm = norm(nm)
            if nm:
                speakers.append(nm)

    # Segment durations from Turns+Sync
    seg_count = 0
    seg_durs = []
    for turn in root.iter('Turn'):
        st = turn.attrib.get('startTime')
        et = turn.attrib.get('endTime')
        try:
            start = float(st) if st not in (None,'') else None
            end = float(et) if et not in (None,'') else None
        except ValueError:
            start, end = None, None
        syncs = []
        for s in turn.findall('Sync'):
            t = s.attrib.get('time')
            try:
                if t not in (None,''):
                    syncs.append(float(t))
            except ValueError:
                pass
        syncs.sort()
        if syncs:
            for i in range(len(syncs)-1):
                d = syncs[i+1] - syncs[i]
                if d > 0:
                    seg_durs.append(d); seg_count += 1
            if end is not None and syncs:
                d = end - syncs[-1]
                if d > 0:
                    seg_durs.append(d); seg_count += 1
        else:
            if start is not None and end is not None and end > start:
                seg_durs.append(end - start); seg_count += 1

    return audio_filename, speakers, seg_count, seg_durs

trs_files = []
for rr, dd, ff in os.walk(TRS_DIR):
    for fn in ff:
        if fn.lower().endswith('.trs'):
            trs_files.append(os.path.join(rr, fn))
trs_files.sort()
if not trs_files:
    print(f"[ERROR] No .trs files found under {TRS_DIR}", file=sys.stderr)
    sys.exit(2)

# Aggregates
total_segments = 0
segment_durations = []
file_segment_rows = []    # per TRS file
audio2speakers_trs = {}   # audio_filename -> set of speaker tids (normalized, as found in TRS)
warnings = []

for p in trs_files:
    try:
        audio_fn, spk_list, seg_count, seg_durs = parse_trs_file(p)
    except Exception as e:
        warnings.append(f"Failed to parse {p}: {e}")
        continue

    total_segments += seg_count
    segment_durations.extend(seg_durs)
    mf = mean(seg_durs) if seg_durs else 0.0
    file_segment_rows.append({
        'file': os.path.relpath(p, TRS_DIR),
        'audio_filename': audio_fn,
        'n_segments': seg_count,
        'mean_segment_duration_s': f"{mf:.3f}",
        'total_duration_s': f"{(sum(seg_durs) if seg_durs else 0.0):.3f}",
    })
    audio2speakers_trs[norm_key(audio_fn)] = {norm_key(s) for s in spk_list if s}

# ---- Derive per-recording location/dialect via TRS speakers -> metadata ----
region_counts = defaultdict(lambda: {'recordings': 0, 'unique_speakers': set()})
dialect_counts = defaultdict(lambda: {
    'recordings': 0,
    'unique_speakers': set(),
    'segments': 0,
    'total_duration_s': 0.0
})

for row in file_segment_rows:
    audio_key = norm_key(row['audio_filename'])
    trsspeakers = audio2speakers_trs.get(audio_key, set())

    matched = [s for s in trsspeakers if s in tid2attrs]
    if not matched:
        warnings.append(f"No TRS speakers matched metadata for audio '{row['audio_filename']}'")
        continue

    # Counties from matched speakers
    counties = [tid2attrs[s]['county'] for s in matched if tid2attrs[s].get('county')]
    places  = [tid2attrs[s]['place'] for s in matched if tid2attrs[s].get('place')]

    county_choice = ''
    if counties:
        c_counts = Counter([norm_key(c) for c in counties])
        county_choice_norm, _ = c_counts.most_common(1)[0]
        for c in counties:
            if norm_key(c) == county_choice_norm:
                county_choice = c
                break
        if len({norm_key(c) for c in counties}) > 1:
            warnings.append(f"Mixed counties in audio '{row['audio_filename']}': {sorted({c for c in counties})}; using majority '{county_choice}'")
    else:
        warnings.append(f"No county found via metadata speakers for audio '{row['audio_filename']}'")
        county_choice = ''

    # Region aggregation (use county if available, else place)
    region_label = county_choice if county_choice else (places[0] if places else '')
    region_key = region_label if region_label else '(unknown region)'

    # Update region counts (recordings + unique speakers)
    region_counts[region_key]['recordings'] += 1
    region_counts[region_key]['unique_speakers'].update(matched)

    # Dialect via mapping
    card = map_county_to_cardinal(county_choice)
    card_key = card if card else '(unknown cardinal4)'

    # Update dialect aggregates
    dialect_counts[card_key]['recordings'] += 1
    dialect_counts[card_key]['unique_speakers'].update(matched)
    # Add segment stats from this file
    segs = int(row['n_segments'])
    # row['total_duration_s'] is a string formatted to 3 decimals; recompute from file rows list
    # safer to pull from original segment_durations per file, but we don't store per-file durs map.
    # Instead, recompute sum as mean * count when necessary:
    try:
        total_dur = float(row.get('total_duration_s', '0'))
    except:
        # fallback
        total_dur = float(row['mean_segment_duration_s']) * segs
    dialect_counts[card_key]['segments'] += segs
    dialect_counts[card_key]['total_duration_s'] += total_dur

# ---- Validate 4 speakers per place from metadata ONLY ----
violations = []
for place, spks in sorted(place2speakers.items(), key=lambda x: norm_key(x[0])):
    if len(spks) != 4:
        violations.append({
            'place_type': PLACE_LEVEL,
            'place': place,
            'unique_speakers': len(spks)
        })

# ---- Write outputs ----
def write_csv(path, fieldnames, rows):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# Global stats (speakers from metadata ONLY)
global_rows = [{
    'total_unique_speakers_metadata': len(all_speakers),
    'speakers_female': gender_counts.get('female', 0),
    'speakers_male': gender_counts.get('male', 0),
    'speakers_unknown': gender_counts.get('unknown', 0),
    'total_segments_from_trs': sum(int(r['n_segments']) for r in file_segment_rows),
    'mean_segment_duration_s': f"{(mean([float(r['mean_segment_duration_s']) for r in file_segment_rows if r['n_segments'] != 0]) if file_segment_rows else 0.0):.3f}",
    'speaker_place_level_validated': PLACE_LEVEL
}]
write_csv(os.path.join(OUTDIR, 'global_stats.csv'),
          ['total_unique_speakers_metadata','speakers_female','speakers_male','speakers_unknown',
           'total_segments_from_trs','mean_segment_duration_s','speaker_place_level_validated'],
          global_rows)

# Segments per TRS file
write_csv(os.path.join(OUTDIR, 'segments_per_file.csv'),
          ['file','audio_filename','n_segments','mean_segment_duration_s','total_duration_s'],
          file_segment_rows)

# Region (county or fallback place) counts
region_rows = []
for region, d in sorted(region_counts.items(), key=lambda x: norm_key(x[0])):
    region_rows.append({
        'region_label': region,
        'recordings': d['recordings'],
        'unique_speakers': len(d['unique_speakers']),
    })
write_csv(os.path.join(OUTDIR, 'region_counts.csv'),
          ['region_label','recordings','unique_speakers'],
          region_rows)

# Dialect (cardinal) counts with new duration/segment stats
dialect_rows = []
for card, d in sorted(dialect_counts.items(), key=lambda x: x[0]):
    segs = d['segments']
    total_dur = d['total_duration_s']
    mean_dur = (total_dur / segs) if segs > 0 else 0.0
    dialect_rows.append({
        'dialect_cardinal4': card,
        'recordings': d['recordings'],
        'unique_speakers': len(d['unique_speakers']),
        'segments': segs,
        'mean_segment_duration_s': f"{mean_dur:.3f}",
        'total_duration_s': f"{total_dur:.3f}",
    })
write_csv(os.path.join(OUTDIR, 'dialect_counts.csv'),
          ['dialect_cardinal4','recordings','unique_speakers','segments','mean_segment_duration_s','total_duration_s'],
          dialect_rows)

# Violations of "4 speakers per place"
write_csv(os.path.join(OUTDIR, 'places_with_non_four_speakers.csv'),
          ['place_type','place','unique_speakers'],
          violations)

# ---- Console summary ----
print("\n=== NorDia / NDC Dataset Stats ===")
print(f"TRS files parsed:           {len(file_segment_rows)}")
print(f"Unique speakers (metadata): {len(set(all_speakers))} "
      f"(female={Counter(speaker2gender.values()).get('female',0)}, male={Counter(speaker2gender.values()).get('male',0)}, unknown={Counter(speaker2gender.values()).get('unknown',0)})")
print(f"Total segments (TRS):       {sum(int(r['n_segments']) for r in file_segment_rows)}")
print(f"Dialect rows:               {len(dialect_rows)}")
print(f"Speaker-place validation:   level='{PLACE_LEVEL}'")
if violations:
    print(f"[CHECK] Places with != 4 speakers ({len(violations)}):")
    for v in violations[:50]:
        print(f" - {v['place_type']}: {v['place']} -> {v['unique_speakers']} speakers")
    if len(violations) > 50:
        print(f" ... {len(violations)-50} more")
else:
    print("[CHECK] All places have exactly 4 speakers ✅")

if warnings:
    print("\n[WARNINGS]")
    for w in warnings[:80]:
        print(" - " + w)
    if len(warnings) > 80:
        print(f" ... {len(warnings)-80} more")

print(f"\nOutputs written to: {OUTDIR}/")
PYCODE

echo "Done."