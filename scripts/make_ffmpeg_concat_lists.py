# make_ffmpeg_concat_lists.py
import csv
from pathlib import Path
import argparse

def make_concat_for_speaker(speaker_id, label_path, audio_path, out_path):
    """
    label_path: Audacity label file for this speaker (start \t end \t label)
    """
    lines = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for row in f:
            if not row.strip():
                continue
            s, e, *_ = row.strip().split('\t')
            s = float(s); e = float(e)
            lines.append((s, e))

    list_file = Path(out_path) / f"{speaker_id}.list"
    with open(list_file, 'w', encoding='utf-8') as f:
        for (s, e) in lines:
            f.write(f"file '{Path(audio_path).as_posix()}'\n")
            f.write(f"inpoint {s:.6f}\n")
            f.write(f"outpoint {e:.6f}\n")
    return list_file

# Example usage:
# list_file = make_concat_for_speaker("spk1", "out/labels/spk1_alvdal_02uk.txt", "alvdal_01um-02uk.wav", "out/lists")
# Then run ffmpeg command below.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('txt_dir', help='Path to txt file')
    ap.add_argument('audio_dir', help='Path to audio file')
    ap.add_argument('--out', help='Output directory for .list files', default='out/lists')

    args = ap.parse_args()
    file_name = Path(args.txt_dir)
    speaker_id = file_name.split('-',1)[0]

    
