import pandas as pd
import numpy as np

# Hardcode your mapping from county -> cardinal_four
county_to_cardinal = {
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

# Hardcode your pickle files
pkl_files = [
    "/home/idatro/repo/did_prosody_whisper-main/datasets/nordia_data/train_data.pkl",
    "/home/idatro/repo/did_prosody_whisper-main/datasets/nordia_data/validation_data.pkl",
    "/home/idatro/repo/did_prosody_whisper-main/datasets/nordia_data/test_data.pkl"
]

for pkl_path in pkl_files:
    print(f"\nProcessing: {pkl_path}")
    df = pd.read_pickle(pkl_path)

    if "cardinal_four" not in df.columns or "county" not in df.columns:
        print(f"Skipping {pkl_path}: missing required columns.")
        continue

    # Find NaN rows in cardinal_four
    nan_mask = df["cardinal_four"].isna()
    print(f"NaN count before: {nan_mask.sum()}")

    # Fill NaNs using mapping
    df.loc[nan_mask, "cardinal_four"] = df.loc[nan_mask, "county"].map(county_to_cardinal)

    # Save back to pickle
    df.to_pickle(pkl_path)

    # Verify
    remaining_nans = df["cardinal_four"].isna().sum()
    print(f"NaN count after: {remaining_nans}")
    if remaining_nans > 0:
        print("WARNING: Some NaNs remain. Check if mapping covers all counties.")
    else:
        print("✅ All NaNs filled successfully.")

tail -f /home/idatro/repo/did_prosody_whisper-main/train_regression.log