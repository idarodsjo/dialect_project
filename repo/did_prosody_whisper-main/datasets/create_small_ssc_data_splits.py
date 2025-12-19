import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dialect_mapper.mapper import mapper_methods
import json
import time 

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
#print("Sys path updated:", sys.path)
#print("Current directory:", os.getcwd())
from CoordMapper.coord_mapper import CoordMapper
csv_data = pd.read_csv('/home/idatro/repo/DialectMapper/dialect_mapper/mapping_data/muni_county_namedDialect_numericDialect_mapping_manual_additions_renamed_2024_cardinals.csv')
#print(f'Old counties: {csv_data["old_county"].unique()}')
#print(f'New counties: {csv_data["new_county_2024"].unique()}')

# full SSC dataset
ssc_json_official = pd.read_json('/home/idatro/ssc/ssc_v1_0.jsonl', lines=True)
ssc_json_official['audio_path'] = ssc_json_official['audio_path'].astype(str)

def ensure_list(x):
    if isinstance(x, str):
        # Convert string to list
        return json.loads(x)
    return x  # Already a list

ssc_json_official['speakers'] = ssc_json_official['speakers'].apply(ensure_list)

# init mapper
mm = mapper_methods()
mm.enable_stortinget_corrections()

# rename 'audio_path' (match in training script)
ssc_json_official.rename(columns={'audio_path': 'full_audio_file'}, inplace=True)

#  smaller subset for quick testing
subset_data = ssc_json_official.sample(frac=0.1, random_state=42).reset_index(drop=True)


# helpers for dialect mapping columms

def get_county(row, csv_data=csv_data):
    old_county = row.get('speakers')[0].get('birth_county') if row.get('speakers') else None
    mapping = csv_data[csv_data['old_county'] == old_county]
    if mapping.empty:
        mapping = csv_data[csv_data['new_county'] == old_county] # rogaland does not have old county in csv
        if mapping.empty:
            print(f'No mapping found for old_county: {old_county}')
            return None
    new_county = mapping['new_county_2024'].iloc[0]
    return new_county



def get_dialect_region(row, county):   
    # Prefer dialect from speakers if available
    speakers = row.get('speakers', [])
    if speakers and speakers[0].get('dialect'):
        return speakers[0]['dialect']
    if pd.notnull(county):
        region = mm.get_cardinal_four_by_new_county(county)
        if region:
            return region
    return None


def is_missing_like(val):
    """Treat None/NaN/''/'None'/'nan' (case-insensitive) as missing."""
    if val is None:
        return True
    if isinstance(val, float) and pd.isna(val):
        return True
    if isinstance(val, str) and val.strip().lower() in ("", "none", "nan"):
        return True
    return False



# adding geo and dialect columns
subset_data['municipality'] = None


subset_data['county'] = subset_data.apply(lambda row: get_county(row), axis=1)

# Remove rows where county is missing/invalid
missing_county_mask = subset_data['county'].apply(is_missing_like)
n_missing = int(missing_county_mask.sum())
if n_missing > 0:
    print(f"[INFO] Removing {n_missing} rows with missing/invalid county.")
    # Optional: write removed rows for inspection
    try:
        subset_data.loc[missing_county_mask, ['full_audio_file', 'speakers', 'old_county_raw']].to_csv(
            'debug_missing_county.csv', index=False
        )
        print("[INFO] Wrote debug_missing_county.csv with removed rows.")
    except Exception as e:
        print(f"[WARN] Could not write debug_missing_county.csv: {e}")

subset_data = subset_data.loc[~missing_county_mask].reset_index(drop=True)
print(f"[INFO] Subset size after removing missing county: {len(subset_data)}")



subset_data['dialect_region'] = subset_data.apply(
    lambda row: get_dialect_region(row, row['county']), axis=1
)


# Filter out rows with multiple dialect regions
removed_rows = subset_data[subset_data['dialect_region'].apply(lambda x: isinstance(x, list))]
print(f"Removed {len(removed_rows)} rows with multiple dialect regions.")
subset_data = subset_data[subset_data['dialect_region'].apply(lambda x: not isinstance(x, list))]
subset_data = subset_data[subset_data['dialect_region'].notnull()]  # remove nulls
subset_data['dialect_region'] = subset_data['dialect_region'].astype(str) # ensure string


# log missing municipalities
missing_muni_count = subset_data['municipality'].isnull().sum()
print(f"Number of missing municipalities: {missing_muni_count}")

# Add coordinates for each row based on county

cm = CoordMapper()  # Nominatim-backed geocoder
geo_cache = {}      # cache to avoid repeated requests: key = (place, area)


def get_lat_lon_for_row(row):
    """
    Prefer municipality+county when municipality is available.
    Otherwise fall back to county only.
    Returns (lat, lon) or (None, None) if not found.
    """
    muni = row.get('municipality')
    county = row.get('county')

    # Normalize to strings (CoordMapper expects strings)
    place = None
    area = None

    if isinstance(muni, str) and muni.strip():
        place = muni.strip()
        area = (county or "").strip() if isinstance(county, str) else ""
    else:
        # No municipality → use county as 'place', and empty area
        place = (county or "").strip() if isinstance(county, str) else ""
        area = ""

    if not place:
        return (None, None)

    key = (place, area)
    if key in geo_cache:
        return geo_cache[key]

    try:
        coords = cm.get_coordinates(place, area)  # (lat, lon) or None
        if coords is None:
            geo_cache[key] = (None, None)
            return (None, None)
        lat, lon = coords
        geo_cache[key] = (lat, lon)
        return (lat, lon)
    except Exception as e:
        # Be resilient to transient geocoding errors
        print(f"[WARN] Geocoding failed for ({place}, {area}): {e}")
        geo_cache[key] = (None, None)
        return (None, None)


# Compute latitude/longitude columns
subset_data['latitude'], subset_data['longitude'] = zip(*subset_data.apply(get_lat_lon_for_row, axis=1))

# Drop rows where coordinates could not be obtained
coord_missing_mask = subset_data['latitude'].isna() | subset_data['longitude'].isna()
n_coord_missing = int(coord_missing_mask.sum())
if n_coord_missing > 0:
    print(f"[INFO] Removing {n_coord_missing} rows with unresolved coordinates.")
    try:
        subset_data.loc[coord_missing_mask,
                        ['full_audio_file', 'municipality', 'county', 'old_county_raw', 'dialect_region']].to_csv(
            'debug_missing_coords.csv', index=False
        )
        print("[INFO] Wrote debug_missing_coords.csv with removed rows.")
    except Exception as e:
        print(f"[WARN] Could not write debug_missing_coords.csv: {e}")

subset_data = subset_data.loc[~coord_missing_mask].reset_index(drop=True)
print(f"[INFO] Subset size after coordinate resolution: {len(subset_data)}")




train_data, temp_data = train_test_split(subset_data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

train_data.to_pickle('/home/idatro/repo/did_prosody_whisper-main/datasets/ssc_data/train_data.pkl')
val_data.to_pickle('/home/idatro/repo/did_prosody_whisper-main/datasets/ssc_data/validation_data.pkl')
test_data.to_pickle('/home/idatro/repo/did_prosody_whisper-main/datasets/ssc_data/test_data.pkl')

print("Datasets updated with 'dialect_region' and 'full_audio_file' columns and saved.")
train_data[['full_audio_file', 'municipality', 'county', 'dialect_region', 'latitude', 'longitude']].head(50).to_csv('train_preview.csv', index=False)
print("Preview of training data saved to 'train_preview.csv'.")