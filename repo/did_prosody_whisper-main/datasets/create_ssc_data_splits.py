import pandas as pd

# this should come with the download of the SSC
ssc_json_official = pd.read_json('ssc/ssc_v1_0.jsonl', lines=True)
ssc_json_official['audio_path'] = ssc_json_official['audio_path'].astype(str)

for split in ['train', 'validation', 'test']:
    split_key = pd.read_pickle(f'repo/did_prosody_whisper-main/datasets/ssc_data/{split}_key.pkl')
    split_key['audio_path'] = split_key['audio_path'].astype(str)

    split_data = split_key.merge(ssc_json_official, on='audio_path', how='inner')
    split_data.to_pickle(f'repo/did_prosody_whisper-main/datasets/ssc_data/{split}_data.pkl')

