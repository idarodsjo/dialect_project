# # Getting the Nordavinden og Sola database
#
# We want to grab all of the data for the [Nordavinen og Sola](http://www.hf.ntnu.no/nos/) database. They don't seem to have a nice download button so let's do some webscraping!

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import wget
import soundfile as sf

from dialect_mapper import mapper_methods

### Setting all the directoryies and paths first

wav_base = 'http://www.hf.ntnu.no/nos/'

# NOTE: Change this path wo wherever you wish the data to go to
output_dir = ''
# NOTE: Chnage this to your local version of the dialect mapper
dialect_mapping = pd.read_csv('DialectMapper/dialect_mapper/mapping_data/muni_county_namedDialect_numericDialect_mapping_manual_additions_renamed_2024_cardinals.csv')

audio_dir = os.path.join(output_dir, 'audio')
if not os.path.isdir(audio_dir):
    os.mkdir(audio_dir)
metadata_dir = os.path.join(output_dir, 'metadata')
if not os.path.isdir(metadata_dir):
    os.mkdir(metadata_dir)

### --- Methods ---
def get_audio_duration(audio_file):
    data, samplerate = sf.read(audio_file)
    return len(data) / samplerate

### --- MAIN ----

all_tabu_regions = [n if len(n) == 2 else '0' + n for n in [str(i) for i in range(1, 24)]]
informant_tables = []

for tabu_region in all_tabu_regions:
    print(f'Working on region {tabu_region}')
    response = requests.get(f'https://www.hf.ntnu.no/nos/dialect.php?t={tabu_region}')
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html')
    tables = soup.find_all('table')
    for table in tables:
        trs = table.find_all('tr')
        if 'nos' + tabu_region in trs[0].text.strip():
            informant_tables.append(table)
    # to try to be nice to their server we're going to sleep for a little bit now
    time.sleep(2.2)

print('We have {} informant tables to process!'.format(len(informant_tables)))

informant_data_list = []

for informant_table in informant_tables:
    informant_data = {}

    trs = informant_table.find_all('tr')
    tds = trs[1].find_all('td')

    informant_data['id'] = trs[0].text.strip()
    informant_data['sted'] = tds[1].text.strip()
    informant_data['wav'] = os.path.join(wav_base, 'res', 'wav', informant_data['id'] + '.wav')
    informant_data['mp3'] = os.path.join(wav_base, 'res', 'mp3', informant_data['id'] + '.mp3')
    for tds_i in range(2, 6):
        k, v = ''.join(tds[tds_i].text.split()).split(':')
        informant_data[k.lower()] = v
    # index 6 is the audio file info, but we constructed that ourselves above
    informant_data['xsampa'] = ' '.join(tds[7].find('font').text.split()).strip('[').strip(']').strip()
    # index 8 is the IPA but it's an image not text so we'll skip
    k, v = ' '.join(tds[9].text.split()).split(':')
    informant_data[k.lower()] = v.strip()
    tds2 = tds[10].find_all('td')
    for tds_i in range(8):
        k, v = ' '.join(tds2[tds_i].text.split()).split(':')
        informant_data[k.lower()] = v.strip()
    informant_data_list.append(informant_data)

informant_df = pd.DataFrame(informant_data_list)

local_wav = []
for wav_file in list(informant_df['wav']):
    local_filename = wget.download(wav_file, out=audio_dir)
    local_wav.append(local_filename)
    time.sleep(1.5)
informant_df['local_wav'] = local_wav


informant_df.to_csv(os.path.join(
    metadata_dir, 
    'informant_data.csv'),
    index=False
)
informant_df.to_pickle(os.path.join(
    metadata_dir,
    'informant_data.pkl'
    )
)

print('\ndata saved to {}'.format(output_dir))

# now convert it to the 4 dialect regions used

mm = mapper_methods()

tabu_to_cardinal_four = {pair[0]:pair[1] for pair in zip(dialect_mapping['numeric_dialect'], dialect_mapping['cardinal_four'])}

### the four-region version
nvos_key_four = []
for row in informant_df.itertuples():
    duration = get_audio_duration(row.local_wav)
    kommune = row.kommune
    # fix human data entry errors
    if kommune == 'NordreLand':
        kommune = 'Nordre Land'
    if kommune == 'Borre': # Borre is a village in Horten
        kommune = 'Horten'
    # deal with out exceptions
    dialect_tabu = tabu_to_cardinal_four[int(row[7])]
    dialect_pob = mm.get_cardinal_four(kommune)
    if dialect_tabu != dialect_pob:
        if row.id == 'nos12001':
            dialect = 'west'
        elif row.id == 'nos20001':
            dialect = 'east' # change our 1 southerner to east
        elif row.id == 'nos21001': 
            dialect = 'east'
        elif row.id == 'nos21005':
            dialect = 'east'
        else:
            dialect = dialect_tabu
            print(dialect_tabu, dialect_pob)
    else:
        dialect = dialect_tabu

    nvos_key_four.append({
        'id' : row.id ,
        'tabu_region' : row[7], # tabu.0-region
        'dialect_tabu' : dialect_tabu,
        'dialect_pob' : dialect_pob,
        'dialect' : dialect,
        'full_audio_path' :  row.local_wav,
        'duration' : duration
    })

nvos_key_four = pd.DataFrame(nvos_key_four)
nvos_key_four.to_csv(os.path.join(ouput_dir, 'key_4Dialects.csv'), index=False)
nvos_key_four.drop(columns=['dialect_tabu', 'dialect_pob']).to_pickle(os.path.join(ouput_dir, 'key_4Dialects.pkl'))

nvos_key_four[nvos_key_four['dialect'] != nvos_key_four['dialect_pob']]
