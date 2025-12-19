import os
import pandas as pd

nvos_half = pd.read_pickle('/home/plparson/deepthought_development/dialect-identification/datasets/datasets_nvos/key_4Dialects.pkl')
nvos_full = pd.read_pickle('/home/plparson/deepthought_development/dialect-identification/datasets/datasets_nvos_full/key_4Dialects.pkl')

wav_output_dir = '/home/plparson/ssc_data/wavs'
reaper_output_dir = '/home/plparson/REAPER_f0s/NVOS'

reaper_commands = []

for audio_file in list(nvos_half['full_audio_path']): 
    output_file_f0 = os.path.join(reaper_output_dir, os.path.basename(audio_file).replace('.wav', '.f0'))
    
    if os.path.isfile(output_file_f0):
        print('{} already exists, skipping REAPER analysis.\n'.format(output_file_f0))
    else:
        reaper_commands.append(
            ' '.join([
                '/home/plparson/REAPER/build/reaper',
                '-i',
                audio_file,
                '-f',
                output_file_f0, 
                '-a'
            ]),
        )

with open(os.path.join('/home/plparson/REAPER_f0s/get_f0s_nvos_halves.sh'), 'w') as open_f:
    open_f.write(
        '\n'.join(reaper_commands)
    )

reaper_commands = []

for audio_file in list(nvos_full['full_audio_path']): 
    output_file_f0 = os.path.join(reaper_output_dir, os.path.basename(audio_file).replace('.wav', '.f0'))
    
    if os.path.isfile(output_file_f0):
        print('{} already exists, skipping REAPER analysis.\n'.format(output_file_f0))
    else:
        reaper_commands.append(
            ' '.join([
                '/home/plparson/REAPER/build/reaper',
                '-i',
                audio_file,
                '-f',
                output_file_f0, 
                '-a'
            ]),
        )

with open(os.path.join('/home/plparson/REAPER_f0s/get_f0s_nvos_full.sh'), 'w') as open_f:
    open_f.write(
        '\n'.join(reaper_commands)
    )

print('Done! Now run:\nsource ', '/home/plparson/REAPER_f0s/get_f0s_nvos_halves.sh', '\nsource', '/home/plparson/REAPER_f0s/get_f0s_nvos_full.sh')