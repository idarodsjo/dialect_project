import os
import subprocess
import pandas as pd
import concurrent.futures

from tqdm import tqdm

all_ssc = []
for data_split_file in os.listdir('/home/plparson/ssc_data/data_splits_noSpeakerOverlap_fourDialect'):
    all_ssc.append(
        pd.read_pickle(os.path.join('/home/plparson/ssc_data/data_splits_noSpeakerOverlap_fourDialect', data_split_file))
    )
all_ssc = pd.concat(all_ssc)
audio_files = list(all_ssc['full_audio_file'])

wav_output_dir = '/home/plparson/ssc_data/wavs'
reaper_output_dir = '/home/plparson/REAPER_f0s/SSC'

ffmpeg_commands = []
reaper_commands = []

pool = concurrent.futures.ThreadPoolExecutor(max_workers=20)

with open(os.path.join('/home/plparson/REAPER_f0s/ssc_subprocess_run_log.txt'), 'w') as open_log:
    temp_wav_file = '/home/plparson/REAPER_f0s/SSC/temp.wav'
    for audio_file in audio_files: # tqdm(audio_files[:3]):
        output_file_wav = os.path.join(wav_output_dir, os.path.basename(audio_file).replace('.mp3', '.wav'))
        output_file_f0 = os.path.join(reaper_output_dir, os.path.basename(audio_file).replace('.mp3', '.f0'))
                
        if os.path.isfile(output_file_f0):
            open_log.write('{} already exists, skipping REAPER analysis.\n'.format(output_file_f0))
        else:
        #     # shell out to ffmpeg to make the mp3 into a wav
        #     subprocess.run(
        #         [
        #             'ffmpeg', 
        #             '-y',
        #             '-i', 
        #             audio_file, 
        #             temp_wav_file
        #         ],
        #         stdout=open_log,
        #         stderr=open_log
        #     )
            pool.submit(
                # now shell out to REAPER (if necessary, takes a while to run so we'll first check if we have files)
                subprocess.run(
                    [
                        '/home/plparson/REAPER/build/reaper',
                        '-i',
                        output_file_wav,
                        '-f',
                        output_file_f0, 
                        '-a'
                    ],
                    stdout=open_log,
                    stderr=open_log
                )
            )

# with open('/home/plparson/ssc_data/make_wavs.sh', 'w') as open_f:
#     open_f.write('\n'.join(ffmpeg_commands))

# with open('/home/plparson/ssc_data/make_reapers.sh', 'w') as open_f:
#     open_f.write('\n'.join(reaper_commands))
pool.shutdown(wait=True)

print('Done')