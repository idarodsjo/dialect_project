import os
import pandas as pd
# import re

# from collections import namedtuple
from scipy.io import wavfile
from ssc_low_pass import get_mean_f0_from_reaper_file


def praat_monotonize_template(input_file_name: str, output_file_name: str, file_duration: float, average_hz: float):
    input_base = os.path.basename(input_file_name).replace('.wav', '').replace('.mp3', '')
    template =  '\n'.join([line.strip() for line in '''
    Read from file: "{0}"
    selectObject: "Sound {1}"
    To Manipulation: 0.01, 75, 600
    selectObject: "Manipulation {1}"
    Create PitchTier: "{1}", 0, {2}
    selectObject: "PitchTier {1}"
    Add point: 0, {3}
    selectObject: "Manipulation {1}"
    plusObject: "PitchTier {1}"
    Replace pitch tier
    selectObject: "Manipulation {1}"
    Get resynthesis (overlap-add)
    Save as WAV file: "{4}"
    selectObject: "Sound {1}"
    plusObject: "PitchTier {1}"
    plusObject: "Manipulation {1}"
    Remove
    selectObject: "Sound {1}"
    Remove
    '''.format(
        input_file_name, 
        input_base, 
        file_duration,
        average_hz, 
        output_file_name.replace('.mp3', '.wav')
    ).split('\n')])
    return template

def get_file_duration(wav_file:str):
    samplerate, data = wavfile.read(wav_file)
    return len(data) / samplerate

if __name__ == '__main__':

    wav_output_dir = '/home/plparson/ssc_data/wavs'
    
    ### test first
    # ssc_info = pd.read_pickle('/home/plparson/ssc_data/data_splits_noSpeakerOverlap_fourDialect/ssc_dialect_test.pkl')

    ### now the rest
    ssc_info = []
    for data_split_file in os.listdir('/home/plparson/ssc_data/data_splits_noSpeakerOverlap_fourDialect'):
        if 'test' not in data_split_file:
            ssc_info.append(
                pd.read_pickle(os.path.join('/home/plparson/ssc_data/data_splits_noSpeakerOverlap_fourDialect', data_split_file))
            )
    ssc_info = pd.concat(ssc_info)

    reaper_output_dir = '/home/plparson/REAPER_f0s/SSC'
    ssc_reaper_files = [os.path.join(reaper_output_dir, os.path.basename(audio_file).replace('.mp3', '.f0')) for audio_file in ssc_info['full_audio_file']]

    monotonize_output_dir = '/home/plparson/ssc_data/data_monotonize_audio'
    ssc_monotonize_output_files = [os.path.join(monotonize_output_dir, 'audio', os.path.basename(audio_file)) for audio_file in ssc_info['full_audio_file']]

    monotonize_scripts = ''

    reaper_no_pitch_measurements = []

    for orig_file, f0_file, mono_out_file in zip(list(ssc_info['full_audio_file']), ssc_reaper_files, ssc_monotonize_output_files):
        mean_f0 = get_mean_f0_from_reaper_file(f0_file)
        orig_file_wav = os.path.join(wav_output_dir, os.path.basename(orig_file).replace('.mp3', '.wav'))
        if mean_f0:
            monotonize_scripts += praat_monotonize_template(
                orig_file_wav,
                mono_out_file,
                get_file_duration(orig_file_wav),
                mean_f0
            )
        else:
            reaper_no_pitch_measurements.append(f0_file)

    with open(os.path.join(monotonize_output_dir, 'reaper_files_without_any_f0_values.txt'), 'a') as open_f:
        open_f.write(
            '\n'.join(reaper_no_pitch_measurements)
        )

    output_script_loc = '/home/plparson/deepthought_development/praat_scripts/ssc_monotonize.praat'
    with open(output_script_loc, 'w') as open_f:
        open_f.write(monotonize_scripts)

    print('Now go run: praat --run ', output_script_loc)