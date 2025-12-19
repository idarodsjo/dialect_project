import os
import pandas as pd
from scipy.io import wavfile
from nvos_low_pass import get_mean_f0_from_reaper_file


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

    nvos_half = pd.read_pickle('/home/plparson/deepthought_development/dialect-identification/datasets/datasets_nvos/key_4Dialects.pkl')
    nvos_full = pd.read_pickle('/home/plparson/deepthought_development/dialect-identification/datasets/datasets_nvos_full/key_4Dialects.pkl')

    reaper_output_dir = '/home/plparson/REAPER_f0s/NVOS'

    nvos_reaper_files_half = [os.path.join(reaper_output_dir, os.path.basename(audio_file).replace('.wav', '.f0')) for audio_file in nvos_half['full_audio_path']]
    nvos_reaper_files_full = [os.path.join(reaper_output_dir, os.path.basename(audio_file).replace('.wav', '.f0')) for audio_file in nvos_full['full_audio_path']]

    monotonize_output_dir = '/home/plparson/deepthought_development/dialect-identification/datasets/datasets_nvos/audio_monotonize'
    nvos_monotonize_output_files_half = [os.path.join(monotonize_output_dir, os.path.basename(audio_file)) for audio_file in nvos_half['full_audio_path']]
    monotonize_output_dir = '/home/plparson/deepthought_development/dialect-identification/datasets/datasets_nvos_full/audio_monotonize'
    nvos_monotonize_output_files_full = [os.path.join(monotonize_output_dir, os.path.basename(audio_file)) for audio_file in nvos_full['full_audio_path']]

    monotonize_scripts = ''

    reaper_no_pitch_measurements = []

    for orig_file, f0_file, mono_out_file in zip(list(nvos_half['full_audio_path']), nvos_reaper_files_half, nvos_monotonize_output_files_half):
        mean_f0 = get_mean_f0_from_reaper_file(f0_file)
        if mean_f0:
            monotonize_scripts += praat_monotonize_template(
                orig_file,
                mono_out_file,
                get_file_duration(orig_file),
                mean_f0
            )
        else:
            reaper_no_pitch_measurements.append(f0_file)

    for orig_file, f0_file, mono_out_file in zip(list(nvos_full['full_audio_path']), nvos_reaper_files_full, nvos_monotonize_output_files_full):
        mean_f0 = get_mean_f0_from_reaper_file(f0_file)
        if mean_f0:
            monotonize_scripts += praat_monotonize_template(
                orig_file,
                mono_out_file,
                get_file_duration(orig_file),
                mean_f0
            )
        else:
            reaper_no_pitch_measurements.append(f0_file)

    if reaper_no_pitch_measurements:
        with open(os.path.join(monotonize_output_dir, 'reaper_files_without_any_f0_values.txt'), 'a') as open_f:
            open_f.write(
                '\n'.join(reaper_no_pitch_measurements)
            )

    output_script_loc = '/home/plparson/deepthought_development/praat_scripts/nvos_monotonize.praat'
    with open(output_script_loc, 'w') as open_f:
        open_f.write(monotonize_scripts)

    print('Now go run: praat --run ', output_script_loc)