import os
import numpy as np
import pandas as pd
import re

from collections import namedtuple

reaper_row = namedtuple('reaper_row', ['time', 'has_measurement', 'f0_value'])

def make_reaper_row(line_match):
    return reaper_row(
        float(line_match.group(1)),
        float(line_match.group(2)),
        float(line_match.group(3))
    )

def get_mean_f0_from_reaper_file(reaper_file_name):
    with open(reaper_file_name, 'r') as open_f:
        lines = open_f.read()

    data_line_pat = re.compile('([\d\.]+) ([01]) ([\d\.\-]+)')

    lines = lines.split('\n')

    data_lines = [make_reaper_row(data_line_pat.search(l)) for l in lines if data_line_pat.match(l)]
    valid_measurement_lines = [l for l in data_lines if l.has_measurement]
    if len(valid_measurement_lines) > 0:
        measured_lines = pd.DataFrame(valid_measurement_lines)
        # print('Mean F0: {}'.format(measured_lines['f0_value'].mean()))
        return measured_lines['f0_value'].mean()

def get_new_fit_results(x):
    a = 4.20192827e+02 
    b = 1.24362435e-02
    return a * (1 - np.exp(-b* x))

def praat_hann_band_template(input_file_name: str, output_file_name: str, filter_hz=400):
    input_base = os.path.basename(input_file_name).replace('.wav', '').replace('.mp3', '')
    template =  '\n'.join([line.strip() for line in '''
    Read from file: "{}"
    selectObject: "Sound {}"
    Filter (pass Hann band): 0, {}, {}
    selectObject: "Sound {}_band"
    Scale peak: 0.99
    Save as WAV file: "{}"
    selectObject: "Sound {}_band"
    Remove
    selectObject: "Sound {}"
    Remove
    '''.format(
        input_file_name, 
        input_base, 
        filter_hz, # the "to" frequency
        filter_hz / 4, # smoothing
        input_base, 
        output_file_name.replace('.mp3', '.wav'),
        input_base,
        input_base
    ).split('\n')])
    return template

if __name__ == '__main__':

    nvos_half = pd.read_pickle('/home/plparson/deepthought_development/dialect-identification/datasets/datasets_nvos/key_4Dialects.pkl')
    nvos_full = pd.read_pickle('/home/plparson/deepthought_development/dialect-identification/datasets/datasets_nvos_full/key_4Dialects.pkl')

    reaper_output_dir = '/home/plparson/REAPER_f0s/NVOS'

    reaper_output_dir = '/home/plparson/REAPER_f0s/NVOS'
    nvos_reaper_files_half = [os.path.join(reaper_output_dir, os.path.basename(audio_file).replace('.wav', '.f0')) for audio_file in nvos_half['full_audio_path']]
    nvos_reaper_files_full = [os.path.join(reaper_output_dir, os.path.basename(audio_file).replace('.wav', '.f0')) for audio_file in nvos_full['full_audio_path']]

    low_pass_output_dir = '/home/plparson/deepthought_development/dialect-identification/datasets/datasets_nvos/audio_low_pass'
    nvos_low_pass_output_files_half = [os.path.join(low_pass_output_dir, os.path.basename(audio_file)) for audio_file in nvos_half['full_audio_path']]
    low_pass_output_dir = '/home/plparson/deepthought_development/dialect-identification/datasets/datasets_nvos_full/audio_low_pass'
    nvos_low_pass_output_files_full = [os.path.join(low_pass_output_dir, os.path.basename(audio_file)) for audio_file in nvos_full['full_audio_path']]

    hann_templates = ''

    reaper_no_pitch_measurements = []
    to_process_count = 0

    for orig_file, f0_file, low_out_file in zip(list(nvos_half['full_audio_path']), nvos_reaper_files_half, nvos_low_pass_output_files_half):
        ### NOTE: changing to check if output file exists yet
        if not os.path.isfile(low_out_file):
            mean_f0 = get_mean_f0_from_reaper_file(f0_file)
            if mean_f0:
                cutoff_freq = get_new_fit_results(mean_f0)
                hann_templates += praat_hann_band_template(
                    orig_file,
                    low_out_file,
                    cutoff_freq
                )
                to_process_count += 1
            else:
                reaper_no_pitch_measurements.append(f0_file)

    for orig_file, f0_file, low_out_file in zip(list(nvos_full['full_audio_path']), nvos_reaper_files_full, nvos_low_pass_output_files_full):
        ### NOTE: changing to check if output file exists yet
        if not os.path.isfile(low_out_file):
            mean_f0 = get_mean_f0_from_reaper_file(f0_file)
            if mean_f0:
                cutoff_freq = get_new_fit_results(mean_f0)
                hann_templates += praat_hann_band_template(
                    orig_file,
                    low_out_file,
                    cutoff_freq
                )
                to_process_count += 1
            else:
                reaper_no_pitch_measurements.append(f0_file)

    if reaper_no_pitch_measurements:
        with open(os.path.join('/home/plparson/deepthought_development/dialect-identification/datasets/datasets_nvos/', 'reaper_files_without_any_f0_values.txt'), 'w') as open_f:
            open_f.write(
                '\n'.join(reaper_no_pitch_measurements)
            )

    output_script_loc = '/home/plparson/deepthought_development/praat_scripts/nvos_low_pass_hann_band.praat'
    with open(output_script_loc, 'w') as open_f:
        open_f.write(hann_templates)

    print(to_process_count, 'files to process in Praat script')

    print('Now go run: praat --run', output_script_loc)
