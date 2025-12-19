import os
import numpy as np
import pandas as pd
import re
import tqdm

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
    ### test first 
    # ssc_info = pd.read_pickle('/home/plparson/ssc_data/data_splits_noSpeakerOverlap_fourDialect/ssc_dialect_test.pkl')

    # ### now the rest
    # ssc_info = []
    # for data_split_file in os.listdir('/home/plparson/ssc_data/data_splits_noSpeakerOverlap_fourDialect'):
    #     if 'test' not in data_split_file:
    #         ssc_info.append(
    #             pd.read_pickle(os.path.join('/home/plparson/ssc_data/data_splits_noSpeakerOverlap_fourDialect', data_split_file))
    #         )
    # ssc_info = pd.concat(ssc_info)

    # reaper_output_dir = '/home/plparson/REAPER_f0s/SSC'
    # ssc_reaper_files = [os.path.join(reaper_output_dir, os.path.basename(audio_file).replace('.mp3', '.f0')) for audio_file in ssc_info['full_audio_file']]

    # low_pass_output_dir = '/home/plparson/ssc_data/data_low_pass_audio'
    # ssc_low_pass_output_files = [os.path.join(low_pass_output_dir, 'audio', os.path.basename(audio_file)) for audio_file in ssc_info['full_audio_file']]

    # hann_templates = ''

    # reaper_no_pitch_measurements = []
    # to_process_count = 0

    # for orig_file, f0_file, low_out_file in zip(list(ssc_info['full_audio_file']), ssc_reaper_files, ssc_low_pass_output_files):
    #     ### NOTE: changing to check if output file exists yet
    #     if not os.path.isfile(low_out_file):
    #         mean_f0 = get_mean_f0_from_reaper_file(f0_file)
    #         if mean_f0:
    #             cutoff_freq = get_new_fit_results(mean_f0)
    #             hann_templates += praat_hann_band_template(
    #                 orig_file,
    #                 low_out_file,
    #                 cutoff_freq
    #             )
    #             to_process_count += 1
    #         else:
    #             reaper_no_pitch_measurements.append(f0_file)

    # with open(os.path.join(low_pass_output_dir, 'reaper_files_without_any_f0_values.txt'), 'w') as open_f:
    #     open_f.write(
    #         '\n'.join(reaper_no_pitch_measurements)
    #     )

    # output_script_loc = '/home/plparson/deepthought_development/praat_scripts/ssc_low_pass_hann_band.praat'
    # with open(output_script_loc, 'w') as open_f:
    #     open_f.write(hann_templates)

    # print(to_process_count, 'files to process in Praat script')

    # print('Now go run: praat --run', output_script_loc)



    ### what's the range of F0 cutoffs we have?
    ssc_info = []
    for data_split_file in os.listdir('/home/plparson/ssc_data/data_splits_noSpeakerOverlap_fourDialect'):
        if 'test' not in data_split_file:
            ssc_info.append(
                pd.read_pickle(os.path.join('/home/plparson/ssc_data/data_splits_noSpeakerOverlap_fourDialect', data_split_file))
            )
    ssc_info = pd.concat(ssc_info)

    reaper_output_dir = '/home/plparson/REAPER_f0s/SSC'
    ssc_reaper_files = [os.path.join(reaper_output_dir, os.path.basename(audio_file).replace('.mp3', '.f0')) for audio_file in ssc_info['full_audio_file']]

    # low_pass_output_dir = '/home/plparson/ssc_data/data_low_pass_audio'
    # ssc_low_pass_output_files = [os.path.join(low_pass_output_dir, 'audio', os.path.basename(audio_file)) for audio_file in ssc_info['full_audio_file']]

    # hann_templates = ''

    reaper_no_pitch_measurements = []
    # to_process_count = 0

    lowest = 10000000000
    highest = 0

    for f0_file in tqdm.tqdm(ssc_reaper_files): # , ssc_low_pass_output_files):
        ### NOTE: changing to check if output file exists yet
        mean_f0 = get_mean_f0_from_reaper_file(f0_file)
        if mean_f0:
            cutoff_freq = get_new_fit_results(mean_f0)
            if cutoff_freq > highest:
                highest = cutoff_freq
            if cutoff_freq < lowest:
                lowest = cutoff_freq
        else:
            reaper_no_pitch_measurements.append(f0_file)

    print('The highest cutoff freq we have for the SSC dataset is: ', highest, 'and the lowest is: ', lowest)