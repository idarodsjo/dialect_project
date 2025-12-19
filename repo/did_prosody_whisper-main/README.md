# did_prosody_whisper
Model training code used in support of the paper: [Effects of prosodic information on dialect classification using Whisper features](https://www.isca-archive.org/interspeech_2025/parsons25_interspeech.html)

## Setting up the datasets

### SSC

The Stortinget Speech Corpus can be accessed through the National Library of Norway's Language Bank [HERE](https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-91/)

1) Download the dataset
2) Adjust the paths in [create_ssc_data](/datasets/create_ssc_data_splits.py) to your local ones
3) Run the script to create the train/val/test files

### NVOS

The North Wind and the Sun dataset can be found [HERE](https://www.hf.ntnu.no/nos/)

1) Download the dataset using provided [script](./datasets/get_nvos_database.py)

### Audio manipulation

Both methods of audio manipulation are reliant on the average F0 of the utterance/segment. We used [REAPER](https://github.com/google/REAPER) to find this. 

1) Run the REAPER script on the desired data (see [nvos_reaper_f0s.py](/datasets/nvos_reaper_f0s.py) and [ssc_reaper_f0s.py](/datasets/ssc_reaper_f0s.py) for guidance)
2) Run the scripts ([nvos_low_pass.py](/datasets/nvos_low_pass.py) and [ssc_low_pass.py](/datasets/ssc_low_pass.py)) to create the Praat script for low-pass modification
3) Run the scripts ([nvos_monotonize.py](/datasets/nvos_low_pass.py) and [ssc_monotonize.py](/datasets/ssc_low_pass.py)) to create the Praat script for monotonization modification 
4) Run the created Praat scripts

## Training the model(s)

See [train_model_4Dialect.sh](./shell_scripts/train_model_4Dialect.sh) for example

## Evaluating the model(s)

See [eval_model_4Dialect.sh](./shell_scripts/eval_model_4Dialect.sh) for example