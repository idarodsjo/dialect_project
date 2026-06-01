# dialect_project
Code for my project topic at NTNU regarding dialect recognition using geographical coordinates.

The dataset is not included in the repository, but the scripts for pre-processind and separation of speakers are found here.

The following steps can be followed to create the dataset with separated speakers:
(1) Find the absolute path for the dataset (found in talebase/data/speech_raw/NDC/norske_lydiler)
(2) Copy path into processing.sh and run script to perform pre-processing (downsamping and mixing down to mono)
(3) Again, adjust absolute paths in separate_speakers.py and run the script

Now, you should have the dataset ready to use for the DID model. 

**Training and Evaluation**
Training is run via shell_scripts/
  - Use *4fold script to run for 4fold-validation
  - multitask nordia runs the python script directly

Training and evaluation scripts (python) are in repo/did_prosody_whisper-main. Here the regression head architecture is defined.

Saved model checkpoints (classification, regression and multitask) are found on NTNU server.
Speaker-separated .wav files are in ndc_speaker_outputs/speaker_wavs.
