#!/bin/bash

TRS_DIR="talebase/data/speech_raw/ScanDia/NorDia/fonetisk-ortografisk-transkripsjoner-til-nedlasting"
TG_DIR="preprocessed_NorDia/trs_TextGrids"
mkdir -p $TG_DIR

for trs_file in $TRS_DIR/*conc.ort.trs; do
    base_name=$(basename "$trs_file" conc.ort.trs)
    tg_file="$TG_DIR/${base_name}.TextGrid"

    # Make output file and ensure it's writable
    touch "$tg_file"
    chmod +w "$tg_file"
    
    # Convert .trs to .TextGrid using a perl script
    perl scripts/trs_to_tg.pl "$trs_file" > "$tg_file"
done

# Count number of .conc.ort.trs files and .TextGrid files to ensure same amount
num_trs_files=$(ls $TRS_DIR/*conc.ort.trs | wc -l)
num_tg_files=$(ls $TG_DIR/*.TextGrid | wc -l)

if [ $num_trs_files -ne $num_tg_files ]; then
    echo "Warning: Number of .conc.ort.trs files ($num_trs_files) does not match number of .TextGrid files ($num_tg_files)"
fi