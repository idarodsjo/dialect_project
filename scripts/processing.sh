#!/bin/bash
set -e

# This script performs pre-processing of files in a directory (16kHz and Mono for all .wav files in directory)

SUFFIX="_prcssd"

TB_DIR="talebase/data/speech_raw/ScanDia/NorDia/norske-lydfiler-fra-ndc-"
NEW_DIRECTORY="preprocessed_NorDia/norske-lydfiler-fra-ndc-"

#mkdir -p "$NEW_DIRECTORY"

# Go through every subdir in /ScanDia/NorDia/ with .wav files
for ((i=4; i<=6; i++)); do
	# Check if processed files already exist
	# TBD
	# Iterate through files in NorDia/ subdirs
	for file in "${TB_DIR}${i}"/*.wav; do
		if [ -f "$file" ]; then
			#echo "Identified as file"
			filename=$(basename "$file")
			base_name="${filename%.*}"
			extension="${filename##*.}"
		
			new_filename="${base_name}${SUFFIX}.${extension}"

			# Create NEW_DIRECTORY+i if it does not exist
			mkdir -p "${NEW_DIRECTORY}${i}"
				
			# Pre-processing
			channels=$(soxi -c "$file")
			if [ "$channels" -eq 1 ]; then
				sox -v 0.95 "$file" -r 16k "${NEW_DIRECTORY}${i}/${new_filename}" gain -n remix 1
			else
				sox -v 0.95 "$file" -r 16k "${NEW_DIRECTORY}${i}/${new_filename}" gain -n channels 1
			#echo "Processed file: '${new_filename}'"		
			fi
		fi
	done
echo "Copied all files into '${NEW_DIRECTORY}${i}'"
done
