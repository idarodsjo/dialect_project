#!/bin/bash

BASE_DIR="preprocessed_NorDia/norske-lydfiler-fra-ndc-"
#BASE_DIR="talebase/data/speech_raw/ScanDia/NorDia/norske-lydfiler-fra-ndc-"
for (( i=1; i<=1; i++ ))
do
	cd "${BASE_DIR}${i}"

	num_files=$(find . -name "*.wav" -type f | wc -l)
	s_rate=$(find -name "*.wav" -exec soxi {} \; | grep "Sample Rate" | sort | uniq -c)
	ch=$(find -name "*.wav" -exec soxi {} \; | grep "Channels" | sort | uniq -c)
	echo "INFO ON ${BASE_DIR}${i}"
	echo "Number of files: $num_files"
	echo "Sampling rate: $s_rate"
	echo "Channels: $ch"

	echo "Stats per file:"

	for file in *.wav; do
		echo "File: $file"

		echo "  Channel 1:"
        	sox "$file" -n remix 1 stat 2>&1 | grep -E "RMS.*amplitude|Maximum amplitude"


		echo "  Channel 2:"
        	sox "$file" -n remix 2 stat 2>&1 | grep -E "RMS.*amplitude|Maximum amplitude"
	done
	cd -
done
