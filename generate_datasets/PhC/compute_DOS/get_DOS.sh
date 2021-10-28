#!/bin/sh
## NOTE: Python and MATLAB need to be added to PATH first
echo "This script reads h5 files of multiple seeds <h5prefix>-s<seed>-tm.h5, computes the DOS of all the data, and consolidates all the data into a single file <h5prefix>-tm.h5."

## Input identifiers
read -p "Enter h5 file prefix: " h5prefix
read -p "Enter all seeds (each separated by space; e.g. 1 2 3 4): " seeds

## Creates ./txt_<prefix>_sg<sg#> folder in rootdir and generates .txt files for input to Matlab.
nsam=$((python parsedat_forDOS.py --h5prefix $h5prefix --seeds $seeds) 2>&1);
echo Total samples: $nsam;

## Calculate DOS using GRR in Matlab. Creates ./DOS_<prefix>_sg<sg#> folder
## Should output code 0 if runs error free
matlab -nodisplay -nodesktop -r \
"try, run_GRR('$h5prefix',$nsam,'./'), catch, exit(1), end, exit(0);"
echo "Matlab exit code: $?";

## Add DOS back to h5 file
python addDOSintoh5.py --h5prefix $h5prefix --nsam $nsam;

## Remove text files and directories
rm -r ./txt_"$h5prefix";
rm -r ./DOS_"$h5prefix";
