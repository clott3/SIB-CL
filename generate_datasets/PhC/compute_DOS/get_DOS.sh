#!/bin/sh
## NOTE: Python and MATLAB need to be added to PATH first
## This script combines h5 files of multiple seeds <h5prefix>-s<seed>-tm.h5 into a single file <h5prefix>-tm.h5 with their DOS computed and added.

## MODIFY IDENTIFIERS HERE ##
h5prefix=mf1
seeds=1\ 2\ 3\ 4

#####

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

# Remove text files and directories
rm -r ./txt_"$h5prefix";
rm -r ./DOS_"$h5prefix";
