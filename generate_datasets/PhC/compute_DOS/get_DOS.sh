#!/bin/sh
## NOTE: Python and MATLAB need to be added to PATH first
## This script combines h5 files of multiple seeds <h5prefix>-s<seed>-tm.h5 into a single file <h5prefix>-tm.h5 with their DOS computed and added.

## MODIFY IDENTIFIERS HERE ##
h5prefix=mf1
#h5prefix=mf1
seeds=1\ 2\ 3\ 4 
for sgno in 3
do

#####

## Creates ./txt_<prefix>_sg<sg#> folder in rootdir and generates .txt files for input to Matlab. 
#python parsedat_forDOS.py --sgnum $sgno --h5prefix mf2-c0 --seeds 1 2 3 4 5 6 7; 
nsam=$((python parsedat_forDOS.py --sgnum $sgno --h5prefix $h5prefix --seeds $seeds) 2>&1);
echo Total samples: $nsam;

## Calculate DOS using GRR in Matlab. Creates ./DOS_<prefix>_sg<sg#> folder
## Should output code 0 if runs error free
matlab -nodisplay -nodesktop -r \
"try, run_GRR($sgno,'$h5prefix',$nsam,'./'), catch, exit(1), end, exit(0);"
echo "Matlab exit code: $?";

## Add DOS back to h5 file
python addDOSintoh5.py --sgnum $sgno --h5prefix $h5prefix --nsam $nsam;

# Remove text files and directories
rm -r ./txt_"$h5prefix"_sg"$sgno";
rm -r ./DOS_"$h5prefix"_sg"$sgno";

done
 











