#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o phcbs.out
#SBATCH --job-name=phcbs
#SBATCH --open-mode=append

for ptlr in 1e-4 1e-3
do
for ss in 16 32 64
do
bscl=512; 
iden=phc_bs;
python main.py --predict "bandstructures" --train "sibcl" --ssl_mode "simclr" --batchsize_cl $bscl --batchsize $ss --iden $iden --learning_rate $ptlr --pg_uniform --scale --translate_pbc --no_finetune; 
for nsam in 50 100 200 400 800 1600 3200
do
for ftlr in 1e-4 1e-3
do 
for ftseed in 1 2 3
do
echo finetuning nsam$nsam ftlr$ftlr ftseed$ftseed;
python main.py --predict "bandstructures" --train "sibcl" --ssl_mode "simclr" --batchsize_cl $bscl --batchsize $ss --iden $iden --learning_rate $ptlr --pg_uniform --scale --translate_pbc --no_pretrain --learning_rate_ft $ftlr --nsam $nsam --ftseed $ftseed; 
done
done
done
done
done
