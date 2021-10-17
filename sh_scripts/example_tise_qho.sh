#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o tise2dqho.out
#SBATCH --job-name=tise2dqho
#SBATCH --open-mode=append

for ptlr in 1e-5 1e-4
do
for ss in 32 64 128
do
bscl=128; 
iden=tise2d_sho;
python main.py --predict "eigval" --train "sibcl" --ndim 2 --tisesource "qho" --ssl_mode "simclr" --batchsize_cl $bscl --batchsize $ss --iden $iden --learning_rate $ptlr --pg_uniform --no_finetune; 
for nsam in 50 100 200 400 800 1600 3200
do
for ftlr in 1e-4 1e-3
do 
for ftseed in 1 2 3
do
echo finetuning nsam$nsam ftlr$ftlr ftseed$ftseed;
python main.py --predict "eigval" --train "sibcl" --ndim 2 --tisesource "qho" --ssl_mode "simclr" --batchsize_cl $bscl --batchsize $ss --iden $iden --learning_rate $ptlr --pg_uniform --no_pretrain --learning_rate_ft $ftlr --nsam $nsam --ftseed $ftseed; 
done
done
done
done
done
