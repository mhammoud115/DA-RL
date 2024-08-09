#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J LorenzRL
#SBATCH -o output/Lorenz96.%J.out
#SBATCH -e errors/Lorenz96.%J.err
#SBATCH --time=02:59:00
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --constraint=v100

DEVICE=cuda
NUM_GPUS=1
MY_MASTER_ADDR=127.0.0.1
MY_MASTER_PORT=$(shuf -i 30000-60000 -n 1)

NPL=$1
gamma=$2
LR=$3
batchSize=$4
total_timesteps=$5
VFCOEFF=$6
MAXGRADNORM=$7

AssimSteps=4
ObsNoise=1.0

LOG_DIR=trainedModels_L96_assimSteps${AssimSteps}_obsNosie${ObsNoise}_Markov1Step_TrueNorm/neurons${NPL}_LR${LR}_Gamma${gamma}_BS${batchSize}_totSteps${total_timesteps}_assimStep${AssimSteps}_obsNosie${ObsNoise}_maxgradnorm${MAXGRADNORM}_vfcoeff${VFCOEFF}/

CUDA_VISIBLE_DEVICES=0 python3 trainICs.py \
--log_dir $LOG_DIR \
--neurons_perLayer $NPL \
--learning_rate $LR \
--gamma $gamma \
--batchSize $batchSize \
--total_timesteps $total_timesteps \
--vf_coeff $VFCOEFF \
--max_grad_norm $MAXGRADNORM \
--assim_freq $AssimSteps \
--obs_Noise $ObsNoise \
--device $DEVICE
