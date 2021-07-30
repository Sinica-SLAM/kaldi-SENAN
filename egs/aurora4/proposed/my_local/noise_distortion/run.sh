#/bin/bash

. ./cmd.sh
. ./path.sh

# Prepare noise dir -> data/train_si84_noise_mismatch_sp_hires
python3 my_local/noise_distortion/prepare_noise_dir.py

# Extract 40-D MFCC
data_dirs="train_si84_noise_mismatch_sp_hires test_A_noise_mismatch_hires test_B_noise_mismatch_hires test_C_noise_mismatch_hires test_D_noise_mismatch_hires"
for data_dir in $data_dirs; do
    steps/make_mfcc.sh --nj 20 --mfcc-config conf/mfcc_hires.conf \
        --cmd "$train_cmd" data/$data_dir
    steps/compute_cmvn_stats.sh data/$data_dir
    utils/fix_data_dir.sh data/$data_dir
done