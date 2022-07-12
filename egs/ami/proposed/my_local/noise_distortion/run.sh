#/bin/bash

. ./cmd.sh
. ./path.sh

# Prepare noise dir -> data/train_si84_noise_mismatch_sp_hires_40
python3 my_local/noise_distortion/prepare_noise_dir.py

# Extract 40-D MFCC
data_dir="train_cleaned_noise_mismatch_sp_hires_40"
steps/make_mfcc.sh --nj 20 --mfcc-config conf/mfcc_hires40.conf \
    --cmd "$train_cmd" data/sdm1/$data_dir
steps/compute_cmvn_stats.sh data/sdm1/$data_dir
utils/fix_data_dir.sh data/sdm1/$data_dir

# Comb
src=data/sdm1/${data_dir}
dest=data/sdm1/${data_dir}_comb
min_seg_len=1.55
utils/data/combine_short_segments.sh $src $min_seg_len $dest
