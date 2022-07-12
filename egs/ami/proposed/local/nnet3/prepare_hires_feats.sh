#!/usr/bin/env bash

set -e -o pipefail

# this is called from the other nnet and chain training scripts.
# It prepares high-resolution MFCC features for purposes of getting
# clean MFCC on the speed-perturbed data.
# These will either be from the IHM data (with the --use-ihm-ali true option),
# or with the target data as given by the --mic option.
#
# please see local/nnet3/run_tdnn.sh and local/chain/run_tdnn.sh for examples of
# usage.

stage=0
mic=sdm1
nj=30
min_seg_len=1.55  # min length in seconds... we do this because chain training
                  # will discard segments shorter than 1.5 seconds.  Must remain in
                  # sync with the same option given to run_ivector_common.sh.
                  # Set it to empty string to skip combining segments.
use_ihm_ali=true # If true, we use alignments from the IHM data (which is better..
                  # don't set this to true if $mic is set to ihm.)
train_set=train_cleaned   # you might set this to e.g. train_cleaned.
hires_suffix=40 # 40 for 40D, 80 for 80D

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if $use_ihm_ali; then
  ihm_suffix=_ihmdata
  maybe_ihm="IHM " # for printed messages
  [ "$mic" == "ihm" ] && \
    echo "$0: you cannot specify --use-ihm-ali true if the microphone is ihm." && \
    exit 1;
else
  ihm_suffix=
  maybe_ihm=
fi

for f in data/${mic}/${train_set}${ihm_suffix}/utt2spk; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

if [ $stage -le 8 ]; then
  echo "$0: preparing directory for ${maybe_ihm}speed-perturbed data (for alignment)"
  utils/copy_data_dir.sh data/${mic}/${train_set}${ihm_suffix}_sp data/${mic}/${train_set}${ihm_suffix}_sp_hires_${hires_suffix}
  
  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  utils/data/perturb_data_dir_volume.sh data/${mic}/${train_set}${ihm_suffix}_sp_hires_${hires_suffix}

fi

if [ $stage -le 9 ]; then
  echo "$0: making MFCC features for speed-perturbed ${maybe_ihm}data"
  steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires${hires_suffix}.conf \
    --cmd "$train_cmd" data/${mic}/${train_set}${ihm_suffix}_sp_hires_${hires_suffix}
  steps/compute_cmvn_stats.sh data/${mic}/${train_set}${ihm_suffix}_sp_hires_${hires_suffix}
  echo "$0: fixing input data-dir to remove nonexistent features, in case some "
  echo ".. speed-perturbed segments were too short."
  utils/fix_data_dir.sh data/${mic}/${train_set}${ihm_suffix}_sp_hires_${hires_suffix}
fi

if [ ! -z "$min_seg_len" ]; then
  if [ $stage -le 10 ]; then
    echo "$0: combining short segments of 40-dimensional speed-perturbed ${maybe_ihm}MFCC data"
    src=data/${mic}/${train_set}${ihm_suffix}_sp_hires_${hires_suffix}
    dest=data/${mic}/${train_set}${ihm_suffix}_sp_hires_${hires_suffix}_comb
    utils/data/combine_short_segments.sh $src $min_seg_len $dest
    # re-use the CMVN stats from the source directory, since it seems to be slow to
    # re-compute them after concatenating short segments.
    cp $src/cmvn.scp $dest/
    utils/fix_data_dir.sh $dest
  fi
fi


exit 0;
