# kaldi-SENAN

kaldi-SENAN is the implementation of speech-enhanced and noise-aware network (SENAN, see the following paper) built on the open-sourced Kaldi toolkit.
Example scripts for Aurora-4 task are provided and located at egs/aurora4/proposed. Scripts for AMI task are also provided.

>Hung-Shin Lee, Pin-Yuan Chen, Yu Tsao, and Hsin-Min Wang, "[Speech-enhanced and noise-aware networks for robust speech recognition](https://arxiv.org/abs/2203.13696)," submitted to Interspeech 2022.

---

## Prerequisites

Follow kaldi installation steps and install this project.

## Aurora-4 example

1. In stage 8 of `run.sh`, change command to

```bash
# TDNN-F as AM + proposed model

local/chain/tuning/run_tdnn-1a_mtae_mfcc-mfcc-cont_noise-stats.sh
```

```bash
# TDNN-F as AM + SpecAugment + proposed model

local/chain/tuning/run_tdnn-1a_mtae_mfcc-mfcc-cont_noise-stats_specaugment.sh 
```

```bash
# CNN-TDNN-F as AM + proposed model

local/chain/tuning/run_cnn-tdnn-1c_mtae_mfcc-mfcc-cont_noise-stats.sh
```

```bash
# CNN-TDNN-F as AM + SpecAugment + proposed model

local/chain/tuning/run_cnn-tdnn-1c_mtae_mfcc-mfcc-cont_noise-stats_specaugment.sh 
```

2. The weight for the two output layers can be changed by modifying frame_weight_dae and frame_weight_dspae in `run_{tdnn-1a,cnn-tdnn-1c}\_mtae\_*.sh`

## AMI example

1. In stage 11 of `run.sh`, change command to

```bash
# CNN-TDNN-F as AM + SpecAugment

local/chain/tuning/run_cnn-tdnn-1c_specaugment.sh
```

```bash
# TDNN-F as AM + SpecAugment + proposed model

local/chain/tuning/run_cnn-tdnn-1c_mtae_fbank-mfcc-t_noise-t_specaugment.sh
```

