# kaldi-SENAN
kaldi-SENAN is the implementation of speech-enhanced and noise-aware network (SENAN) built on open source Kaldi toolkit. 
Example scripts for Aurora-4 task are provided and located at egs/aurora4/proposed.

## Prerequisites
Follow kaldi installation steps and install this project

## Aurora-4 example
1. In stage 8 of run.sh, change command to 
```
local/chain/tuning/run_tdnn-1a_mtae_mfcc-mfcc-cont_noise-stats.sh # TDNN-F as AM + proposed model 
```
, 
```
local/chain/tuning/run_tdnn-1a_mtae_mfcc-mfcc-cont_noise-stats_specaugment.sh  # TDNN-F as AM + SpecAugment + proposed model 
```
, 
```
local/chain/tuning/run_cnn-tdnn-1c_mtae_mfcc-mfcc-cont_noise-stats.sh # CNN-TDNN-F as AM + proposed model 
```
or 
```
local/chain/tuning/run_cnn-tdnn-1c_mtae_mfcc-mfcc-cont_noise-stats_specaugment.sh # CNN-TDNN-F as AM + SpecAugment + proposed model 
```

2. The weight for the two output layers can be changed by modifying frame_weight_dae and frame_weight_dspae in run_{tdnn-1a,cnn-tdnn-1c}\_mtae\_*.sh
