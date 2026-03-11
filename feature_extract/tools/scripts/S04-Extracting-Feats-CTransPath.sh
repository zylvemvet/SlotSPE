#!/bin/bash
set -e

# Sample patches of SIZE x SIZE at MAG (as used in S03)
MAG=20
SIZE=256

# Path where CLAM is installed
DIR_REPO=../CLAM

#study name
STUDY=stad
# Root path to pathology images
DIR_RAW_DATA=/Data/Pathology/Slides/${STUDY}
DIR_EXP_DATA=/Data/Pathology/Slides/${STUDY}_patch

# Sub-directory to the patch coordinates generated from S03
SUBDIR_READ=tiles-${MAG}x-s${SIZE}

# Arch to be used for patch feature extraction (CONCH is strongly recommended)
ARCH=CTransPath

# Model path 
# You need to download it from https://drive.google.com/file/d/1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX/view?usp=sharing
# and then name it as `ctranspath.pth`.
MODEL_CKPT=/Data/Pathology/checkpoints/ctranspath/ctranspath.pth

# Sub-directory to the patch features 
SUBDIR_SAVE=/Data/Pathology/${ARCH}/${STUDY}

cd ${DIR_REPO}

echo "running for extracting features from all tiles"
CUDA_VISIBLE_DEVICES=0 python3 extract_features_fp.py \
    --arch ${ARCH} \
    --ckpt_path ${MODEL_CKPT} \
    --data_h5_dir ${DIR_EXP_DATA}/${SUBDIR_READ} \
    --data_slide_dir ${DIR_RAW_DATA} \
    --csv_path ${DIR_EXP_DATA}/${SUBDIR_READ}/process_list_autogen.csv \
    --feat_dir ${SUBDIR_SAVE} \
    --target_patch_size ${SIZE} \
    --batch_size 128 \
    --slide_ext .svs \
    --auto_skip
