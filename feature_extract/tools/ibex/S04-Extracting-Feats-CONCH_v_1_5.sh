#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J PIBD
#SBATCH -o PIBD.%J.out
#SBATCH -e PIBD.%J.err
#SBATCH --mail-user=yilan.zhang@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --constraint=a100
#SBATCH --account conf-aaai-2025.08.04-gaox
set -e

module load cuda/11.7.1

source "/ibex/user/zhany0x/miniconda3/bin/activate"
conda activate PIBD

cd /ibex/user/zhany0x/project/GeneralPIBD/feature_extract/tools/ibex/

######################################################################
# Please carefully read the notes (1,2,3,4,5,6) for a successful run
######################################################################

# Sample patches of SIZE x SIZE at MAG (as used in S03)
# Note 1: following the setting of TITAN, MAG (magnification) should be set to 20
MAG=20
# Note 2: following the setting of TITAN, SIZE (the patch size at 20x) should be set to 512
SIZE=512
# Note 3: following the setting of CONCH_v1.5, TARGET_PATCH_SIZE should be set to 448
TARGET_PATCH_SIZE=448

# Path where CLAM is installed
DIR_REPO=../CLAM

#study name
STUDY=stad
# Root path to pathology images
DIR_RAW_DATA=/ibex/project/c2277/data/Pathology/Slides/${STUDY}
DIR_EXP_DATA=/ibex/project/c2277/data/Pathology/Slides/${STUDY}_patch

# Sub-directory to the patch coordinates generated from S03
SUBDIR_READ=tiles-${MAG}x-s${SIZE}

# Arch to be used for patch feature extraction (CONCH is strongly recommended)
ARCH=CONCH_v1.5

# Model path
# Note 4: You need to first apply for its access rights via https://huggingface.co/MahmoodLab/TITAN
# and then download the whole repo to your local path, e.g., /path/to/TITAN
# Note 5: /path/to/TITAN/conch_tokenizer.py: modify the code at line 17: "MahmoodLab/TITAN" to "/path/to/TITAN"
# Note 6: /path/to/TITAN/conch_v1_5.py: modify the code at line 687: let checkpoint_path = "/path/to/TITAN"
#         and comment the lines 682-686
MODEL_CKPT=/ibex/project/c2277/data/Pathology/checkpoints/TITAN/

# Sub-directory to the patch features
SUBDIR_SAVE=/ibex/project/c2277/data/Pathology/${ARCH}/${STUDY}

cd ${DIR_REPO}

echo "running for extracting features from all tiles"
CUDA_VISIBLE_DEVICES=0 python3 extract_features_fp.py \
    --arch ${ARCH} \
    --ckpt_path ${MODEL_CKPT} \
    --data_h5_dir ${DIR_EXP_DATA}/${SUBDIR_READ} \
    --data_slide_dir ${DIR_RAW_DATA} \
    --csv_path ${DIR_EXP_DATA}/${SUBDIR_READ}/process_list_autogen.csv \
    --feat_dir ${SUBDIR_SAVE} \
    --target_patch_size ${TARGET_PATCH_SIZE} \
    --batch_size 512 \
    --slide_ext .svs \
    --auto_skip