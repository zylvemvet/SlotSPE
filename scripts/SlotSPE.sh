#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J SlotSPE
#SBATCH -o SlotSPE.%J.out
#SBATCH -e SlotSPE.%J.err
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --constraint=a100

module load cuda/11.7.1

eval "$(conda shell.bash hook)"
conda activate slotspe

cd /ibex/user/zhany0x/project/SlotSPE

DATA_ROOT_DIR="/ibex/project/c2277/data/Pathology/UNI" # where are the TCGA features stored?
RESULT_DIR="/ibex/user/zhany0x/experiment/survival_camera_ready/" # where is the results stored?
DATA_PATH='/ibex/user/zhany0x/project/SlotSPE/dataset_csv/'
#TYPE_OF_PATH="combine" # what type of pathways?
SEEDS=(3 2 1)
STUDIES=("brca" "coadread" "lusc" "hnsc")
TEMPS=(0.01)
TOP_K=(0.25)
METHODS=("SlotSPE")
SLOTS_WSI=(8)
SLOTS_OMICS=(8)
SLOT_ITERS=(10)

for TOPK in ${TOP_K[@]}; do
for TEMP in ${TEMPS[@]}; do
for SLOTS_W in ${SLOTS_WSI[@]}; do
for SLOTS_O in ${SLOTS_OMICS[@]}; do
for SEED in ${SEEDS[@]}; do
for STUDY in ${STUDIES[@]}; do
for METHOD in ${METHODS[@]}; do
for SLOT_ITERS in ${SLOT_ITERS[@]}; do
python survival.py \
--data_root_dir "${DATA_ROOT_DIR}/${STUDY}/pt_files" \
--results_dir $RESULT_DIR \
--data_path $DATA_PATH \
--specific_simple "iter_${SLOT_ITERS}_temp_${TEMP}_topk_${TOPK}_missing" \
--n_classes 4 \
--num_patches 4096 \
--encoding_dim 1024 \
--max_epochs 30 \
--batch_size 32 \
--seed $SEED \
--study $STUDY \
--method $METHOD \
--rna_format "Pathways" \
--label_col "survival_months_dss" \
--bag_loss "nll_surv" \
--signature "combine" \
--slot_num_omics $SLOTS_O \
--slot_num_wsi $SLOTS_W \
--slot_iters $SLOT_ITERS \
--temperature $TEMP \
--topk_ratio $TOPK \
--omic_missing \
--top_k_method "parallel_topk_st"
done
done
done
done
done
done
done
done
wait