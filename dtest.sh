#!/bin/bash
#SBATCH -A pi_liushu       # 替换为你的账户名
#SBATCH -p gpu8Q           # 替换为你的 GPU 分区名称
#SBATCH -N 1               # 只请求 1 个节点
#SBATCH --gres=gpu:1       # 请求 1 个 GPU
#SBATCH --ntasks=1         # 启动 1 个任务
#SBATCH --cpus-per-task=4  # 每个任务分配 4 个 CPU 核心
#SBATCH -J test_model      # 作业名称
#SBATCH -q gpuq
set -e

# Default Testing Parameters
DATA_ROOTPATH="/public/home/hpc244712231/20250428/test/MPDD-Young"
MODEL_PATH="/public/home/hpc244712231/20250428/MPDDmy/kfold_checkpoints"
AUDIO_DEEP_METHOD="wav2vec"
AUDIO_PREDESIGNED_METHOD="mfccs"
VISUAL_DEEP_METHOD="densenet"
VISUAL_PREDESIGNED_METHOD="openface"
SPLITWINDOW="5s"
LABELCOUNT=2
TRACK_OPTION="Track2"
FEATURE_MAX_LEN=45       
BATCH_SIZE=1
DEVICE="cuda"
N_FOLDS=5
PERSONALITY_FILE=""

# 解析命令行参数
for arg in "$@"; do
  case $arg in
    --data_rootpath=*) DATA_ROOTPATH="${arg#*=}" ;;
    --model_path=*) MODEL_PATH="${arg#*=}" ;;
    --audio_deep_method=*) AUDIO_DEEP_METHOD="${arg#*=}" ;;
    --audio_predesigned_method=*) AUDIO_PREDESIGNED_METHOD="${arg#*=}" ;;
    --visual_deep_method=*) VISUAL_DEEP_METHOD="${arg#*=}" ;;
    --visual_predesigned_method=*) VISUAL_PREDESIGNED_METHOD="${arg#*=}" ;;
    --splitwindow_time=*) SPLITWINDOW="${arg#*=}" ;;
    --feature_max_len=*) FEATURE_MAX_LEN="${arg#*=}" ;;
    --labelcount=*) LABELCOUNT="${arg#*=}" ;;
    --track_option=*) TRACK_OPTION="${arg#*=}" ;;
    --batch_size=*) BATCH_SIZE="${arg#*=}" ;;
    --device=*) DEVICE="${arg#*=}" ;;
    --num_folds=*) N_FOLDS="${arg#*=}" ;;
    --personality_file=*) PERSONALITY_FILE="${arg#*=}" ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

# 构建测试命令
cmd="python dtest.py \
    --data_rootpath=$DATA_ROOTPATH \
    --model_path=$MODEL_PATH \
    --audio_deep_method=$AUDIO_DEEP_METHOD \
    --audio_predesigned_method=$AUDIO_PREDESIGNED_METHOD \
    --visual_deep_method=$VISUAL_DEEP_METHOD \
    --visual_predesigned_method=$VISUAL_PREDESIGNED_METHOD \
    --splitwindow_time=$SPLITWINDOW \
    --labelcount=$LABELCOUNT \
    --feature_max_len=$FEATURE_MAX_LEN \
    --track_option=$TRACK_OPTION \
    --batch_size=$BATCH_SIZE \
    --device=$DEVICE \
    --num_folds=$N_FOLDS"

# 如果指定了personality_file参数，添加到命令中
if [ -n "$PERSONALITY_FILE" ]; then
    cmd="$cmd --personality_file=$PERSONALITY_FILE"
fi

echo -e "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh
