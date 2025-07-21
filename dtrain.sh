#!/bin/bash
#SBATCH -A pi_liushu        # 替换为你的账户名
#SBATCH -p gpu4Q            # 替换为你的 GPU 分区名称
#SBATCH -N 1                # 只请求 1 个节点
#SBATCH --gres=gpu:1        # 请求 1 个 GPU
#SBATCH --ntasks=1          # 启动 1 个任务
#SBATCH --cpus-per-task=4   # 每个任务分配 4 个 CPU 核心
#SBATCH -J dpgmm_train      # 作业名称
#SBATCH -q gpuq

set -e

# 默认训练参数
data_rootpath="/public/home/hpc244712231/20250428/MPDD-Young" # 数据集根目录
# 修改为支持四种特征类型
AUDIO_DEEP_METHOD="wav2vec"
AUDIO_PREDESIGNED_METHOD="mfccs"
VISUAL_DEEP_METHOD="densenet"
VISUAL_PREDESIGNED_METHOD="openface"
SPLITWINDOW="1s"                      # 窗口时长
LABELCOUNT=2                       # 标签类别数量
TRACK_OPTION="Track2"
FEATURE_MAX_LEN=200                     # 最大特征长度
BATCH_SIZE=8
NUM_EPOCHS=500
DEVICE="cuda"

# DP-GMM特定参数
MAX_COMPONENTS=15                     # DP-GMM最大组件数
CONCENTRATION_PRIOR=0.01              # DP浓度先验
LR=0.0004                             # 学习率

# 用户级特征参数
USE_USER_FEATURES=true               # 是否使用用户级特征
USER_FEATURE_METHOD="lstm"            # 用户特征聚合方法

# 伪标签生成相关参数
GENERATE_LABELS=false                 # 是否生成伪标签
MODEL_PATH=""                         # 预训练模型路径
CONFIDENCE_THRESHOLD=0.99             # 伪标签置信度阈值

# 解析命令行参数
for arg in "$@"; do
  case $arg in
    --data_rootpath=*) data_rootpath="${arg#*=}" ;;
    # 更新为四种特征类型参数
    --audio_deep_method=*) AUDIO_DEEP_METHOD="${arg#*=}" ;;
    --audio_predesigned_method=*) AUDIO_PREDESIGNED_METHOD="${arg#*=}" ;;
    --visual_deep_method=*) VISUAL_DEEP_METHOD="${arg#*=}" ;;
    --visual_predesigned_method=*) VISUAL_PREDESIGNED_METHOD="${arg#*=}" ;;
    --splitwindow_time=*) SPLITWINDOW="${arg#*=}" ;;
    --labelcount=*) LABELCOUNT="${arg#*=}" ;;
    --track_option=*) TRACK_OPTION="${arg#*=}" ;;
    --feature_max_len=*) FEATURE_MAX_LEN="${arg#*=}" ;;
    --batch_size=*) BATCH_SIZE="${arg#*=}" ;;
    --num_epochs=*) NUM_EPOCHS="${arg#*=}" ;;
    --device=*) DEVICE="${arg#*=}" ;;
    --max_components=*) MAX_COMPONENTS="${arg#*=}" ;;
    --concentration_prior=*) CONCENTRATION_PRIOR="${arg#*=}" ;;
    --lr=*) LR="${arg#*=}" ;;
    # 添加用户特征参数
    --use_user_features) USE_USER_FEATURES=true ;;
    --user_feature_method=*) USER_FEATURE_METHOD="${arg#*=}" ;;
    --generate_labels) GENERATE_LABELS=true ;;
    --model_path=*) MODEL_PATH="${arg#*=}" ;;
    --confidence_threshold=*) CONFIDENCE_THRESHOLD="${arg#*=}" ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

# 构建基本命令，更新为支持四种特征类型
cmd="python dtrain.py \
    --data_rootpath=$data_rootpath \
    --audio_deep_method=$AUDIO_DEEP_METHOD \
    --audio_predesigned_method=$AUDIO_PREDESIGNED_METHOD \
    --visual_deep_method=$VISUAL_DEEP_METHOD \
    --visual_predesigned_method=$VISUAL_PREDESIGNED_METHOD \
    --splitwindow_time=$SPLITWINDOW \
    --labelcount=$LABELCOUNT \
    --track_option=$TRACK_OPTION \
    --feature_max_len=$FEATURE_MAX_LEN \
    --batch_size=$BATCH_SIZE \
    --num_epochs=$NUM_EPOCHS \
    --device=$DEVICE \
    --max_components=$MAX_COMPONENTS \
    --concentration_prior=$CONCENTRATION_PRIOR \
    --lr=$LR"

# 添加用户特征相关参数
if [ "$USE_USER_FEATURES" = true ]; then
    cmd="$cmd --use_user_features --user_feature_method=$USER_FEATURE_METHOD"
fi

# 添加伪标签生成相关参数
if [ "$GENERATE_LABELS" = true ]; then
    cmd="$cmd --generate_labels"
    
    if [ ! -z "$MODEL_PATH" ]; then
        cmd="$cmd --model_path=$MODEL_PATH"
    fi
    
    cmd="$cmd --confidence_threshold=$CONFIDENCE_THRESHOLD"
fi

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh
