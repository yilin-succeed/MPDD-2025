import os
import json
import torch
import numpy as np
import pandas as pd
import argparse
import logging
import pickle
from torch.utils.data import DataLoader, Dataset, Subset
from model.model2 import FullModel3
from dataset import *
from dtrain import *
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def prepare_user_history(dataset, device):
    """准备用户历史特征数据"""
    user_data_dict = {}
    
    for i in range(len(dataset)):
        user_id, personality_feat, visual_feats, audio_feats, _ = dataset[i]
        visual_deep_feat, visual_predesigned_feat = visual_feats
        audio_deep_feat, audio_predesigned_feat = audio_feats
        
        if user_id not in user_data_dict:
            user_data_dict[user_id] = {
                'visual_deep': [],
                'visual_predesigned': [],
                'audio_deep': [],
                'audio_predesigned': []
            }
        
        user_data_dict[user_id]['visual_deep'].append(visual_deep_feat)
        user_data_dict[user_id]['visual_predesigned'].append(visual_predesigned_feat)
        user_data_dict[user_id]['audio_deep'].append(audio_deep_feat)
        user_data_dict[user_id]['audio_predesigned'].append(audio_predesigned_feat)
    
    # 转换为张量并移到设备上
    user_history_data = {}
    for user_id, features in user_data_dict.items():
        user_history_data[user_id] = {
            'visual_deep': torch.stack(features['visual_deep']).to(device),
            'visual_predesigned': torch.stack(features['visual_predesigned']).to(device),
            'audio_deep': torch.stack(features['audio_deep']).to(device),
            'audio_predesigned': torch.stack(features['audio_predesigned']).to(device)
        }
    
    return user_history_data

def main():
    parser = argparse.ArgumentParser(description="Generate predictions for test data")
    parser.add_argument('--labelcount', type=int, default=2, help="Number of data categories (2 or 3)")
    parser.add_argument('--track_option', type=str, required=True, help="Track1 or Track2")
    parser.add_argument('--data_rootpath', type=str, required=True, help="Root path to test dataset")
    parser.add_argument('--model_path', type=str, required=True, help="Path to model checkpoints")
    parser.add_argument('--audio_deep_method', type=str, default='wav2vec', help="Deep audio feature method")
    parser.add_argument('--audio_predesigned_method', type=str, default='myopensmile', help="Predesigned audio feature")
    parser.add_argument('--visual_deep_method', type=str, default='densenet', help="Deep visual feature method")
    parser.add_argument('--visual_predesigned_method', type=str, default='openface', help="Predesigned visual feature")
    parser.add_argument('--splitwindow_time', type=str, default='5s', help="Time window (e.g. '5s')")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for testing")
    parser.add_argument('--device', type=str, default='cuda', help="Device for testing")
    parser.add_argument('--num_folds', type=int, default=5, help="Number of folds")
    parser.add_argument('--feature_max_len', type=int, default=10, help="Max feature length")
    parser.add_argument('--user_feature_method', type=str, default='lstm', choices=['mean', 'lstm'], help="User feature method")
    parser.add_argument('--personality_file', type=str, default=None, help="File name of the personalized features file (optional)")
    parser.add_argument('--max_components', type=int, default=10, help="Maximum number of components for DP-GMM")
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 设置特征路径
    audio_deep_path = os.path.join(args.data_rootpath, f"{args.splitwindow_time}", 'Audio', f"{args.audio_deep_method}")
    audio_predesigned_path = os.path.join(args.data_rootpath, f"{args.splitwindow_time}", 'Audio', f"{args.audio_predesigned_method}")
    visual_deep_path = os.path.join(args.data_rootpath, f"{args.splitwindow_time}", 'Visual', f"{args.visual_deep_method}")
    visual_predesigned_path = os.path.join(args.data_rootpath, f"{args.splitwindow_time}", 'Visual', f"{args.visual_predesigned_method}")
    args.personality_file = os.path.join('/public/home/hpc244712231/20250428/test/MPDD-Young/labels/personalized_test.json')
    personality_file=args.personality_file
    # 模型路径
    model_dir = os.path.join(
        args.model_path, 
        f"{args.labelcount}-{args.splitwindow_time}-{args.audio_deep_method}+{args.audio_predesigned_method}-"
        f"{args.visual_deep_method}+{args.visual_predesigned_method}-user_{args.user_feature_method}"
    )
    logger.info(f"Using model directory: {model_dir}")
    
    # 加载人格特征
    personality_file = os.path.join(args.data_rootpath, 'labels', 'processed_data.json')
    logger.info(f"Loading personality features from {personality_file}")
    personality_features,test_id = load_personality_features(personality_file)
    logger.info(f"Loaded {len(personality_features)} users with personality features")
    
    # 加载测试数据
    test_json_path = os.path.join(args.data_rootpath, 'labels', 'Testing_files.json')
    logger.info(f"Loading test data from {test_json_path}")
    test_samples = load_train_data2(test_json_path, personality_features)
    logger.info(f"Loaded {len(test_samples)} test samples")
    
    # 创建测试数据集
    test_dataset = MultiModalPersonalityDataset4(
        samples=test_samples,
        personality_features=personality_features,
        label_count=args.labelcount,
        audio_deep_path=audio_deep_path,
        audio_predesigned_path=audio_predesigned_path,
        visual_deep_path=visual_deep_path,
        visual_predesigned_path=visual_predesigned_path,
        isTest=True
    )
    logger.info(f"Created test dataset with {len(test_dataset)} samples")
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=lambda batch: fixed_length_collate_fn(
            batch, 
            args.feature_max_len,
            args.visual_deep_method,
            args.visual_predesigned_method,
            args.audio_deep_method,
            args.audio_predesigned_method
        )
    )
    
    # 准备用户历史数据
    logger.info("Preparing user history data...")
    user_history_data = prepare_user_history(test_dataset, device)
    logger.info(f"Prepared history data for {len(user_history_data)} users")
    
    # 加载单个指定模型
    model = None
    model_weights_path = os.path.join('/public/home/hpc244712231/20250428/MPDDmy/kfold_checkpoints/2-1s-wav2vec+mfccs-densenet+openface-user_lstm/best_model_f1_0.9434.pth')
    
    if not os.path.exists(model_weights_path):
        logger.error(f"Model weights file not found: {model_weights_path}")
        return
    cluster_info_path = os.path.join('/public/home/hpc244712231/20250428/MPDDmy/av+feat-kfold_checkpoints/cluster_info.pkl')
    with open(cluster_info_path, 'rb') as f:
        cluster_info = pickle.load(f)
        # 更新聚类数量
    actual_k = cluster_info['actual_k']
    # 初始化模型
    visual_feature_types = [args.visual_deep_method, args.visual_predesigned_method]
    audio_feature_types = [args.audio_deep_method, args.audio_predesigned_method]
    if personality_file:
        model = FullModel3(
                  K=15,
                  p2k_hidden=32, 
                  p2k_out_dim=64,
                  iafn_embed_dim=64, 
                  iafn_heads=2, 
                  gate_embed_dim=64,
                  visual_feature_types=visual_feature_types, 
                  audio_feature_types=audio_feature_types,
                  num_classes=args.labelcount,
                  dropout=0.5
              ).to(device)
    else:
        model = FullModel3(
                K=args.max_components,
                p2k_hidden=32, 
                p2k_out_dim=64,
                iafn_embed_dim=64, 
                iafn_heads=2, 
                gate_embed_dim=64,
                visual_feature_types=visual_feature_types, 
                audio_feature_types=audio_feature_types,
                num_classes=args.labelcount,
                dropout=0.5
            ).to(device)
        model.has_personality = False
    '''
    # 尝试加载聚类信息
    cluster_info_path = os.path.join(model_dir, 'cluster_info.pkl')
    
    if os.path.exists(cluster_info_path):
        try:
            # 从文件加载聚类信息
            model.load_clustering_info(cluster_info_path, device)
            logger.info("Loaded clustering info")
        except Exception as e:
            logger.warning(f"Failed to load clustering info: {e}")
    '''      

    # 加载模型权重
    try:
        state_dict = torch.load(model_weights_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"Successfully loaded weights from {model_weights_path}")
        model.eval()
    except Exception as e:
        logger.error(f"Error loading model weights: {e}")
        return
        
    logger.info("Model loaded successfully")
    
    # 进行预测
    all_user_ids = []
    all_original_paths = []
    pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            user_ids = batch[0]
            all_user_ids.extend(user_ids)
            
            # 保存原始路径，用于后面格式化ID
            if hasattr(test_dataset, 'samples'):
                batch_indices = [i for i in range(len(test_dataset)) if test_dataset.samples[i]['user_id'] in user_ids]
                batch_paths = [test_dataset.samples[i]['audio_feature_path'] for i in batch_indices]
                all_original_paths.extend(batch_paths)
            
            # 将张量转移到设备
            personality_feat = batch[1].to(device) if batch[1] is not None else None
            visual_deep_feat = batch[2].to(device) if batch[2] is not None else None
            visual_predesigned_feat = batch[3].to(device) if batch[3] is not None else None
            audio_deep_feat = batch[4].to(device) if batch[4] is not None else None
            audio_predesigned_feat = batch[5].to(device) if batch[5] is not None else None
            
            # 单模型预测
            logits, _ = model(personality_feat, visual_deep_feat, visual_predesigned_feat,  
                             audio_deep_feat, audio_predesigned_feat,
                             None, None, user_ids, user_history_data)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            pred.extend(preds)  # 注意：这里改为extend而不是append

    test_ids = ['_'.join([part.lstrip('0') for part in item["audio_feature_path"].replace(".npy", "").split('_')]) for item in test_samples]
    
    # 确定输出列名
    pred_col_name = f"{args.splitwindow_time}_{'bin' if args.labelcount == 2 else 'tri'}"
    
    # 输出到CSV
    result_dir = f"./answer_{args.track_option}"
    os.makedirs(result_dir, exist_ok=True)
    csv_file = f"{result_dir}/submission.csv"
    
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=["ID"])

    if "ID" in df.columns:
        df = df.set_index("ID")  
    else:
        df = pd.DataFrame(index=test_ids)

    df.index.name = "ID"

    pred = np.array(pred) 
    if len(pred) != len(test_ids):
        logger.error(f"Prediction length {len(pred)} does not match test ID length {len(test_ids)}")
        raise ValueError("Mismatch between predictions and test IDs")

    new_df = pd.DataFrame({pred_col_name: pred}, index=test_ids)
    df[pred_col_name] = new_df[pred_col_name]
    df = df.reindex(test_ids)
    df.to_csv(csv_file)

    logger.info(f"Testing complete. Results saved to: {csv_file}.")

if __name__ == '__main__':
    main()
