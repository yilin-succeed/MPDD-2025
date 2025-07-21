from datetime import datetime
from collections import defaultdict
import random
import os
import time
import argparse
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader, Subset
import numpy as np
import csv
from torch.utils.data import Sampler
from sklearn.cluster import KMeans
from model.model2 import FullModel2,FullModel3
from sklearn.model_selection import train_test_split
from dataset import *
import json
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import torch.nn.functional as F
from collections import Counter
class Opt:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)
from sklearn.model_selection import StratifiedKFold
from collections import Counter

# Helper function to find the best model in a directory
def find_best_model(model_dir):
    if not os.path.exists(model_dir):
        return None
        
    best_model_path = None
    best_f1 = -1
    
    for file in os.listdir(model_dir):
        if file.startswith('best_model_') and file.endswith('.pth'):
            try:
                if '_f1_' in file:
                    f1_str = file.split('_f1_')[1].split('.pth')[0]
                    f1 = float(f1_str)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model_path = os.path.join(model_dir, file)
            except Exception as e:
                continue
                
    return best_model_path



from sklearn.model_selection import StratifiedKFold
import numpy as np
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0.0, path_dir='./'):
        """
        Args:
            patience (int): 多少epoch不提升时停止训练。
            verbose (bool): 是否打印信息。
            delta (float): 提升的阈值（微小改进视为无提升）。
            path_dir (str): 保存模型的目录路径。
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path_dir = path_dir
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_metric = -np.inf
        self.best_path = None

    def __call__(self, metric, model):
        score = metric
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, score)
            self.best_metric = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                if self.verbose:
                    print("Early stopping triggered.")
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, score)
            self.best_metric = score
            self.counter = 0
    
    def save_checkpoint(self, model, score):
        """保存模型"""
        self.best_path = os.path.join(self.path_dir, f'best_model_f1_{score:.4f}.pth')
        torch.save(model.state_dict(), self.best_path)
        if self.verbose:
            print(f"Saving model with metric: {score:.4f}")




# 伪标签数据集类
class PseudoLabeledDataset(Dataset):
    def __init__(self, base_dataset, indices, pseudo_labels):
        self.base_dataset = base_dataset
        self.indices = indices
        self.pseudo_labels = pseudo_labels
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        data = self.base_dataset[self.indices[idx]]
        # 返回原始数据，但替换标签为伪标签
        return (data[0], self.pseudo_labels[idx]) + data[2:]

def fixed_length_collate_fn(batch, fixed_len=30, 
                          visual_deep_method='resnet', 
                          visual_predesigned_method='openface',
                          audio_deep_method='wav2vec',
                          audio_predesigned_method='myopensmile'):
    """
    根据指定的特征类型动态设置特征维度的collate函数
    
    Args:
        batch: 数据批次
        fixed_len: 固定序列长度
        visual_deep_method: 视觉深度特征方法 ('resnet', 'densenet')
        visual_predesigned_method: 视觉预设计特征方法 ('openface')
        audio_deep_method: 音频深度特征方法 ('wav2vec')
        audio_predesigned_method: 音频预设计特征方法 ('mfccs', 'opensmile', 'myopensmile')
    """
    user_ids, personality_feats, visual_feats_tuple, audio_feats_tuple, labels = zip(*batch)
    
    # 分离深度特征和预设计特征
    visual_deep_feats, visual_predesigned_feats = zip(*visual_feats_tuple)
    audio_deep_feats, audio_predesigned_feats = zip(*audio_feats_tuple)
    
    # 处理人格特征和标签
    personality_feats = torch.stack(personality_feats)
    labels = torch.stack(labels)
    
    # 根据特征类型设置维度
    if visual_deep_method == 'resnet':
        EXPECTED_VISUAL_DEEP_DIM = 1000
    elif visual_deep_method == 'densenet':
        EXPECTED_VISUAL_DEEP_DIM = 1000
    else:
        EXPECTED_VISUAL_DEEP_DIM = 1000  # 默认值
        
    # 视觉预设计特征维度
    EXPECTED_VISUAL_PREDESIGNED_DIM = 709  # openface 是唯一选项
    
    # 音频深度特征维度
    EXPECTED_AUDIO_DEEP_DIM = 512  # wav2vec 是唯一选项
    
    # 音频预设计特征维度
    if audio_predesigned_method == 'mfccs':
        EXPECTED_AUDIO_PREDESIGNED_DIM = 64
    elif audio_predesigned_method in ['opensmile', 'myopensmile']:
        EXPECTED_AUDIO_PREDESIGNED_DIM = 6373
    else:
        EXPECTED_AUDIO_PREDESIGNED_DIM = 64  # 默认值

    # 处理特征
    padded_visual_deep_feats = []
    padded_visual_predesigned_feats = []
    padded_audio_deep_feats = []
    padded_audio_predesigned_feats = []
    
    # 处理视觉特征
    for deep_feat, predesigned_feat in zip(visual_deep_feats, visual_predesigned_feats):
        # 修正深度特征维度
        if deep_feat.shape[1] != EXPECTED_VISUAL_DEEP_DIM:
            if deep_feat.shape[1] < EXPECTED_VISUAL_DEEP_DIM:
                padding = torch.zeros(deep_feat.shape[0], EXPECTED_VISUAL_DEEP_DIM - deep_feat.shape[1])
                deep_feat = torch.cat([deep_feat, padding], dim=1)
            else:
                deep_feat = deep_feat[:, :EXPECTED_VISUAL_DEEP_DIM]
        
        # 修正预设计特征维度
        if predesigned_feat.shape[1] != EXPECTED_VISUAL_PREDESIGNED_DIM:
            if predesigned_feat.shape[1] < EXPECTED_VISUAL_PREDESIGNED_DIM:
                padding = torch.zeros(predesigned_feat.shape[0], EXPECTED_VISUAL_PREDESIGNED_DIM - predesigned_feat.shape[1])
                predesigned_feat = torch.cat([predesigned_feat, padding], dim=1)
            else:
                predesigned_feat = predesigned_feat[:, :EXPECTED_VISUAL_PREDESIGNED_DIM]
        
        # 固定序列长度
        seq_len = deep_feat.shape[0]
        if seq_len < fixed_len:
            padding_deep = torch.zeros(fixed_len - seq_len, EXPECTED_VISUAL_DEEP_DIM)
            padded_deep_feat = torch.cat([deep_feat, padding_deep], dim=0)
            
            padding_predesigned = torch.zeros(fixed_len - seq_len, EXPECTED_VISUAL_PREDESIGNED_DIM)
            padded_predesigned_feat = torch.cat([predesigned_feat, padding_predesigned], dim=0)
        else:
            padded_deep_feat = deep_feat[:fixed_len]
            padded_predesigned_feat = predesigned_feat[:fixed_len]
        
        padded_visual_deep_feats.append(padded_deep_feat)
        padded_visual_predesigned_feats.append(padded_predesigned_feat)
    
    # 处理音频特征
    for deep_feat, predesigned_feat in zip(audio_deep_feats, audio_predesigned_feats):
        # 修正深度特征维度
        if deep_feat.shape[1] != EXPECTED_AUDIO_DEEP_DIM:
            if deep_feat.shape[1] < EXPECTED_AUDIO_DEEP_DIM:
                padding = torch.zeros(deep_feat.shape[0], EXPECTED_AUDIO_DEEP_DIM - deep_feat.shape[1])
                deep_feat = torch.cat([deep_feat, padding], dim=1)
            else:
                deep_feat = deep_feat[:, :EXPECTED_AUDIO_DEEP_DIM]
        
        # 修正预设计特征维度
        if predesigned_feat.shape[1] != EXPECTED_AUDIO_PREDESIGNED_DIM:
            if predesigned_feat.shape[1] < EXPECTED_AUDIO_PREDESIGNED_DIM:
                padding = torch.zeros(predesigned_feat.shape[0], EXPECTED_AUDIO_PREDESIGNED_DIM - predesigned_feat.shape[1])
                predesigned_feat = torch.cat([predesigned_feat, padding], dim=1)
            else:
                predesigned_feat = predesigned_feat[:, :EXPECTED_AUDIO_PREDESIGNED_DIM]
        
        # 固定序列长度
        seq_len = deep_feat.shape[0]
        if seq_len < fixed_len:
            padding_deep = torch.zeros(fixed_len - seq_len, EXPECTED_AUDIO_DEEP_DIM)
            padded_deep_feat = torch.cat([deep_feat, padding_deep], dim=0)
            
            padding_predesigned = torch.zeros(fixed_len - seq_len, EXPECTED_AUDIO_PREDESIGNED_DIM)
            padded_predesigned_feat = torch.cat([predesigned_feat, padding_predesigned], dim=0)
        else:
            padded_deep_feat = deep_feat[:fixed_len]
            padded_predesigned_feat = predesigned_feat[:fixed_len]
        
        padded_audio_deep_feats.append(padded_deep_feat)
        padded_audio_predesigned_feats.append(padded_predesigned_feat)
    
    # 堆叠特征
    visual_deep_feats = torch.stack(padded_visual_deep_feats)
    visual_predesigned_feats = torch.stack(padded_visual_predesigned_feats)
    audio_deep_feats = torch.stack(padded_audio_deep_feats)
    audio_predesigned_feats = torch.stack(padded_audio_predesigned_feats)
    
    return user_ids, personality_feats, visual_deep_feats, visual_predesigned_feats, audio_deep_feats, audio_predesigned_feats, labels

def fixed_length_validate(model, val_loader, device, label_count=3):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            user_ids, personality_feat, visual_deep_feat, visual_predesigned_feat, audio_deep_feat, audio_predesigned_feat, label = batch
            
            # 将数据移到设备上
            personality_feat = personality_feat.to(device)
            visual_deep_feat = visual_deep_feat.to(device)
            visual_predesigned_feat = visual_predesigned_feat.to(device)
            audio_deep_feat = audio_deep_feat.to(device)
            audio_predesigned_feat = audio_predesigned_feat.to(device)
            label = label.long().to(device)
            
            # 前向传播 - 不再使用掩码
            logits, _ = model(personality_feat, visual_deep_feat, visual_predesigned_feat, 
                             audio_deep_feat, audio_predesigned_feat,
                             None, None, user_ids)  # 传递None作为mask
            _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    print("验证标签分布:", Counter(all_labels))
    print("验证预测分布:", Counter(all_preds))
    
    labels_all = list(range(label_count))
    accuracy = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    confusion = confusion_matrix(all_labels, all_preds, labels=labels_all)
    
    return accuracy, f1_weighted, f1_macro, confusion
'''
def fixed_length_train_loop(full_train_loader, val_loader, model, device, epochs, lr, lambda_reg, save_dir,
               monitor_metric='f1_weighted', label_count=3, early_stopping_patience=0, min_epochs=100,user_history_data=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_reg)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_metric = 0
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True,path_dir=save_dir)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        all_train_preds = []
        all_train_labels = []
        
        for batch_idx, batch in enumerate(full_train_loader):
            user_ids, personality_feat, visual_deep_feat, visual_predesigned_feat, audio_deep_feat, audio_predesigned_feat, label = batch
            
            # 将数据移到设备上
            personality_feat = personality_feat.to(device)
            visual_deep_feat = visual_deep_feat.to(device)
            visual_predesigned_feat = visual_predesigned_feat.to(device)
            audio_deep_feat = audio_deep_feat.to(device)
            audio_predesigned_feat = audio_predesigned_feat.to(device)
            label = label.long().to(device)
            
            optimizer.zero_grad()
            # 前向传播 - 不再使用掩码
            logits, _ = model(personality_feat, visual_deep_feat, visual_predesigned_feat, 
                             audio_deep_feat, audio_predesigned_feat,
                             None, None, user_ids, user_history_data)  # 传递None作为mask
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, preds = torch.max(logits, 1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(label.cpu().numpy())
        
        # 训练集整体指标
        epoch_acc = accuracy_score(all_train_labels, all_train_preds)
        epoch_f1w = f1_score(all_train_labels, all_train_preds, average='weighted')
        epoch_f1m = f1_score(all_train_labels, all_train_preds, average='macro')
        
        print(f"Epoch {epoch+1} Training - Loss: {running_loss/len(full_train_loader):.4f}, "
              f"Accuracy: {epoch_acc:.4f}, F1_w: {epoch_f1w:.4f}, F1_m: {epoch_f1m:.4f}")
        
        # 验证
        val_acc, val_f1w, val_f1m, val_cm = fixed_length_validate(model, val_loader, device, label_count)
        
        print(f"Epoch {epoch+1} Validation - Acc: {val_acc:.4f}, F1_w: {val_f1w:.4f}, F1_m: {val_f1m:.4f}")
        print("Confusion Matrix:\n", val_cm)
        
        metric_value = val_f1w if monitor_metric == 'f1_weighted' else (val_f1m if monitor_metric == 'f1_macro' else val_acc)
        
        # 保存模型
        if metric_value > best_metric:
            best_metric = metric_value
            best_model_path = os.path.join(save_dir, f'best_model_f1_{metric_value:.4f}.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"保存最佳模型到: {best_model_path}, 指标: {metric_value:.4f}")
        
        # 只有在完成最小训练周期后才检查早停条件
        if epoch + 1 >= min_epochs:
        # 查找当前目录中的最佳模型
            best_model_path = find_best_model(save_dir)
            
            if best_model_path:
                # 创建临时模型副本，加载找到的最佳权重
                temp_model = copy.deepcopy(model)
                temp_model.load_state_dict(torch.load(best_model_path, map_location=device))
                early_stopping(metric_value, temp_model)  # 传入加载了最佳权重的模型
            else:
                # 如果未找到最佳模型文件，使用当前模型
                early_stopping(metric_value, model)
                
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                # 确保模型加载最佳权重
                if best_model_path:
                    model.load_state_dict(torch.load(best_model_path, map_location=device))
                break
        else:
            print(f"当前epoch {epoch+1} < 最小训练周期 {min_epochs}，继续训练...")
    
    # 训练结束，加载保存的最佳模型
    return early_stopping.best_metric
'''

def fixed_length_train_loop(full_train_loader, val_loader, model, device, epochs, lr, lambda_reg, save_dir,
               monitor_metric='f1_weighted', label_count=3, early_stopping_patience=10, min_epochs=300,user_history_data=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_reg)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_metric = 0
    best_acc = 0
    best_f1w = 0
    best_f1m = 0
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True,path_dir=save_dir)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        all_train_preds = []
        all_train_labels = []
        
        for batch_idx, batch in enumerate(full_train_loader):
            user_ids, personality_feat, visual_deep_feat, visual_predesigned_feat, audio_deep_feat, audio_predesigned_feat, label = batch
            
            # 将数据移到设备上
            personality_feat = personality_feat.to(device)
            visual_deep_feat = visual_deep_feat.to(device)
            visual_predesigned_feat = visual_predesigned_feat.to(device)
            audio_deep_feat = audio_deep_feat.to(device)
            audio_predesigned_feat = audio_predesigned_feat.to(device)
            label = label.long().to(device)
            
            optimizer.zero_grad()
            logits, _ = model(personality_feat, visual_deep_feat, visual_predesigned_feat, 
                             audio_deep_feat, audio_predesigned_feat,
                             None, None, user_ids, user_history_data)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, preds = torch.max(logits, 1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(label.cpu().numpy())
        
        epoch_acc = accuracy_score(all_train_labels, all_train_preds)
        epoch_f1w = f1_score(all_train_labels, all_train_preds, average='weighted')
        epoch_f1m = f1_score(all_train_labels, all_train_preds, average='macro')
        
        print(f"Epoch {epoch+1} Training - Loss: {running_loss/len(full_train_loader):.4f}, "
              f"Accuracy: {epoch_acc:.4f}, F1_w: {epoch_f1w:.4f}, F1_m: {epoch_f1m:.4f}")
        
        val_acc, val_f1w, val_f1m, val_cm = fixed_length_validate(model, val_loader, device, label_count)
        
        print(f"Epoch {epoch+1} Validation - Acc: {val_acc:.4f}, F1_w: {val_f1w:.4f}, F1_m: {val_f1m:.4f}")
        print("Confusion Matrix:\n", val_cm)
        
        metric_value = val_f1w if monitor_metric == 'f1_weighted' else (val_f1m if monitor_metric == 'f1_macro' else val_acc)
        
        if metric_value > best_metric:
            best_metric = metric_value
            best_acc = val_acc
            best_f1w = val_f1w
            best_f1m = val_f1m
            best_model_path = os.path.join(save_dir, f'best_model_f1_{metric_value:.4f}.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"保存最佳模型到: {best_model_path}, 指标: {metric_value:.4f}")
        
        if epoch + 1 >= min_epochs:
            best_model_path = find_best_model(save_dir)
            if best_model_path:
                temp_model = copy.deepcopy(model)
                temp_model.load_state_dict(torch.load(best_model_path, map_location=device))
                early_stopping(metric_value, temp_model)
            else:
                early_stopping(metric_value, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                if best_model_path:
                    model.load_state_dict(torch.load(best_model_path, map_location=device))
                break
        else:
            print(f"当前epoch {epoch+1} < 最小训练周期 {min_epochs}，继续训练...")
    
    # 返回一个字典，包含多个最佳指标
    return {
        'accuracy': best_acc,
        'f1_weighted': best_f1w,
        'f1_macro': best_f1m
    }


def fixed_length_generate_pseudo_labels(model, unlabeled_loader, device, output_file, confidence_threshold=0.95):
    model.eval()
    all_predictions = []
    all_confidences = []
    all_user_ids = []
    all_pseudo_samples = []
    
    # 获取原始样本列表
    dataset = unlabeled_loader.dataset
    if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
        original_samples = dataset.dataset.samples
        sample_indices = dataset.indices
    else:
        original_samples = dataset.samples
        sample_indices = list(range(len(original_samples)))
    
    # 跟踪批次中样本的全局索引
    global_idx = 0
    
    with torch.no_grad():
        for batch in unlabeled_loader:
            # 从批次中获取数据
            user_ids, personality_feat, visual_deep_feat, visual_predesigned_feat, audio_deep_feat, audio_predesigned_feat, _ = batch
            
            # 将数据移到设备上
            personality_feat = personality_feat.to(device)
            visual_deep_feat = visual_deep_feat.to(device)
            visual_predesigned_feat = visual_predesigned_feat.to(device)
            audio_deep_feat = audio_deep_feat.to(device)
            audio_predesigned_feat = audio_predesigned_feat.to(device)
            
            # 前向传播 - 不再使用掩码
            outputs, _ = model(personality_feat, visual_deep_feat, visual_predesigned_feat, 
                             audio_deep_feat, audio_predesigned_feat, 
                             None, None, user_id=user_ids)  # 传递None作为mask
            probs = F.softmax(outputs, dim=1)
            
            # 获取预测和置信度
            confidences, predictions = torch.max(probs, dim=1)
            
            # 收集批次数据
            batch_size = len(user_ids)
            all_predictions.extend(predictions.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_user_ids.extend(user_ids)
            
            for i in range(batch_size):
                idx = sample_indices[global_idx + i]
                original_sample = original_samples[idx]
        
                # 创建伪标签样本 - 保留原始样本的所有字段，仅更改标签
                pseudo_sample = original_sample.copy()  # 复制原始样本的所有字段
                pseudo_sample['bin_category'] = int(predictions[i].item())
                pseudo_sample['confidence'] = float(confidences[i].item())
                all_pseudo_samples.append(pseudo_sample)
            
            global_idx += batch_size
    
    # 区分高置信度和所有预测
    high_confidence = [sample for sample in all_pseudo_samples if sample['confidence'] >= confidence_threshold]
    
    print(f"生成了 {len(high_confidence)} 个高置信度伪标签 (置信度阈值 >= {confidence_threshold})")
    print(f"总共 {len(all_pseudo_samples)} 个预测")
    print("预测概率示例（前10个样本）:", probs[:10].cpu().numpy())
    print("预测标签示例（前10个样本）:", predictions[:10].cpu().numpy())
    print("置信度示例（前10个样本）:", confidences[:10].cpu().numpy())

    # 保存伪标签
    result = {
        'high_confidence': high_confidence,
        'all': all_pseudo_samples
    }
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
        
    print(f"伪标签已保存至: {output_file}")
    
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MDPP Model with K-Fold Cross-Validation")
    # 基本参数
    parser.add_argument('--labelcount', type=int, default=3, help="Number of data categories (2, 3, or 5).")
    parser.add_argument('--track_option', type=str, required=True, help="Track1 or Track2")
    parser.add_argument('--feature_max_len', type=int, required=True, help="Max length of feature.")
    parser.add_argument('--data_rootpath', type=str, required=True, help="Root path to the program dataset")
    parser.add_argument('--train_json', type=str, required=False, help="File name of the training JSON file")
    parser.add_argument('--personalized_features_file', type=str, default=None, help="File name of the personalized features file (optional)")
    # 分别指定不同的特征类型
    parser.add_argument('--audio_deep_method', type=str, default='wav2vec', choices=['wav2vec'], help="Method for deep audio features.")
    parser.add_argument('--audio_predesigned_method', type=str, default='mfccs', choices=['mfccs', 'opensmile', 'myopensmile'], help="Method for predesigned audio features.")
    parser.add_argument('--visual_deep_method', type=str, default='resnet', choices=['resnet', 'densenet'], help="Method for deep visual features.")
    parser.add_argument('--visual_predesigned_method', type=str, default='openface', choices=['openface'], help="Method for predesigned visual features.")
    parser.add_argument('--splitwindow_time', type=str, default='1s', help="Time window for splitted features. e.g. '1s' or '5s'")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    parser.add_argument('--lambda_reg', type=float, default=5e-4, help="L2正则化强度")
    parser.add_argument('--num_epochs', type=int, default=1, help="Number of epochs to train the model")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to train the model on")
    parser.add_argument('--max_components', type=int, default=10, help="Maximum number of components for DP-GMM")
    parser.add_argument('--concentration_prior', type=float, default=1.0, help="Concentration prior for DP-GMM")
    parser.add_argument('--consistency_weight', type=float, default=0.1, help="Weight for consistency loss in semi-supervised learning")
    # K-fold特定参数
    parser.add_argument('--num_folds', type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument('--do_pseudo_labeling', action='store_true', help="Whether to use pseudo-labeling")
    parser.add_argument('--initial_epochs', type=int, default=300, help="Number of epochs for initial training")
    parser.add_argument('--pseudo_epochs', type=int, default=200, help="Number of epochs for each pseudo-label iteration")
    parser.add_argument('--confidence_threshold', type=float, default=0.9, help="Confidence threshold for pseudo labels")
    parser.add_argument('--max_iterations', type=int, default=3, help="Maximum number of pseudo-label iterations")
    # 用户级特征参数
    parser.add_argument('--use_user_features', action='store_true', help="Whether to use user-level features")
    parser.add_argument('--user_feature_method', type=str, default='lstm', choices=['mean', 'lstm'], help="Method for aggregating user-level features")
    args = parser.parse_args()
    
    # 设置固定长度参数
    FIXED_LENGTH = args.feature_max_len
    
    # 设置路径 - 分为深度特征和预设计特征两种路径
    audio_deep_path = os.path.join(args.data_rootpath, 'Training', f"{args.splitwindow_time}", 'Audio', f"{args.audio_deep_method}")
    audio_predesigned_path = os.path.join(args.data_rootpath, 'Training', f"{args.splitwindow_time}", 'Audio', f"{args.audio_predesigned_method}")
    visual_deep_path = os.path.join(args.data_rootpath, 'Training', f"{args.splitwindow_time}", 'Visual', f"{args.visual_deep_method}")
    visual_predesigned_path = os.path.join(args.data_rootpath, 'Training', f"{args.splitwindow_time}", 'Visual', f"{args.visual_predesigned_method}")
    
    args.train_json = os.path.join('/public/home/hpc244712231/20250428/MPDD-Young/Training/labels/Training_Validation_files2.json')
    args.personalized_features_file = None#os.path.join('/public/home/hpc244712231/20250428/MPDD-Young/Training/labels/personalized_train.json')
    personalized_features_file=os.path.join('/public/home/hpc244712231/20250428/MPDD-Young/Training/labels/personalized_train.json')
    # 加载标记数据的人格特征
    train_personality_features, train_user_ids = load_personality_features(personalized_features_file)
    # 加载未标记数据的人格特征
    unlabeled_personality_file = '/public/home/hpc244712231/20250428/MPDD-Young/Training/labels/personalized_train.json'
    unlabeled_personality_features, unlabeled_user_ids = load_personality_features(unlabeled_personality_file)
    
    # 加载训练数据
    train_samples = load_train_data2(args.train_json, train_personality_features)
    #print(f"未标记数据集大小: {len(train_samples)}")
    # 创建完整数据集
    dataset = MultiModalPersonalityDataset4(
        samples=train_samples,
        personality_features=train_personality_features,
        label_count=args.labelcount,
        audio_deep_path=audio_deep_path,
        audio_predesigned_path=audio_predesigned_path,
        visual_deep_path=visual_deep_path,
        visual_predesigned_path=visual_predesigned_path,
        isTest=False
    )
    
    # 设置设备
    device = torch.device(args.device)
    
    # 加载未标记数据
    unlabeled_json_path = '/public/home/hpc244712231/20250428/test/MPDD-Young/labels/Testing_files.json'
    unlabeled_samples = load_train_data2(unlabeled_json_path, unlabeled_personality_features)
    
    # 测试集路径
    test_audio_deep_path = os.path.join('/public/home/hpc244712231/20250428/test/MPDD-Young', f"{args.splitwindow_time}", 'Audio', f"{args.audio_deep_method}")
    test_audio_predesigned_path = os.path.join('/public/home/hpc244712231/20250428/test/MPDD-Young', f"{args.splitwindow_time}", 'Audio', f"{args.audio_predesigned_method}")
    test_visual_deep_path = os.path.join('/public/home/hpc244712231/20250428/test/MPDD-Young', f"{args.splitwindow_time}", 'Visual', f"{args.visual_deep_method}")
    test_visual_predesigned_path = os.path.join('/public/home/hpc244712231/20250428/test/MPDD-Young', f"{args.splitwindow_time}", 'Visual', f"{args.visual_predesigned_method}")
    
    unlabeled_dataset = MultiModalPersonalityDataset4(
        samples=unlabeled_samples,
        personality_features=unlabeled_personality_features,
        label_count=args.labelcount,
        audio_deep_path=test_audio_deep_path,
        audio_predesigned_path=test_audio_predesigned_path,
        visual_deep_path=test_visual_deep_path,
        visual_predesigned_path=test_visual_predesigned_path,
        isTest=True
    )
    
    print(f"完整标记数据集大小: 264")
    #print(f"未标记数据集大小: {len(dataset)}")
    # 样本测试，获取特征维度
    sample_idx = 0
    _, _, visual_feats, audio_feats, _ = dataset[sample_idx]
    visual_deep_feat, visual_predesigned_feat = visual_feats
    audio_deep_feat, audio_predesigned_feat = audio_feats
    
    print(f"样本视觉深度特征维度: {visual_deep_feat.shape}")
    print(f"样本视觉预设计特征维度: {visual_predesigned_feat.shape}")
    print(f"样本音频深度特征维度: {audio_deep_feat.shape}")
    print(f"样本音频预设计特征维度: {audio_predesigned_feat.shape}")
    
    # 准备用户级特征 - 修改后不再存储掩码
    if args.use_user_features:
        print("正在准备用户级特征...")
        
        # 构建用户数据字典 - 训练集
        train_user_data_dict = {}
        for i in range(len(dataset)):
            user_id, personality_feat, visual_feats, audio_feats, label = dataset[i]
            visual_deep_feat, visual_predesigned_feat = visual_feats
            audio_deep_feat, audio_predesigned_feat = audio_feats
            
            if user_id not in train_user_data_dict:
                train_user_data_dict[user_id] = {
                    'visual_deep': [], 'visual_predesigned': [],
                    'audio_deep': [], 'audio_predesigned': []
                }
            train_user_data_dict[user_id]['visual_deep'].append(visual_deep_feat)
            train_user_data_dict[user_id]['visual_predesigned'].append(visual_predesigned_feat)
            train_user_data_dict[user_id]['audio_deep'].append(audio_deep_feat)
            train_user_data_dict[user_id]['audio_predesigned'].append(audio_predesigned_feat)
            
        # 构建用户数据字典 - 测试集
        test_user_data_dict = {}
        for i in range(len(unlabeled_dataset)):
            user_id, personality_feat, visual_feats, audio_feats, _ = unlabeled_dataset[i]
            visual_deep_feat, visual_predesigned_feat = visual_feats
            audio_deep_feat, audio_predesigned_feat = audio_feats
            
            if user_id not in test_user_data_dict:
                test_user_data_dict[user_id] = {
                    'visual_deep': [], 'visual_predesigned': [],
                    'audio_deep': [], 'audio_predesigned': []
                }
            test_user_data_dict[user_id]['visual_deep'].append(visual_deep_feat)
            test_user_data_dict[user_id]['visual_predesigned'].append(visual_predesigned_feat)
            test_user_data_dict[user_id]['audio_deep'].append(audio_deep_feat)
            test_user_data_dict[user_id]['audio_predesigned'].append(audio_predesigned_feat)
    
    # 创建模型工厂函数
    def create_model(device):
        # 定义特征类型和维度
        visual_feature_types = [args.visual_deep_method, args.visual_predesigned_method]
        audio_feature_types = [args.audio_deep_method, args.audio_predesigned_method]
        
        # 创建模型
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
        
        # 执行DP-GMM个性化聚类
        if args.personalized_features_file:
            all_personality_features = {**train_personality_features, **unlabeled_personality_features}
            all_personality_tensors = []
            all_user_ids = []
            for user_id, feat in all_personality_features.items():
                all_personality_tensors.append(feat)
                all_user_ids.append(user_id)
            
            all_personality_tensors = torch.stack(all_personality_tensors)
            
            # 使用用户级多模态特征进行聚类
            model.fit_dpgmm(all_personality_tensors, all_user_ids)
            # 直接保存聚类信息到文件
            cluster_info = {
                'cluster_centers': model.cluster_centers.cpu().numpy() if hasattr(model, 'cluster_centers') else None,
                'actual_k': model.actual_k if hasattr(model, 'actual_k') else model.K,
                'cluster_labels_dict': model.cluster_labels_dict,
                'cluster_id_to_indices': model.cluster_id_to_indices,
                'scaler_mean': model.scaler_mean.cpu().numpy() if hasattr(model, 'scaler_mean') else None,
                'scaler_scale': model.scaler_scale.cpu().numpy() if hasattr(model, 'scaler_scale') else None
            }
            
            # 保存聚类信息到文件
            cluster_info_path = os.path.join(save_dir, 'cluster_info.pkl')
            with open(cluster_info_path, 'wb') as f:
                pickle.dump(cluster_info, f)
            print(f"聚类信息已保存至: {cluster_info_path}")
        else:
            print("跳过个性化聚类，使用纯多模态模式")
            model.has_personality = False
        
        return model
    
    # 设置保存目录
    save_dir = f'./kfold_checkpoints/{args.labelcount}-{args.splitwindow_time}-{args.audio_deep_method}+{args.audio_predesigned_method}-{args.visual_deep_method}+{args.visual_predesigned_method}'
    if args.use_user_features:
        save_dir += f'-user_{args.user_feature_method}'
    os.makedirs(save_dir, exist_ok=True)
    
    def fixed_length_train_with_kfold(dataset, unlabeled_dataset, model_constructor, device, 
                       num_folds=5, epochs=50, batch_size=32, lr=5e-5, lambda_reg=1e-4,
                       confidence_threshold=0.9, save_dir='./kfold_checkpoints', user_history_data=None):
        from sklearn.model_selection import KFold
        import numpy as np
        import os
        import torch
        from torch.utils.data import DataLoader, Subset
      
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        indices = np.arange(len(dataset))
      
        fold_metrics = []
        best_overall_f1 = -float('inf')
        best_overall_model_state = None
        best_overall_model_fold_dir = None
      
        for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
            print(f"\n====== Fold {fold+1}/{num_folds} ======")
      
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
      
            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=lambda batch: fixed_length_collate_fn(
                    batch,
                    FIXED_LENGTH,
                    args.visual_deep_method,
                    args.visual_predesigned_method,
                    args.audio_deep_method,
                    args.audio_predesigned_method
                )
            )
      
            val_loader = DataLoader(
                val_subset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda batch: fixed_length_collate_fn(
                    batch,
                    FIXED_LENGTH,
                    args.visual_deep_method,
                    args.visual_predesigned_method,
                    args.audio_deep_method,
                    args.audio_predesigned_method
                )
            )
      
            model = model_constructor(device)
      
            fold_save_dir = os.path.join(save_dir, f'fold_{fold+1}')
            os.makedirs(fold_save_dir, exist_ok=True)
      
            print("\n====== 初始训练阶段 ======")
            best_metric = fixed_length_train_loop(
                full_train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                device=device,
                epochs=epochs,
                lr=lr,
                lambda_reg=lambda_reg,
                save_dir=fold_save_dir,
                monitor_metric='f1_weighted',
                label_count=getattr(dataset, 'label_count', 3),
                user_history_data=user_history_data
            )
      
            metrics = {
                'fold': fold + 1,
                'accuracy': best_metric.get('accuracy', 0),
                'f1_weighted': best_metric.get('f1_weighted', 0),
                'f1_macro': best_metric.get('f1_macro', 0)
            }
            fold_metrics.append(metrics)
      
            if metrics['f1_weighted'] > best_overall_f1:
                best_overall_f1 = metrics['f1_weighted']
                best_overall_model_state = model.state_dict()
                best_overall_model_fold_dir = fold_save_dir
      
            # 可选：加载最佳模型进行伪标签等后续操作
            best_model_path = find_best_model(fold_save_dir)
            if best_model_path:
                print(f"加载初始阶段最佳模型: {best_model_path}")
                model.load_state_dict(torch.load(best_model_path, map_location=device))
      
        avg_acc = np.mean([m['accuracy'] for m in fold_metrics])
        avg_f1w = np.mean([m['f1_weighted'] for m in fold_metrics])
        avg_f1m = np.mean([m['f1_macro'] for m in fold_metrics])
      
        print("\n====== 交叉验证总结 ======")
        print(f"平均性能: Acc={avg_acc:.4f}, F1_weighted={avg_f1w:.4f}, F1_macro={avg_f1m:.4f}")
      
        for m in fold_metrics:
            print(f"Fold {m['fold']}: Acc={m['accuracy']:.4f}, F1_weighted={m['f1_weighted']:.4f}, F1_macro={m['f1_macro']:.4f}")
      
        if best_overall_model_state is not None:
            best_model_path = os.path.join(save_dir, f'best_overall_model_f1_{best_overall_f1:.4f}.pth')
            torch.save(best_overall_model_state, best_model_path)
            print(f"保存全局最佳模型到: {best_model_path}")
      
        return avg_acc, avg_f1w, avg_f1m, best_model_path
    def fixed_length_train_single_split(dataset, unlabeled_dataset, model_constructor, device, 
                                       epochs=50, batch_size=32, lr=5e-5, lambda_reg=1e-4, save_dir='./checkpoints', 
                                       user_history_data=None, val_ratio=0.1):
        import os
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, Subset
        from sklearn.model_selection import train_test_split
    
        indices = np.arange(len(dataset))
    
        # 如果dataset有标签，先获得标签列表用于分层采样
        if hasattr(dataset, 'labels'):
            labels = dataset.labels  # 假设dataset.labels是标签列表，或换成取标签的其它方法
        elif hasattr(dataset, 'label_count') and dataset.label_count > 1:
            # 如果没有labels属性，手动提取标签
            labels = []
            for i in indices:
                _, label = dataset[i][0], dataset[i][-1]  # 根据数据结构修改
                labels.append(label)
        else:
            labels = None
    
        # 分割训练和验证集
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            random_state=42,
            stratify=labels if labels is not None else None
        )
    
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
    
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: fixed_length_collate_fn(
                batch,
                FIXED_LENGTH,
                args.visual_deep_method,
                args.visual_predesigned_method,
                args.audio_deep_method,
                args.audio_predesigned_method
            )
        )
    
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: fixed_length_collate_fn(
                batch,
                FIXED_LENGTH,
                args.visual_deep_method,
                args.visual_predesigned_method,
                args.audio_deep_method,
                args.audio_predesigned_method
            )
        )
    
        model = model_constructor(device)
        os.makedirs(save_dir, exist_ok=True)
    
        print("\n====== 开始训练 ======")
        best_metric = fixed_length_train_loop(
            full_train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            device=device,
            epochs=epochs,
            lr=lr,
            lambda_reg=lambda_reg,
            save_dir=save_dir,
            monitor_metric='f1_weighted',
            label_count=getattr(dataset, 'label_count', 3),
            user_history_data=user_history_data
        )
    
        best_model_path = find_best_model(save_dir)    
        print("\n====== 训练结束 ======")
        metrics = {
            'accuracy': best_metric.get('accuracy', 0),
            'f1_weighted': best_metric.get('f1_weighted', 0),
            'f1_macro': best_metric.get('f1_macro', 0)
        }
    
        print(f"性能指标: Acc={metrics['accuracy']:.4f}, F1_weighted={metrics['f1_weighted']:.4f}, F1_macro={metrics['f1_macro']:.4f}")
    
        return metrics['accuracy'], metrics['f1_weighted'], metrics['f1_macro'], best_model_path

    
    # 定义带伪标签的K折交叉验证函数
    def fixed_length_train_with_kfold_and_pseudo_labeling(dataset, unlabeled_dataset, model_constructor, device, 
                                             num_folds=5, initial_epochs=50, pseudo_epochs=5, batch_size=32, 
                                             lr=5e-5, lambda_reg=1e-4, confidence_threshold=0.9, 
                                             max_iterations=3, save_dir='./kfold_checkpoints',user_history_data=None):
        from sklearn.model_selection import KFold
        import numpy as np
        from collections import Counter
        
        # 准备K折交叉验证
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        indices = np.arange(len(dataset))
        
        # 存储每折模型对未标记数据的预测
        all_fold_predictions = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
            print(f"\n====== Fold {fold+1}/{num_folds} ======")
            
            # 创建训练和验证子集
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            print(f"训练集大小: {len(train_subset)}, 验证集大小: {len(val_subset)}")
            
            # 初始化伪标签数据集为空
            pseudo_labeled_samples = []
            
            # 创建数据加载器，使用fixed_length_collate_fn
            train_loader = DataLoader(
                train_subset, 
                batch_size=batch_size, 
                shuffle=True,
                collate_fn=lambda batch: fixed_length_collate_fn(
                    batch, 
                    FIXED_LENGTH,
                    args.visual_deep_method,
                    args.visual_predesigned_method,
                    args.audio_deep_method,
                    args.audio_predesigned_method
                )
            )
            
            val_loader = DataLoader(
                val_subset, 
                batch_size=batch_size, 
                shuffle=False,
                collate_fn=lambda batch: fixed_length_collate_fn(
                    batch, 
                    FIXED_LENGTH,
                    args.visual_deep_method,
                    args.visual_predesigned_method,
                    args.audio_deep_method,
                    args.audio_predesigned_method
                )
            )
            
            unlabeled_loader = DataLoader(
                unlabeled_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                collate_fn=lambda batch: fixed_length_collate_fn(
                    batch, 
                    FIXED_LENGTH,
                    args.visual_deep_method,
                    args.visual_predesigned_method,
                    args.audio_deep_method,
                    args.audio_predesigned_method
                )
            )
            
            # 创建该折的模型
            model = model_constructor(device)
            
            # 为该折创建保存目录
            fold_save_dir = os.path.join(save_dir, f'fold_{fold+1}')
            os.makedirs(fold_save_dir, exist_ok=True)
            
            # 初始训练阶段
            print("\n====== 初始训练阶段 ======")
            best_metric = fixed_length_train_loop(
                full_train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                device=device,
                epochs=initial_epochs,
                lr=lr,
                lambda_reg=lambda_reg,
                save_dir=fold_save_dir,
                monitor_metric='f1_weighted',
                label_count=dataset.label_count if hasattr(dataset, 'label_count') else 3,
                user_history_data=user_history_data
            )
            
            # 加载初始阶段的最佳模型
            best_model_path = find_best_model(fold_save_dir)
            if best_model_path:
                print(f"加载初始阶段最佳模型: {best_model_path}")
                model.load_state_dict(torch.load(best_model_path, map_location=device))
            
            # 伪标签迭代阶段
            for iteration in range(max_iterations):
                print(f"\n====== 伪标签迭代 {iteration+1}/{max_iterations} ======")
                
                # 生成伪标签
                pseudo_labels_file = os.path.join(fold_save_dir, f'pseudo_labels_iter_{iteration+1}.json')
                pseudo_result = fixed_length_generate_pseudo_labels(
                    model=model,
                    unlabeled_loader=unlabeled_loader,
                    device=device,
                    output_file=pseudo_labels_file,
                    confidence_threshold=0.85
                )
                
                # 获取高置信度样本
                high_confidence_samples = pseudo_result['high_confidence']
                
                if not high_confidence_samples:
                    print("没有高置信度样本，跳过此迭代")
                    continue
                pseudo_labels = [sample['bin_category'] for sample in high_confidence_samples]
                label_counts = Counter(pseudo_labels)
                print(f"伪标签类别分布: {label_counts}")
                print(f"添加 {len(high_confidence_samples)} 个高置信度样本进行训练")
                '''
                # 创建包含原始数据和伪标签数据的新数据集
                combined_samples = train_samples.copy()
                for sample in high_confidence_samples:
                    sample_copy = sample.copy()
                    combined_samples.append(sample_copy)
                '''
                # 按类别收集高置信度伪标签样本
                samples_per_class = defaultdict(list)
                for sample in high_confidence_samples:
                    samples_per_class[sample['bin_category']].append(sample)
                
                # 找到最小类别数量
                min_class_count = min(len(v) for v in samples_per_class.values())
                
                print(f"最低类别数量（采样数量）: {min_class_count}")
                balanced_pseudo_samples = []

                for label, samples in samples_per_class.items():
                    if len(samples) > min_class_count:
                        selected_samples = random.sample(samples, min_class_count)
                    else:
                        selected_samples = samples  # 等于或少于min_class_count 就全部用
                    balanced_pseudo_samples.extend(selected_samples)
                
                print(f"均衡后伪标签总数: {len(balanced_pseudo_samples)}")
                # 复制原始训练集样本列表
                combined_samples = train_samples.copy()
                
                # 添加均衡采样获得的伪标签样本
                for sample in balanced_pseudo_samples:
                    sample_copy = sample.copy()
                    combined_samples.append(sample_copy)
                combined_dataset = MultiModalPersonalityDataset4(
                    samples=combined_samples,
                    personality_features={**train_personality_features, **unlabeled_personality_features},
                    label_count=args.labelcount,
                    audio_deep_path=audio_deep_path,
                    audio_predesigned_path=audio_predesigned_path,
                    visual_deep_path=visual_deep_path,
                    visual_predesigned_path=visual_predesigned_path,
                    isTest=False
                )
                # 构建用户数据字典 - 训练集
                train_user_data_dict = {}
                for i in range(len(combined_dataset)):
                    user_id, personality_feat, visual_feats, audio_feats, label = combined_dataset[i]
                    visual_deep_feat, visual_predesigned_feat = visual_feats
                    audio_deep_feat, audio_predesigned_feat = audio_feats
                    
                    if user_id not in train_user_data_dict:
                        train_user_data_dict[user_id] = {
                            'visual_deep': [], 'visual_predesigned': [],
                            'audio_deep': [], 'audio_predesigned': []
                        }
                    train_user_data_dict[user_id]['visual_deep'].append(visual_deep_feat)
                    train_user_data_dict[user_id]['visual_predesigned'].append(visual_predesigned_feat)
                    train_user_data_dict[user_id]['audio_deep'].append(audio_deep_feat)
                    train_user_data_dict[user_id]['audio_predesigned'].append(audio_predesigned_feat)
                # 创建训练子集（包括伪标签数据）
                combined_train_idx = list(train_idx)
                # 添加伪标签样本的索引
                combined_train_idx.extend(range(len(dataset), len(dataset) + len(balanced_pseudo_samples)))
                
                combined_train_subset = Subset(combined_dataset, combined_train_idx)
                
                # 创建新的训练加载器
                combined_train_loader = DataLoader(
                    combined_train_subset,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=lambda batch:fixed_length_collate_fn(
                        batch, 
                        FIXED_LENGTH,
                        args.visual_deep_method,
                        args.visual_predesigned_method,
                        args.audio_deep_method,
                        args.audio_predesigned_method
                    )
                )
                # 创建迭代目录
                iter_save_dir = os.path.join(fold_save_dir, f'iter_{iteration+1}')
                os.makedirs(iter_save_dir, exist_ok=True)
                # 在扩展数据集上继续训练
                iter_best_metric = fixed_length_train_loop(
                    full_train_loader=combined_train_loader,
                    val_loader=val_loader,
                    model=model,
                    device=device,
                    epochs=pseudo_epochs,
                    lr=lr / 2,  # 降低学习率进行微调
                    lambda_reg=lambda_reg,
                    save_dir=iter_save_dir,
                    monitor_metric='f1_weighted',
                    label_count=dataset.label_count if hasattr(dataset, 'label_count') else 3,
                    user_history_data=train_user_data_dict
                )
                
                
                # 加载该迭代的最佳模型
                iter_best_model_path = find_best_model(os.path.join(fold_save_dir, f'iter_{iteration+1}'))
                if iter_best_model_path:
                    print(f"加载迭代 {iteration+1} 的最佳模型: {iter_best_model_path}")
                    model.load_state_dict(torch.load(iter_best_model_path, map_location=device))
                if len(high_confidence_samples) == len(unlabeled_dataset):
                    print("所有未标记样本都已达到高置信度阈值，提前结束伪标签迭代")
                    break
        return None, None, None, ensemble_predictions

    args.do_pseudo_labeling=False
    # 执行伪标签K折训练或常规K折训练
    if args.do_pseudo_labeling:
        print("\n开始K折交叉验证与伪标签训练...")
        avg_acc, avg_f1w, avg_f1m, ensemble_predictions = fixed_length_train_with_kfold_and_pseudo_labeling(
            dataset=dataset,
            unlabeled_dataset=unlabeled_dataset,
            model_constructor=create_model,
            device=device,
            num_folds=args.num_folds,
            initial_epochs=args.initial_epochs,
            pseudo_epochs=args.pseudo_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_reg=args.lambda_reg,
            confidence_threshold=args.confidence_threshold,
            max_iterations=args.max_iterations,
            save_dir=save_dir,
            user_history_data=train_user_data_dict
        )
        
        # 保存最终预测结果（仅标签）
        if ensemble_predictions:
            final_predictions = {user_id: info['bin_category'] for user_id, info in ensemble_predictions.items()}
            with open(os.path.join(save_dir, 'final_predictions.json'), 'w') as f:
                json.dump(final_predictions, f, indent=2)
                
            # 保存带置信度的全部信息
            with open(os.path.join(save_dir, 'final_predictions_with_confidence.json'), 'w') as f:
                json.dump(ensemble_predictions, f, indent=2)
    else:
        '''
        # 执行原来的K折交叉验证训练
        print("\n开始K折交叉验证训练...")
        avg_acc, avg_f1w, avg_f1m, best_model = fixed_length_train_with_kfold(
            dataset=dataset,
            unlabeled_dataset=unlabeled_dataset,
            model_constructor=create_model,
            device=device,
            num_folds=args.num_folds,
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_reg=args.lambda_reg,
            confidence_threshold=args.confidence_threshold,
            save_dir=save_dir,
            user_history_data=train_user_data_dict
        )
        '''
        
        acc, f1w, f1m, best_model_path = fixed_length_train_single_split(
            dataset=dataset,
            unlabeled_dataset=unlabeled_dataset,
            model_constructor=create_model,
            device=device,
            epochs=350,
            batch_size=32,
            lr=5e-5,
            lambda_reg=1e-4,
            save_dir=save_dir,
            user_history_data=train_user_data_dict,  # 如果有用户历史数据字典就传，否则传None
            val_ratio=0.2
        )
