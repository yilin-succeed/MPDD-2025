import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json
def load_personality_features(json_path):
    """
    从JSON文件加载人格特征
    
    Args:
        json_path: 包含人格特征的JSON文件路径（用户信息JSON）
        
    Returns:
        personality_features: 人格特征字典 {user_id: tensor}
        user_ids: 用户ID列表
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    personality_features = {}
    user_ids = []
    
    for user_id, user_data in data.items():
        
        # 使用big5_traits
        if 'big5_traits' in user_data:
            features = [
                float(user_data['big5_traits'].get('Extraversion', 0)),
                float(user_data['big5_traits'].get('Agreeableness', 0)),
                float(user_data['big5_traits'].get('Conscientiousness', 0)),
                float(user_data['big5_traits'].get('Neuroticism', 0)),
                float(user_data['big5_traits'].get('Openness', 0))
            ]
        else:
            # 如果没有人格特征，使用零向量
            features = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        personality_features[user_id] = torch.tensor(features, dtype=torch.float32)
        user_ids.append(user_id)
    return personality_features, user_ids

def load_personality_features2(json_path):
    """
    从JSON文件加载人格特征，包括年龄、性别、籍贯和big5特征
    
    Args:
        json_path: 包含人格特征的JSON文件路径（用户信息JSON）
        
    Returns:
        personality_features: 人格特征字典 {user_id: tensor}
        user_ids: 用户ID列表
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    personality_features = {}
    user_ids = []
    
    for user_id, user_data in data.items():
        features = []
        
        # 添加年龄特征 (已编码为0,1,2)
        age = float(user_data.get('age', 0))
        features.append(age)
        
        # 添加性别特征 (已编码为0,1)
        gender = float(user_data.get('gender', 0))
        features.append(gender)
        
        # 添加籍贯特征 (已编码为0-7)
        native_place = float(user_data.get('native_place', 0))
        features.append(native_place)
        
        # 添加big5特征
        if 'big5_traits' in user_data:
            features.extend([
                float(user_data['big5_traits'].get('Extraversion', 0)),
                float(user_data['big5_traits'].get('Agreeableness', 0)),
                float(user_data['big5_traits'].get('Conscientiousness', 0)),
                float(user_data['big5_traits'].get('Neuroticism', 0)),
                float(user_data['big5_traits'].get('Openness', 0))
            ])
        else:
            # 如果没有人格特征，使用零向量
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        personality_features[user_id] = torch.tensor(features, dtype=torch.float32)
        user_ids.append(user_id)
    
    return personality_features, user_ids

def load_train_data(json_path, personality_features):
    """
    从JSON文件加载训练数据
    
    Args:
        json_path: 包含训练样本信息的JSON文件路径
        personality_features: 人格特征字典 {user_id: tensor}
        
    Returns:
        samples: 样本列表，每个样本包含特征路径和标签
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    if personality_features:
        samples = []
        for item in data:
            # 从特征路径中提取用户ID
            audio_path = item.get('audio_feature_path', '')
            user_id_with_zeros = audio_path.split('_')[0] if '_' in audio_path else ''
            
            # 将带前导零的用户ID转换为不带前导零的格式
            try:
                user_id = str(int(user_id_with_zeros))
            except ValueError:
                # 如果无法转换为整数，保持原样
                user_id = user_id_with_zeros
            
            # 只处理有人格特征的用户
            if user_id in personality_features:
                samples.append({
                    'user_id': user_id,  # 存储不带前导零的用户ID
                    'audio_feature_path': item.get('audio_feature_path', ''),
                    'video_feature_path': item.get('video_feature_path', ''),
                    'bin_category': item.get('bin_category', 0),
                    'tri_category': item.get('tri_category', 0)
                })
    else:
        samples = []
        for item in data:
            # 从特征路径中提取用户ID
            audio_path = item.get('audio_feature_path', '')
            user_id_with_zeros = audio_path.split('_')[0] if '_' in audio_path else ''
            samples.append({
                'user_id': user_id_with_zeros,  # 存储不带前导零的用户ID
                'audio_feature_path': item.get('audio_feature_path', ''),
                'video_feature_path': item.get('video_feature_path', ''),
                'bin_category': item.get('bin_category', 0),
                'tri_category': item.get('tri_category', 0)
            })
    return samples

def load_train_data2(json_path, personality_features):
    with open(json_path, 'r') as f:
        data = json.load(f)

    samples = []
    for item in data:
        audio_path = item.get('audio_feature_path', '')
        user_id_with_suffix = audio_path.split('.npy')[0]  # 得到 '6373_1'
        user_id = user_id_with_suffix.split('_')[0]  # 得到 '6373'
        try:
            user_id = str(int(user_id))  # 去掉前导零
        except ValueError:
            pass

        if user_id in personality_features:
            samples.append({
                'user_id': user_id,  # 使用 '6373'
                'audio_feature_path': audio_path,
                'video_feature_path': item.get('video_feature_path', ''),
                'bin_category': item.get('bin_category', 0),
                'tri_category': item.get('tri_category', 0)
            })
    return samples


# 添加一个新的函数，用于加载没有性格特征的训练数据
def load_train_data_without_personality(json_path):
    """
    从JSON文件加载训练数据（不需要性格特征）
    
    Args:
        json_path: 包含训练样本信息的JSON文件路径
        
    Returns:
        samples: 样本列表，每个样本包含特征路径和标签
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    samples = []
    for item in data:
        # 直接添加样本，不需要检查人格特征
        samples.append({
            'user_id': 'dummy',  # 使用虚拟用户ID
            'audio_feature_path': item.get('audio_feature_path', ''),
            'video_feature_path': item.get('video_feature_path', ''),
            'bin_category': item.get('bin_category', 0),
            'tri_category': item.get('tri_category', 0)
        })
    
    return samples

# 修改现有的数据集类，增加对没有性格特征的处理支持
class MultiModalPersonalityDataset(torch.utils.data.Dataset):
    def __init__(self, samples, personality_features=None, label_count=2, 
                 audio_path=None,audio_path2=None, video_path=None, isTest=False):
        """
        多模态人格数据集
        
        Args:
            samples: 样本列表，每个样本包含特征路径和标签
            personality_features: 人格特征字典 {user_id: tensor}，可为None
            label_count: 标签类别数量
            audio_path: 音频特征根路径
            video_path: 视频特征根路径
            isTest: 是否为测试集
        """
        self.samples = samples
        self.personality_features = personality_features
        self.has_personality = personality_features is not None
        self.label_count = label_count
        self.audio_path = audio_path
        self.video_path = video_path
        self.audio_path2=audio_path2
        self.isTest = isTest
        
        # 如果没有性格特征，创建一个默认的全零特征
        if not self.has_personality:
            self.default_personality = torch.zeros(5, dtype=torch.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 获取人格特征
        if self.has_personality:
            user_id = sample['user_id']
            personality_feat = self.personality_features[user_id]
        else:
            # 使用默认的全零特征
            personality_feat = self.default_personality
        
        # 加载音频特征
        audio_feat = None
        if self.audio_path:
            audio_file = os.path.join(self.audio_path, sample['audio_feature_path'])
            if os.path.exists(audio_file):
                audio_feat = np.load(audio_file)
            else:
                print(audio_file)
                # 如果文件不存在，使用零向量
                audio_feat = np.zeros((1, 512))  # 假设维度为512

        # 加载音频特征
        audio_feat2 = None
        if self.audio_path2:
            audio_file = os.path.join(self.audio_path2, sample['audio_feature_path'])
            if os.path.exists(audio_file):
                audio_feat2 = np.load(audio_file)
            else:
                print(audio_file)
                # 如果文件不存在，使用零向量
                audio_feat2 = np.zeros((1, 512))  # 假设维度为512
        
        # 加载视觉特征
        visual_feat = None
        if self.video_path:
            visual_file = os.path.join(self.video_path, sample['video_feature_path'])
            if os.path.exists(visual_file):
                visual_feat = np.load(visual_file)
            else:
                print(visual_file)
                # 如果文件不存在，使用零向量
                visual_feat = np.zeros((1, 512))  # 假设维度为512
        
        # 判断视觉和音频特征的时间步长，按照三者中较短的进行裁剪
        if audio_feat is not None and audio_feat2 is not None and visual_feat is not None:
            audio_time_steps = audio_feat.shape[0]
            audio_time_steps2 = audio_feat2.shape[0]
            visual_time_steps = visual_feat.shape[0]
            
            # 找出三者中最短的时间步长
            min_time_steps = min(audio_time_steps, audio_time_steps2, visual_time_steps)
            
            # 按照最短的时间步长裁剪所有特征
            audio_feat = audio_feat[:min_time_steps]
            audio_feat2 = audio_feat2[:min_time_steps]
            visual_feat = visual_feat[:min_time_steps]        

        
        # 获取标签
        if not self.isTest:
            if self.label_count == 2:
                label = sample.get('bin_category', 0)
            elif self.label_count == 3:
                label = sample.get('tri_category', 0)
            else:
                label = sample.get('bin_category', 0)
        else:
            label = -1  # 测试集没有标签
        
        # 转换为张量
        personality_feat = personality_feat.float()
        visual_feat = torch.tensor(visual_feat, dtype=torch.float32)
        audio_feat = torch.tensor(audio_feat, dtype=torch.float32)
        audio_feat2 = torch.tensor(audio_feat2, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return personality_feat, visual_feat, audio_feat, audio_feat2,label

# 修改现有的数据集类，移除audio_feat2相关内容
class MultiModalPersonalityDataset2(torch.utils.data.Dataset):
    def __init__(self, samples, personality_features=None, label_count=2, 
                 audio_path=None, video_path=None, isTest=False):
        """
        多模态人格数据集
        
        Args:
            samples: 样本列表，每个样本包含特征路径和标签
            personality_features: 人格特征字典 {user_id: tensor}，可为None
            label_count: 标签类别数量
            audio_path: 音频特征根路径
            video_path: 视频特征根路径
            isTest: 是否为测试集
        """
        self.samples = samples
        self.personality_features = personality_features
        self.has_personality = personality_features is not None
        self.label_count = label_count
        self.audio_path = audio_path
        self.video_path = video_path
        self.isTest = isTest
        
        # 如果没有性格特征，创建一个默认的全零特征
        if not self.has_personality:
            self.default_personality = torch.zeros(5, dtype=torch.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
         sample = self.samples[idx]
         user_id = sample['user_id']
         
         # 构建完整ID（从音频路径获取）
         audio_path = sample['audio_feature_path']
         complete_id = audio_path.replace('.npy', '')
         
         # 获取人格特征
         if self.has_personality:
             personality_feat = self.personality_features[user_id]
         else:
             # 使用默认的全零特征
             personality_feat = self.default_personality
         
         # 加载音频特征
         audio_feat = None
         if self.audio_path:
             audio_file = os.path.join(self.audio_path, audio_path)
             if os.path.exists(audio_file):
                 audio_feat = np.load(audio_file)
             else:
                 print(audio_file)
                 # 如果文件不存在，使用零向量
                 audio_feat = np.zeros((1, 512))  # 假设维度为512
         
         # 加载视觉特征
         visual_feat = None
         if self.video_path:
             visual_file = os.path.join(self.video_path, sample['video_feature_path'])
             if os.path.exists(visual_file):
                 visual_feat = np.load(visual_file)
             else:
                 print(visual_file)
                 # 如果文件不存在，使用零向量
                 visual_feat = np.zeros((1, 512))  # 假设维度为512
         
         # 判断视觉和音频特征的时间步长，按照两者中较短的进行裁剪
         if audio_feat is not None and visual_feat is not None:
             audio_time_steps = audio_feat.shape[0]
             visual_time_steps = visual_feat.shape[0]
             
             # 找出两者中最短的时间步长
             min_time_steps = min(audio_time_steps, visual_time_steps)
             
             # 按照最短的时间步长裁剪所有特征
             audio_feat = audio_feat[:min_time_steps]
             visual_feat = visual_feat[:min_time_steps]        
         
         # 获取标签
         if not self.isTest:
             if self.label_count == 2:
                 label = sample.get('bin_category', 0)
             elif self.label_count == 3:
                 label = sample.get('tri_category', 0)
             else:
                 label = sample.get('bin_category', -1)
         else:
             label = -1  # 测试集没有标签
         
         # 转换为张量
         personality_feat = personality_feat.float()
         visual_feat = torch.tensor(visual_feat, dtype=torch.float32)
         audio_feat = torch.tensor(audio_feat, dtype=torch.float32)
         label = torch.tensor(label, dtype=torch.long)
         
         return complete_id, personality_feat, visual_feat, audio_feat, label
        
class MultiModalPersonalityDataset3(torch.utils.data.Dataset):
    def __init__(self, samples, personality_features=None, label_count=2, 
                 audio_path=None, video_path=None, test_audio_path=None, test_video_path=None, 
                 isTest=False):
        """
        多模态人格数据集，支持训练和测试数据的混合，动态选择特征路径
        
        Args:
            samples: 样本列表，每个样本包含特征路径和标签
            personality_features: 人格特征字典 {user_id: tensor}，允许为 None
            label_count: 标签类别数量
            audio_path: 主要音频特征根路径（优先尝试，训练集）
            video_path: 主要视频特征根路径（优先尝试，训练集）
            test_audio_path: 备用音频特征根路径（测试集或未标记数据）
            test_video_path: 备用视频特征根路径（测试集或未标记数据）
            is_test: 是否为测试集
        """
        self.samples = samples
        self.personality_features = personality_features
        self.has_personality = personality_features is not None
        self.label_count = label_count
        self.audio_path = audio_path
        self.video_path = video_path
        self.test_audio_path = test_audio_path
        self.test_video_path = test_video_path
        self.is_test = isTest
        
        # 始终初始化默认人格特征（全零向量）
        self.default_personality = torch.zeros(5, dtype=torch.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        user_id = sample['user_id']
        
        # 构建完整 ID（从音频路径获取）
        audio_path = sample['audio_feature_path']
        complete_id = audio_path.replace('.npy', '')
        
        # 获取人格特征
        if self.has_personality:
            personality_feat = self.personality_features.get(user_id, self.default_personality)
        else:
            personality_feat = self.default_personality
        
        # 加载音频特征：先尝试 audio_path，若文件不存在则尝试 test_audio_path
        audio_feat = None
        audio_file = None
        
        if self.audio_path:
            audio_file = os.path.join(self.audio_path, sample['audio_feature_path'])
            if os.path.exists(audio_file):
                audio_feat = np.load(audio_file)
        
        if audio_feat is None and self.test_audio_path:
            audio_file = os.path.join(self.test_audio_path, sample['audio_feature_path'])
            if os.path.exists(audio_file):
                audio_feat = np.load(audio_file)
        
        if audio_feat is None:
            print(f"音频文件不存在: {audio_file}")
            audio_feat = np.zeros((1, 512))  # 假设维度为 512
        
        # 加载视觉特征：先尝试 video_path，若文件不存在则尝试 test_video_path
        visual_feat = None
        video_file = None
        
        if self.video_path:
            video_file = os.path.join(self.video_path, sample['video_feature_path'])
            if os.path.exists(video_file):
                visual_feat = np.load(video_file)
        
        if visual_feat is None and self.test_video_path:
            video_file = os.path.join(self.test_video_path, sample['video_feature_path'])
            if os.path.exists(video_file):
                visual_feat = np.load(video_file)
        
        if visual_feat is None:
            print(f"视频文件不存在: {video_file}")
            visual_feat = np.zeros((1, 512))  # 假设维度为 512
        
        # 判断视觉和音频特征的时间步长，按照两者中较短的进行裁剪
        if audio_feat is not None and visual_feat is not None:
            audio_time_steps = audio_feat.shape[0]
            visual_time_steps = visual_feat.shape[0]
            
            # 找出两者中最短的时间步长
            min_time_steps = min(audio_time_steps, visual_time_steps)
            
            # 按照最短的时间步长裁剪所有特征
            audio_feat = audio_feat[:min_time_steps]
            visual_feat = visual_feat[:min_time_steps]        
        
        # 获取标签
        if not self.is_test:
            if self.label_count == 2:
                label = sample.get('bin_category', -1)
            elif self.label_count == 3:
                label = sample.get('tri_category', -1)
            else:
                label = sample.get('label', -1)  # 支持伪标签的 'label' 字段
        else:
            label = -1  # 测试集没有标签
        
        # 转换为张量
        personality_feat = personality_feat.float()
        visual_feat = torch.tensor(visual_feat, dtype=torch.float32)
        audio_feat = torch.tensor(audio_feat, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return complete_id, personality_feat, visual_feat, audio_feat, label

class MultiModalPersonalityDataset4(torch.utils.data.Dataset):
    def __init__(self, samples, personality_features=None, label_count=2, 
                 audio_deep_path=None, audio_predesigned_path=None,
                 visual_deep_path=None, visual_predesigned_path=None, 
                 test_audio_deep_path=None, test_audio_predesigned_path=None,
                 test_visual_deep_path=None, test_visual_predesigned_path=None,
                 isTest=False):
        """
        多模态人格数据集，支持多种特征类型、训练和测试数据混合
        
        Args:
            samples: 样本列表，每个样本包含特征路径和标签
            personality_features: 人格特征字典 {user_id: tensor}，允许为 None
            label_count: 标签类别数量
            audio_deep_path: 深度音频特征主路径（训练集）
            audio_predesigned_path: 预设计音频特征主路径（训练集）
            visual_deep_path: 深度视觉特征主路径（训练集）
            visual_predesigned_path: 预设计视觉特征主路径（训练集）
            test_audio_deep_path: 深度音频特征备用路径（测试集）
            test_audio_predesigned_path: 预设计音频特征备用路径（测试集）
            test_visual_deep_path: 深度视觉特征备用路径（测试集）
            test_visual_predesigned_path: 预设计视觉特征备用路径（测试集）
            isTest: 是否为测试集
        """
        self.samples = samples
        self.personality_features = personality_features
        self.has_personality = personality_features is not None
        self.label_count = label_count
        
        # 保存所有特征路径
        self.audio_deep_path = audio_deep_path
        self.audio_predesigned_path = audio_predesigned_path
        self.visual_deep_path = visual_deep_path
        self.visual_predesigned_path = visual_predesigned_path
        
        self.test_audio_deep_path = test_audio_deep_path
        self.test_audio_predesigned_path = test_audio_predesigned_path
        self.test_visual_deep_path = test_visual_deep_path
        self.test_visual_predesigned_path = test_visual_predesigned_path
        
        self.is_test = isTest
        
        # 默认人格特征（全零向量）
        self.default_personality = torch.zeros(5, dtype=torch.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def _load_feature(self, primary_path, backup_path, feature_filename):
        """辅助函数：尝试从主路径或备用路径加载特征，兼容编号补零格式"""
        feature = None

        # 解析文件名，分两部分 xxx_num.npy
        base_name = feature_filename.split('.npy')[0]
        parts = base_name.split('_')
        if len(parts) != 2:
            # 直接用原始文件名查找
            candidate_names = [feature_filename]
        else:
            prefix, num_str = parts
            # 补零格式，3位数字，不足补0
            num_pad = num_str.zfill(3)
            filename_pad = f"{prefix}_{num_pad}.npy"

            # 先尝试补零文件名，再尝试原始文件名
            candidate_names = [filename_pad, feature_filename]

        # 查找主路径
        if primary_path:
            for candidate in candidate_names:
                file_path = os.path.join(primary_path, candidate)
                if os.path.exists(file_path):
                    feature = np.load(file_path)
                    return feature

        # 查找备用路径
        if backup_path:
            for candidate in candidate_names:
                file_path = os.path.join(backup_path, candidate)
                if os.path.exists(file_path):
                    feature = np.load(file_path)
                    return feature

        print(f"特征文件不存在: {primary_path} （尝试过的候选名: {candidate_names}）")
        return np.zeros((1, 512))  # 默认特征维度
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        user_id = sample['user_id']
        
        # 获取人格特征
        if self.has_personality:
            personality_feat = self.personality_features.get(user_id, self.default_personality)
        else:
            personality_feat = self.default_personality
        
        # 获取完整ID
        audio_path = sample['audio_feature_path']
        complete_id = audio_path.replace('.npy', '')
        
        # 加载所有特征
        audio_deep_feat = self._load_feature(
            self.audio_deep_path, self.test_audio_deep_path, sample['audio_feature_path'])
        
        audio_predesigned_feat = self._load_feature(
            self.audio_predesigned_path, self.test_audio_predesigned_path, sample['audio_feature_path'])
        
        visual_deep_feat = self._load_feature(
            self.visual_deep_path, self.test_visual_deep_path, sample['video_feature_path'])
        
        visual_predesigned_feat = self._load_feature(
            self.visual_predesigned_path, self.test_visual_predesigned_path, sample['video_feature_path'])
        
        # 找出所有特征中的最小时间步
        time_steps = [
            audio_deep_feat.shape[0],
            audio_predesigned_feat.shape[0],
            visual_deep_feat.shape[0],
            visual_predesigned_feat.shape[0]
        ]
        min_time_steps = min(time_steps)
        
        # 裁剪所有特征到相同长度
        audio_deep_feat = audio_deep_feat[:min_time_steps]
        audio_predesigned_feat = audio_predesigned_feat[:min_time_steps]
        visual_deep_feat = visual_deep_feat[:min_time_steps]
        visual_predesigned_feat = visual_predesigned_feat[:min_time_steps]
        
        # 获取标签
        if not self.is_test:
            if self.label_count == 2:
                label = sample.get('bin_category', -1)
            elif self.label_count == 3:
                label = sample.get('tri_category', -1)
            else:
                label = sample.get('label', -1)  # 支持伪标签的 'label' 字段
        else:
            label = -1  # 测试集没有标签
        
        # 转换为张量
        personality_feat = torch.tensor(personality_feat, dtype=torch.float32) if not isinstance(personality_feat, torch.Tensor) else personality_feat
        visual_deep_feat = torch.tensor(visual_deep_feat, dtype=torch.float32)
        visual_predesigned_feat = torch.tensor(visual_predesigned_feat, dtype=torch.float32)
        audio_deep_feat = torch.tensor(audio_deep_feat, dtype=torch.float32)
        audio_predesigned_feat = torch.tensor(audio_predesigned_feat, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return complete_id, personality_feat, (visual_deep_feat, visual_predesigned_feat), (audio_deep_feat, audio_predesigned_feat), label
