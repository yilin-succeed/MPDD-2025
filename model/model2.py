import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import AgglomerativeClustering

class SimpleP2K(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=64):
        super(SimpleP2K, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadCrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
    
    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        return attn_output

class DynamicModalFusion(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super(DynamicModalFusion, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.gate = nn.Sequential(
            nn.Linear(embed_dim*2, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, v_feat, a_feat):
        concat_feat = torch.cat([v_feat, a_feat], dim=1)
        weights = self.gate(concat_feat)
        weighted_v = v_feat * weights[:, 0].unsqueeze(-1)
        weighted_a = a_feat * weights[:, 1].unsqueeze(-1)
        fused = weighted_v + weighted_a + self.fc(concat_feat)
        return fused, weights

class MultiViewTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, num_layers=2, dropout=0.1):
        super(MultiViewTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=embed_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        return self.transformer(x)

class IntraModalFusion(nn.Module):
    def __init__(self, deep_dim, predesigned_dim, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.deep_proj = nn.Linear(deep_dim, embed_dim)
        self.predesigned_proj = nn.Linear(predesigned_dim, embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, deep_feat, predesigned_feat):
        # Project features to common space
        deep_emb = self.deep_proj(deep_feat)  # Query
        predesigned_emb = self.predesigned_proj(predesigned_feat)  # Key/Value
        
        # Self-attention with deep as query and predesigned as key/value
        attn_output, _ = self.self_attn(deep_emb, predesigned_emb, predesigned_emb)
            
        attn_output = self.norm(attn_output)
        output = self.fc(attn_output)
        
        return output

class BiModalFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, mod1, mod2):
        # Concatenate modalities along sequence dimension
        joint_rep = torch.cat([mod1, mod2], dim=1)  # [B, T1+T2, D]
        
        attn_output, _ = self.self_attn(joint_rep, joint_rep, joint_rep)
            
        attn_output = self.norm(attn_output + joint_rep)  # Residual connection
        output = self.fc(attn_output)
        
        return output

class SpatialAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [B, num_modalities, D]
        weights = self.fc2(torch.tanh(self.fc1(x)))  # [B, num_modalities, 1]
        weights = F.softmax(weights, dim=1)  # [B, num_modalities, 1]
        
        weighted_sum = (x * weights).sum(dim=1)  # [B, D]
        return weighted_sum, weights.squeeze(-1)  # [B, D], [B, num_modalities]


class IAFN(nn.Module):
    def __init__(self, visual_feature_types=None, audio_feature_types=None, 
                 embed_dim=128, num_heads=4, num_layers=1, dropout=0.1):
        super().__init__()
        
        # 设置默认特征类型（仅用于记录，不再用于确定维度）
        if visual_feature_types is None:
            visual_feature_types = ['resnet', 'openface']
        if audio_feature_types is None:
            audio_feature_types = ['wav2vec', 'myopensmile']
        
        # 特征维度（根据提供的信息）
        visual_dims = {
            'openface': 709,     # 预设计特征
            'resnet': 1000,      # 深度特征
            'densenet': 1000     # 深度特征备选
        }
        audio_dims = {
            'myopensmile': 6373,   # 预设计特征
            'mfccs': 64,         # 预设计特征备选
            'opensmile': 6373,   # 预设计特征备选
            'wav2vec': 512       # 深度特征
        }

        self.embed_dim = embed_dim
        
        # 使用增强版的多层Transformer进行模态内融合
        self.visual_intra_fusion = EnhancedIntraModalFusion(
            deep_dim=visual_dims[visual_feature_types[0]],
            predesigned_dim=visual_dims[visual_feature_types[1]],
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers
        )
        
        self.audio_intra_fusion = EnhancedIntraModalFusion(
            deep_dim=audio_dims[audio_feature_types[0]],
            predesigned_dim=audio_dims[audio_feature_types[1]],
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers
        )
        
        # 使用增强版的多层Transformer进行模态间融合
        self.bimodal_fusion = EnhancedBiModalFusion(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers
        )
        
    def forward(self, visual_deep, visual_predesigned, audio_deep, audio_predesigned):
        # 模态内融合
        v_fused = self.visual_intra_fusion(visual_deep, visual_predesigned)
        a_fused = self.audio_intra_fusion(audio_deep, audio_predesigned)
        
        # 模态间融合
        va_fused = self.bimodal_fusion(v_fused, a_fused)
        
        # 全局池化获得最终特征表示
        va_pooled = va_fused.mean(dim=1)  # [batch_size, embed_dim]
        
        # 计算模态权重
        modal_weights = {
            'visual': torch.ones(visual_deep.size(0), device=visual_deep.device),
            'audio': torch.ones(audio_deep.size(0), device=audio_deep.device)
        }
        
        return va_pooled, modal_weights



class PersonalityMultiModalGate(nn.Module):
    def __init__(self, embed_dim, dropout=0.2):
        super(PersonalityMultiModalGate, self).__init__()
        
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 2),
            nn.Softmax(dim=1)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        
    def forward(self, hp, hm):
        """
        融合人格特征和多模态特征
        
        Args:
            hp: 人格特征 [B, embed_dim]
            hm: 多模态特征 [B, embed_dim]
            
        Returns:
            fused: 融合后的特征 [B, embed_dim]
        """
        # 拼接特征
        concat = torch.cat([hp, hm], dim=1)
        
        # 计算门控权重
        weights = self.gate(concat)  # [B, 2]
        
        # 加权融合
        weighted_hp = hp * weights[:, 0].unsqueeze(1)
        weighted_hm = hm * weights[:, 1].unsqueeze(1)
        
        # 特征融合
        fused = self.fusion(concat) + weighted_hp + weighted_hm  # 添加残差连接
        
        return fused

class ThreeWayFusion(nn.Module):
    def __init__(self, embed_dim, dropout=0.2):
        super(ThreeWayFusion, self).__init__()
        # 计算注意力权重
        self.attention = nn.Sequential(
            nn.Linear(embed_dim * 3, 3),
            nn.Softmax(dim=1)
        )
        
        # 特征组合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        
    def forward(self, hp, hm, hu):
        """
        三路特征融合
        
        Args:
            hp: 人格特征 [B, embed_dim]
            hm: 样本级多模态特征 [B, embed_dim]
            hu: 用户级特征 [B, embed_dim]
        """
        # 拼接所有特征
        combined = torch.cat([hp, hm, hu], dim=1)  # [B, 3*embed_dim]
        
        # 计算注意力权重
        weights = self.attention(combined)  # [B, 3]
        
        # 加权组合
        weighted_hp = hp * weights[:, 0].unsqueeze(1)
        weighted_hm = hm * weights[:, 1].unsqueeze(1)
        weighted_hu = hu * weights[:, 2].unsqueeze(1)
        
        # 特征融合
        output = self.fusion_layer(combined) + weighted_hp + weighted_hm + weighted_hu  # 添加残差连接
        
        return output
def check_tensor_numerics(tensor, tensor_name="tensor"):
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {tensor_name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {tensor_name}")

class EnhancedIntraModalFusion(nn.Module):
    def __init__(self, deep_dim, predesigned_dim=None, embed_dim=128, num_heads=4, dropout=0.1, num_layers=3):
        super().__init__()
        # 特征投影层
        self.deep_projection = nn.Linear(deep_dim, embed_dim)
        if predesigned_dim is not None:
            self.predesigned_projection = nn.Linear(predesigned_dim, embed_dim)
        
        # 多层Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4*embed_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 添加LayerNorm进行稳定化
        self.norm = nn.LayerNorm(embed_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模块权重，防止梯度爆炸"""
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                # Transformer权重使用Xavier初始化
                nn.init.xavier_uniform_(p, gain=0.02)
            elif 'bias' in name:
                nn.init.zeros_(p)
        
    def forward(self, deep_features, predesigned_features=None):
        # 记录输入统计信息
        deep_stats = {
            "mean": deep_features.mean().item(),
            "std": deep_features.std().item(),
            "min": deep_features.min().item(),
            "max": deep_features.max().item()
        }
        #print(f"Deep features stats: {deep_stats}")
        
        if predesigned_features is not None:
            pre_stats = {
                "mean": predesigned_features.mean().item(),
                "std": predesigned_features.std().item(), 
                "min": predesigned_features.min().item(),
                "max": predesigned_features.max().item()
            }
            #print(f"Predesigned features stats: {pre_stats}")
        
        # 分步跟踪投影操作
        #print(f"Deep proj layer weights stats: mean={self.deep_projection.weight.mean().item()}, max={self.deep_projection.weight.max().item()}")
        deep_proj = self.deep_projection(deep_features)
        #print(f"After projection, deep_proj stats: mean={deep_proj.mean().item()}, max={deep_proj.max().item()}")
        
        if hasattr(self, 'predesigned_projection') and predesigned_features is not None:
            #print(f"Predesigned proj layer weights stats: mean={self.predesigned_projection.weight.mean().item()}, max={self.predesigned_projection.weight.max().item()}")
            predesigned_proj = self.predesigned_projection(predesigned_features)
            #print(f"After projection, predesigned_proj stats: mean={predesigned_proj.mean().item()}, max={predesigned_proj.max().item()}")
            x = torch.cat([deep_proj, predesigned_proj], dim=1)
        else:
            x = deep_proj
    
        check_tensor_numerics(x, "x_before_transformer")
    
        x = self.transformer_encoder(x)
    
        # 添加检测Transformer输出后立即检测
        check_tensor_numerics(x, "x_after_transformer")
    
        x = self.norm(x)
    
        check_tensor_numerics(x, "x_after_norm")
    
        return x


class EnhancedBiModalFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1, num_layers=3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4*embed_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 添加LayerNorm进行稳定化
        self.norm = nn.LayerNorm(embed_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模块权重，防止梯度爆炸"""
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.02)
            elif 'bias' in name:
                nn.init.zeros_(p)
        
    def forward(self, v_features, a_features):
        # 拼接特征
        concat_features = torch.cat([v_features, a_features], dim=1)
        
        # 应用Transformer
        x = self.transformer_encoder(concat_features)
        
        # 应用LayerNorm进行稳定化
        x = self.norm(x)
        
        # 数值稳定性检查
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return x

class ThreeWayFusion(nn.Module):
    def __init__(self, embed_dim, dropout=0.2):
        super(ThreeWayFusion, self).__init__()
        # 计算注意力权重
        self.attention = nn.Sequential(
            nn.Linear(embed_dim * 3, 3),
            nn.Softmax(dim=1)
        )
        
        # 特征组合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        
        # 添加额外的LayerNorm
        self.norm = nn.LayerNorm(embed_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模块权重"""
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.02)
            elif 'bias' in name:
                nn.init.zeros_(p)
        
    def forward(self, hp, hm, hu):
        """
        三路特征融合
        
        Args:
            hp: 人格特征 [B, embed_dim]
            hm: 样本级多模态特征 [B, embed_dim]
            hu: 用户级特征 [B, embed_dim]
        """
        # 检查输入是否包含NaN或Inf
        for tensor, name in zip([hp, hm, hu], ['hp', 'hm', 'hu']):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                # 替换NaN和Inf值
                check_tensor_numerics(tensor, "tensor_name")
        
        # 拼接所有特征
        combined = torch.cat([hp, hm, hu], dim=1)  # [B, 3*embed_dim]
        
        # 计算注意力权重
        weights = self.attention(combined)  # [B, 3]
        
        # 加权组合
        weighted_hp = hp * weights[:, 0].unsqueeze(1)
        weighted_hm = hm * weights[:, 1].unsqueeze(1)
        weighted_hu = hu * weights[:, 2].unsqueeze(1)
        
        # 特征融合
        output = self.fusion_layer(combined) + weighted_hp + weighted_hm + weighted_hu
        
        # 应用额外的LayerNorm
        output = self.norm(output)
        
        # 数值稳定性检查
        check_tensor_numerics(output, "output")
        
        return output



class FullModel2(nn.Module):
    def __init__(self, K=10, p2k_hidden=32, p2k_out_dim=64,
                 iafn_embed_dim=128, iafn_heads=4, gate_embed_dim=128,
                 visual_feature_types=None, audio_feature_types=None,
                 num_classes=2, dropout=0.3, grad_clip_value=1.0):
        super(FullModel2, self).__init__()
        self.K = K  # 最大可能的组件数
        self.p2k_hidden = p2k_hidden
        self.p2k_out_dim = p2k_out_dim
        self.active_components = None  # 将在fit_dpgmm中设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.has_personality = True  # 标记是否使用性格特征
        self.dpgmm = None
        self.cluster_labels_dict = None 
        self.cluster_centers = None
        self.dropout = dropout
        self.grad_clip_value = grad_clip_value  # 保存梯度裁剪值
        
        # 设置默认特征类型
        if visual_feature_types is None:
            visual_feature_types = ['resnet', 'openface']
        if audio_feature_types is None:
            audio_feature_types = ['wav2vec', 'myopensmile']

        
        # 记录各特征的标准归一化参数(会在训练过程中更新)
        self.feature_stats = {
            'visual_deep': {'mean': None, 'std': None},
            'visual_predesigned': {'mean': None, 'std': None},
            'audio_deep': {'mean': None, 'std': None},
            'audio_predesigned': {'mean': None, 'std': None},
        }
            
        # P2K核列表
        self.p2k_models = nn.ModuleList([SimpleP2K(input_dim=5, hidden_dim=p2k_hidden, output_dim=p2k_out_dim) for _ in range(K)])
        self.mv_transformer = MultiViewTransformer(embed_dim=p2k_out_dim, num_heads=4, num_layers=1, dropout=dropout)

        # 多模态特征处理 - 修改后的IAFN
        self.iafn = IAFN(
            visual_feature_types=visual_feature_types,
            audio_feature_types=audio_feature_types,
            embed_dim=iafn_embed_dim,
            num_heads=iafn_heads,
            dropout=dropout
        )
        
        self.p2k_to_gate = nn.Sequential(
            nn.Linear(p2k_out_dim, gate_embed_dim),
            nn.LayerNorm(gate_embed_dim),  # 添加LayerNorm
            nn.Dropout(dropout)
        )
        self.iafn_to_gate = nn.Sequential(
            nn.Linear(iafn_embed_dim, gate_embed_dim),
            nn.LayerNorm(gate_embed_dim),  # 添加LayerNorm
            nn.Dropout(dropout)
        )
        self.gate_fusion = PersonalityMultiModalGate(gate_embed_dim, dropout=dropout)
        
        # 用户级特征处理组件
        self.user_feat_proj = nn.Sequential(
            nn.Linear(iafn_embed_dim, gate_embed_dim),
            nn.LayerNorm(gate_embed_dim),  # 添加LayerNorm
            nn.Dropout(dropout)
        )
        
        # 三路融合模块
        self.three_way_fusion = ThreeWayFusion(gate_embed_dim, dropout=dropout)

        # 直接处理多模态特征的线性层（用于无性格特征的情况）
        self.multimodal_only_fc = nn.Sequential(
            nn.Linear(iafn_embed_dim, gate_embed_dim),
            nn.LayerNorm(gate_embed_dim),  # 添加LayerNorm
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(gate_embed_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),  # 添加LayerNorm
            nn.Dropout(dropout),  
            nn.Linear(64, num_classes)
        )
        
        # DP-GMM聚类模型
        self.dpgmm = None
        self.cluster_id_to_indices = None
        self.personality_to_cluster_pos = None
        self.max_cluster_size = None
        self.multi_view_embedding = None
        self.active_components = None  # 存储实际活跃的聚类数量
        
        # 创建默认的性格特征表示
        self.default_personality_embedding = nn.Parameter(
            torch.zeros(p2k_out_dim), requires_grad=True
        )
        self.cluster_labels_dict = None
        
        # 存储用户级特征
        self.user_features_dict = None
        
        # 梯度监控相关
        self.gradient_norms = {}  # 用于存储各层梯度范数
        self.gradient_stats = {'max': 0, 'min': 0, 'mean': 0, 'median': 0}  # 梯度统计信息
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重，防止梯度爆炸"""
        for name, p in self.named_parameters():
            if p.dim() > 1:  # 权重矩阵
                if 'transformer' in name:
                    # Transformer需要更小的初始化值
                    nn.init.xavier_uniform_(p, gain=0.02)
                else:
                    nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
    def normalize_features(self, x, feature_type):
        """特征归一化，处理极端值"""
        if x is None:
            return None
            
        # 检查并替换NaN/Inf值
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 对于预设计音频特征使用特殊处理
        if feature_type == 'audio_predesigned':
            # 使用鲁棒的特征变换
            with torch.no_grad():
                try:
                    # 尝试在GPU上计算
                    q_high = torch.quantile(x.abs().view(-1), 0.99)
                except RuntimeError:
                    # 如果GPU内存不足，转到CPU上计算
                    q_high = torch.quantile(x.abs().cpu().view(-1), 0.99).to(x.device)
                
                # 避免阈值过小
                threshold = max(q_high.item(), 100.0)
                # 对数变换 + 归一化
                sign = torch.sign(x)
                x_abs = torch.clamp(x.abs(), 0, threshold)
                x_norm = torch.log1p(x_abs) / torch.log(torch.tensor(threshold + 1.0, device=x.device))
                return sign * x_norm
        else:
            # 标准归一化 - 按批次计算
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True) + 1e-5
            return (x - mean) / std
     
    def check_gradients(self):
        """检查并记录模型梯度"""
        self.gradient_norms = {}
        all_grads = []
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                # 检查梯度是否包含NaN或Inf
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    # 将NaN和Inf替换为0
                    param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                
                # 记录梯度范数
                grad_norm = param.grad.norm().item()
                self.gradient_norms[name] = grad_norm
                all_grads.append(grad_norm)
                
        if all_grads:
            all_grads = torch.tensor(all_grads)
            self.gradient_stats = {
                'max': all_grads.max().item(),
                'min': all_grads.min().item(),
                'mean': all_grads.mean().item(),
                'median': all_grads.median().item()
            }
        
        return self.gradient_stats
    
    def save_clustering_info(self, filepath):
        """
        保存聚类信息到文件
        
        Args:
            filepath: 保存路径
        """
        # 将张量转换为numpy数组
        cluster_centers = self.cluster_centers.cpu().numpy() if self.cluster_centers is not None else None
        scaler_mean = self.scaler_mean.cpu().numpy() if hasattr(self, 'scaler_mean') else None
        scaler_scale = self.scaler_scale.cpu().numpy() if hasattr(self, 'scaler_scale') else None
        
        # 准备保存的数据
        cluster_info = {
            'actual_k': self.actual_k if hasattr(self, 'actual_k') else self.K,
            'cluster_centers': cluster_centers,
            'cluster_labels_dict': self.cluster_labels_dict,
            'cluster_id_to_indices': self.cluster_id_to_indices,
            'scaler_mean': scaler_mean,
            'scaler_scale': scaler_scale
        }
        
        # 保存到文件
        with open(filepath, 'wb') as f:
            pickle.dump(cluster_info, f)
        
    def load_clustering_info(self, filepath, device=None):
        """
        从文件加载聚类信息
        
        Args:
            filepath: 聚类信息文件路径
            device: 模型运行的设备
        """
        if device is None:
            device = self.device
        
        with open(filepath, 'rb') as f:
            cluster_info = pickle.load(f)
        
        # 更新聚类数量
        self.actual_k = cluster_info['actual_k']
        self.K = self.actual_k
        
        # 重建P2K模型列表，确保大小匹配
        self.p2k_models = nn.ModuleList([
            SimpleP2K(input_dim=5, hidden_dim=self.p2k_hidden, output_dim=self.p2k_out_dim).to(device)
            for _ in range(self.K)
        ])
        
        # 对新创建的P2K模型初始化权重
        for module in self.p2k_models:
            for name, p in module.named_parameters():
                if p.dim() > 1:  # 权重矩阵
                    nn.init.xavier_uniform_(p)
                elif 'bias' in name:
                    nn.init.zeros_(p)
        
        # 加载聚类中心和其他聚类信息
        if 'cluster_centers' in cluster_info and cluster_info['cluster_centers'] is not None:
            self.cluster_centers = torch.tensor(cluster_info['cluster_centers'], device=device)
        
        if 'cluster_labels_dict' in cluster_info:
            self.cluster_labels_dict = cluster_info['cluster_labels_dict']
        
        if 'cluster_id_to_indices' in cluster_info:
            self.cluster_id_to_indices = cluster_info['cluster_id_to_indices']
        
        if 'scaler_mean' in cluster_info and cluster_info['scaler_mean'] is not None:
            self.scaler_mean = torch.tensor(cluster_info['scaler_mean'], device=device)
        
        if 'scaler_scale' in cluster_info and cluster_info['scaler_scale'] is not None:
            self.scaler_scale = torch.tensor(cluster_info['scaler_scale'], device=device)
        
        # 标记模型已加载聚类信息
        self.has_personality = True if self.cluster_centers is not None else False
        
        return cluster_info
    
    def extract_user_features(self, user_data_dict):
        """
        提取用户级特征，使用LSTM处理用户的多个样本序列
        
        Args:
            user_data_dict: 用户数据字典 {user_id: {
                'visual_deep': [...], 
                'visual_predesigned': [...],
                'audio_deep': [...], 
                'audio_predesigned': [...]
            }}
        
        Returns:
            user_features: 用户特征字典 {user_id: tensor}
        """
        device = next(self.parameters()).device
        user_features = {}
        
        # 创建LSTM处理用户序列
        if not hasattr(self, 'user_sequence_lstm'):
            self.user_sequence_lstm = nn.LSTM(
                input_size=self.iafn.embed_dim,
                hidden_size=self.iafn.embed_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            ).to(device)
            # 初始化LSTM权重
            for name, p in self.user_sequence_lstm.named_parameters():
                if 'weight' in name and p.dim() > 1:
                    nn.init.orthogonal_(p)  # LSTM推荐使用正交初始化
                elif 'bias' in name:
                    nn.init.zeros_(p)
        
        for user_id, data in user_data_dict.items():
            visual_deep = [feat.to(device) for feat in data['visual_deep']]
            visual_predesigned = [feat.to(device) for feat in data['visual_predesigned']]
            audio_deep = [feat.to(device) for feat in data['audio_deep']]
            audio_predesigned = [feat.to(device) for feat in data['audio_predesigned']]
            
            # 处理每个样本
            sample_features = []
            for i in range(len(visual_deep)):
                # 添加批次维度
                v_deep = visual_deep[i].unsqueeze(0)
                v_predesigned = visual_predesigned[i].unsqueeze(0)
                a_deep = audio_deep[i].unsqueeze(0)
                a_predesigned = audio_predesigned[i].unsqueeze(0)
                
                # 数值稳定性检查
                v_deep = torch.nan_to_num(v_deep, nan=0.0, posinf=1e6, neginf=-1e6)
                v_predesigned = torch.nan_to_num(v_predesigned, nan=0.0, posinf=1e6, neginf=-1e6)
                a_deep = torch.nan_to_num(a_deep, nan=0.0, posinf=1e6, neginf=-1e6)
                a_predesigned = torch.nan_to_num(a_predesigned, nan=0.0, posinf=1e6, neginf=-1e6)
                
                # 提取多模态特征
                with torch.no_grad():
                    hm_i, _ = self.iafn(v_deep, v_predesigned, a_deep, a_predesigned)
                sample_features.append(hm_i)
            
            if sample_features:
                # 将样本特征堆叠成序列
                sequence = torch.cat(sample_features, dim=0)  # [num_samples, iafn_embed_dim]
                
                # 数值稳定性检查
                sequence = torch.nan_to_num(sequence, nan=0.0, posinf=1e6, neginf=-1e6)
                
                # 如果样本数量太少，可以考虑填充或跳过LSTM
                if sequence.size(0) >= 2:  # 至少需要2个样本才能捕获序列信息
                    # 添加批次维度
                    sequence = sequence.unsqueeze(0)  # [1, num_samples, iafn_embed_dim]
                    
                    # 通过LSTM处理序列
                    with torch.no_grad():
                        lstm_out, (h_n, c_n) = self.user_sequence_lstm(sequence)
                    
                    # 使用最后一个时间步的隐藏状态作为用户表示
                    # 对于双向LSTM，合并两个方向的最终隐藏状态
                    forward = h_n[0]  # 前向LSTM的最终隐藏状态
                    backward = h_n[1]  # 后向LSTM的最终隐藏状态
                    user_feat = torch.cat([forward, backward], dim=1)  # [2*iafn_embed_dim]
                    
                    # 映射回原始维度
                    if not hasattr(self, 'lstm_projection'):
                        self.lstm_projection = nn.Linear(2 * self.iafn.embed_dim, self.iafn.embed_dim).to(device)
                        # 初始化权重
                        nn.init.xavier_uniform_(self.lstm_projection.weight)
                        nn.init.zeros_(self.lstm_projection.bias)
                    
                    user_feat = self.lstm_projection(user_feat.unsqueeze(0)).squeeze(0)  # [iafn_embed_dim]
                else:
                    # 样本太少，回退到平均池化
                    user_feat = sequence.mean(dim=0)  # [iafn_embed_dim]
                
                # 数值稳定性检查
                user_feat = torch.nan_to_num(user_feat, nan=0.0, posinf=1e6, neginf=-1e6)
                    
                user_features[user_id] = user_feat
        
        return user_features
    
    def fit_dpgmm(self, all_personality_features, all_user_ids, user_multimodal_features=None):
        if all_personality_features is None or all_personality_features.shape[0] <= 1:
            self.has_personality = False
            print("警告: 没有足够的性格特征数据进行聚类，将使用纯多模态模式")
            return
        
        device = next(self.parameters()).device
        personality_np = all_personality_features.detach().cpu().numpy()
        
        # 检查数据是否包含NaN或Inf值
        if np.isnan(personality_np).any() or np.isinf(personality_np).any():
            print("警告: 性格特征数据包含NaN或Inf值，将被替换")
            check_tensor_numerics(personality_feat, "personality_feat")
        
        scaler = StandardScaler()
        personality_np = scaler.fit_transform(personality_np)
        self.scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
        self.scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)
        
        # 如果有用户级多模态特征，则与个人特征融合后再进行聚类
        if user_multimodal_features is not None:
            print("检测到用户级多模态特征，将与个人特征融合后再进行聚类")
            
            # 保存用户级特征字典
            self.user_features_dict = {user_id: feat for user_id, feat in zip(all_user_ids, user_multimodal_features)}
            
            mm_features_np = user_multimodal_features.detach().cpu().numpy()
            
            # 检查数据是否包含NaN或Inf值
            if np.isnan(mm_features_np).any() or np.isinf(mm_features_np).any():
                print("警告: 用户级多模态特征包含NaN或Inf值，将被替换")
                mm_features_np = np.nan_to_num(mm_features_np, nan=0.0, posinf=1e6, neginf=-1e6)
                
            mm_scaler = StandardScaler()
            mm_features_np = mm_scaler.fit_transform(mm_features_np)
            
            # 将用户级多模态特征与个人特征融合
            combined_features = np.concatenate([personality_np, mm_features_np], axis=1)
            
            print(f"融合特征维度: {combined_features.shape}")
            clustering_features = combined_features
        else:
            print("仅使用个人特征进行聚类")
            clustering_features = personality_np
        
        # 聚类逻辑
        metrics = {'silhouette': [], 'dbi': [], 'ch': []}
        cluster_labels_dict = {}
        for k in range(2, 9):
            hierarchical = AgglomerativeClustering(n_clusters=k, linkage='ward')
            cluster_labels = hierarchical.fit_predict(clustering_features)
            metrics['silhouette'].append(silhouette_score(clustering_features, cluster_labels))
            metrics['dbi'].append(davies_bouldin_score(clustering_features, cluster_labels))
            metrics['ch'].append(calinski_harabasz_score(clustering_features, cluster_labels))
            cluster_labels_dict[k] = cluster_labels
        
        silhouette_vals = np.array(metrics['silhouette'])
        dbi_vals = np.array(metrics['dbi'])
        ch_vals = np.array(metrics['ch'])
        norm_silhouette = (silhouette_vals + 1) / 2
        norm_dbi = 1 - (dbi_vals - dbi_vals.min()) / (dbi_vals.max() - dbi_vals.min() + 1e-10)
        norm_ch = (ch_vals - ch_vals.min()) / (ch_vals.max() - ch_vals.min() + 1e-10)
        w_silhouette, w_dbi, w_ch = 0.4, 0.3, 0.3
        scores = w_silhouette * norm_silhouette + w_dbi * norm_dbi + w_ch * norm_ch
        best_k = np.argmax(scores) + 2
        best_cluster_labels = cluster_labels_dict[best_k]
        print(f"选择最佳 K={best_k}, 综合得分={scores[best_k - 2]:.4f}")
        
        self.cluster_labels_dict = {user_id: int(label) for user_id, label in zip(all_user_ids, best_cluster_labels)}
        self.cluster_id_to_indices = {k: np.where(best_cluster_labels == k)[0] for k in range(best_k)}
        self.actual_k = best_k
        print(f"实际使用的聚类数量: {self.actual_k}")
        
        # 存储聚类中心（基于原始个人特征，而非融合特征）
        self.cluster_centers = []
        for k in range(self.actual_k):
            cluster_indices = self.cluster_id_to_indices[k]
            if len(cluster_indices) > 0:
                cluster_features = personality_np[cluster_indices]
                center = np.mean(cluster_features, axis=0)
            else:
                center = np.zeros(personality_np.shape[1])
            self.cluster_centers.append(center)
        self.cluster_centers = torch.tensor(self.cluster_centers, dtype=torch.float32, device=device)
        
        # 调整p2k_models并移至正确设备
        if self.actual_k != self.K:
            self.p2k_models = nn.ModuleList([
                SimpleP2K(input_dim=5, hidden_dim=self.p2k_hidden, output_dim=self.p2k_out_dim).to(device)
                for _ in range(self.actual_k)
            ])
            self.K = self.actual_k
            
            # 对新创建的P2K模型初始化权重
            for module in self.p2k_models:
                for name, p in module.named_parameters():
                    if p.dim() > 1:  # 权重矩阵
                        nn.init.xavier_uniform_(p)
                    elif 'bias' in name:
                        nn.init.zeros_(p)
        
        # 聚类分析输出
        print("\n=== 融合特征层次聚类分析 ===")
        total_users = 0
        for k in range(best_k):
            user_indices = self.cluster_id_to_indices[k]
            user_count = len(user_indices)
            total_users += user_count
            user_ids = [all_user_ids[idx] for idx in user_indices]
            print(f"聚类 {k}:")
            print(f"  用户数量: {user_count}")
            print(f"  用户 ID: {user_ids[:10]}{'...' if len(user_ids) > 10 else ''}")
        print(f"总用户数量: {total_users} (应等于数据集大小: {len(all_user_ids)})")
        
        # 保存融合特征对应的聚类结果供后续分析
        if user_multimodal_features is not None:
            self.fused_clustering_features = clustering_features
            self.fused_cluster_labels = best_cluster_labels
    
    def predict_cluster(self, personality_feat):
        """
        使用层次聚类预测人格特征所属的聚类
        
        Args:
            personality_feat: 人格特征 [B, 5]
            
        Returns:
            cluster_ids: 聚类ID [B]
            pos_in_clusters: 簇内位置 [B]（层次聚类无序，设为0）
            cluster_probs: 伪聚类概率 [B, actual_k]（硬分配为 one-hot）
        """
        if not self.has_personality or self.cluster_centers is None:
            batch_size = personality_feat.size(0)
            return np.zeros(batch_size, dtype=int), np.zeros(batch_size, dtype=int), None
        
        with torch.no_grad():
            # 检查输入是否包含NaN或Inf
            if torch.isnan(personality_feat).any() or torch.isinf(personality_feat).any():
                check_tensor_numerics(personality_feat, "personality_feat")
            
            # 标准化输入特征
            personality_std = (personality_feat - self.scaler_mean) / self.scaler_scale
            distances = torch.cdist(personality_std, self.cluster_centers)
            cluster_ids = torch.argmin(distances, dim=1)
            cluster_probs = F.one_hot(cluster_ids, num_classes=self.actual_k).float()
            pos_in_clusters = torch.zeros_like(cluster_ids)  # 层次聚类无簇内排序
            return cluster_ids.cpu().numpy(), pos_in_clusters.cpu().numpy(), cluster_probs
    
    def forward(self, personality_feat, visual_deep, visual_predesigned, audio_deep, audio_predesigned, 
            visual_mask=None, audio_mask=None, user_id=None, user_history_data=None):

        device = personality_feat.device
        batch_size = personality_feat.size(0)

        # 改为检测，不再自动替换nan/inf
        check_tensor_numerics(personality_feat, "personality_feat")
        # 输入特征归一化
        visual_deep = self.normalize_features(visual_deep, 'visual_deep')
        visual_predesigned = self.normalize_features(visual_predesigned, 'visual_predesigned')
        audio_deep = self.normalize_features(audio_deep, 'audio_deep')
        audio_predesigned = self.normalize_features(audio_predesigned, 'audio_predesigned')
        
        # 检查归一化后的特征
        if visual_deep is not None:
            check_tensor_numerics(visual_deep, "normalized_visual_deep")
        if visual_predesigned is not None:
            check_tensor_numerics(visual_predesigned, "normalized_visual_predesigned")
        if audio_deep is not None:
            check_tensor_numerics(audio_deep, "normalized_audio_deep")
        if audio_predesigned is not None:
            check_tensor_numerics(audio_predesigned, "normalized_audio_predesigned")

        # 处理多模态输入
        if all(x is not None for x in [visual_deep, visual_predesigned, audio_deep, audio_predesigned]):
            hm, modal_weights = self.iafn(visual_deep, visual_predesigned, audio_deep, audio_predesigned)
            hm_mapped = self.iafn_to_gate(hm)

            check_tensor_numerics(hm_mapped, "hm_mapped")
        else:
            hm = None
            hm_mapped = None

        # 处理人格特征
        if not self.has_personality or self.cluster_centers is None:
            default_personality = self.default_personality_embedding.expand(batch_size, -1)
            hp_mapped = self.p2k_to_gate(default_personality)
            check_tensor_numerics(hp_mapped, "hp_mapped")
        else:
            cluster_ids, pos_in_clusters, cluster_probs = self.predict_cluster(personality_feat)
            cluster_probs = cluster_probs.to(device)

            hp_all = []
            for i in range(batch_size):
                assigned_cluster = cluster_ids[i]
                p2k_output = self.p2k_models[assigned_cluster](personality_feat[i:i+1])
                hp_all.append(p2k_output)
            hpTrans_all = torch.cat(hp_all, dim=0)
            hp_mapped = self.p2k_to_gate(hpTrans_all)

            check_tensor_numerics(hp_mapped, "hp_mapped")

        # 处理用户级特征 - 直接在forward中提取
        hu_mapped = torch.zeros_like(hp_mapped)
        if user_id is not None and user_history_data is not None:
            # 确保LSTM已初始化
            if not hasattr(self, 'user_sequence_lstm'):
                self.user_sequence_lstm = nn.LSTM(
                    input_size=self.iafn.embed_dim,
                    hidden_size=self.iafn.embed_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True
                ).to(device)
                # 初始化LSTM权重
                for name, p in self.user_sequence_lstm.named_parameters():
                    if 'weight' in name and p.dim() > 1:
                        nn.init.orthogonal_(p)  # LSTM推荐使用正交初始化
                    elif 'bias' in name:
                        nn.init.zeros_(p)

            if not hasattr(self, 'lstm_projection'):
                self.lstm_projection = nn.Linear(2 * self.iafn.embed_dim, self.iafn.embed_dim).to(device)
                # 初始化权重
                nn.init.xavier_uniform_(self.lstm_projection.weight)
                nn.init.zeros_(self.lstm_projection.bias)

            hu_batch = []
            for i, uid in enumerate(user_id):
                if uid in user_history_data:
                    history_data = user_history_data[uid]
                    history_features = []
                    for hist_idx in range(len(history_data['visual_deep'])):
                        v_deep = history_data['visual_deep'][hist_idx].unsqueeze(0).to(device)
                        v_predesigned = history_data['visual_predesigned'][hist_idx].unsqueeze(0).to(device)
                        a_deep = history_data['audio_deep'][hist_idx].unsqueeze(0).to(device)
                        a_predesigned = history_data['audio_predesigned'][hist_idx].unsqueeze(0).to(device)

                        # 替换为检测
                        check_tensor_numerics(v_deep, "user_history visual_deep")
                        check_tensor_numerics(v_predesigned, "user_history visual_predesigned")
                        check_tensor_numerics(a_deep, "user_history audio_deep")
                        check_tensor_numerics(a_predesigned, "user_history audio_predesigned")

                        hist_feat, _ = self.iafn(v_deep, v_predesigned, a_deep, a_predesigned)
                        check_tensor_numerics(hist_feat, "user_history hist_feat")
                        history_features.append(hist_feat)

                    if hm is not None:
                        curr_feat = hm[i:i+1]
                        check_tensor_numerics(curr_feat, "user_history current hm")
                        history_features.append(curr_feat)

                    if history_features:
                        sequence = torch.cat(history_features, dim=0)
                        check_tensor_numerics(sequence, "user_history sequence")

                        if sequence.size(0) >= 2:
                            sequence = sequence.unsqueeze(0)
                            lstm_out, (h_n, c_n) = self.user_sequence_lstm(sequence)
                            forward = h_n[0]
                            backward = h_n[1]
                            user_feat = torch.cat([forward, backward], dim=1)
                            check_tensor_numerics(user_feat, "user_history lstm user_feat")
                            user_feat = self.lstm_projection(user_feat)
                        else:
                            user_feat = sequence.mean(dim=0).unsqueeze(0)

                        check_tensor_numerics(user_feat, "user_history final user_feat")
                        hu_batch.append(user_feat)
                    else:
                        hu_batch.append(torch.zeros(1, self.iafn.embed_dim, device=device))
                else:
                    hu_batch.append(torch.zeros(1, self.iafn.embed_dim, device=device))

            if hu_batch:
                hu = torch.cat(hu_batch, dim=0)
                hu_mapped = self.user_feat_proj(hu)
                check_tensor_numerics(hu_mapped, "hu_mapped")

        # 特征融合
        if hm_mapped is not None:
            hfinal = self.three_way_fusion(hp_mapped, hm_mapped, hu_mapped)
        else:
            hfinal = self.gate_fusion(hp_mapped, hu_mapped)

        check_tensor_numerics(hfinal, "hfinal")

        output = self.classifier(hfinal)
        check_tensor_numerics(output, "output")

        return output, hfinal

class SoftGatingNetwork(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, num_experts=10):
        super(SoftGatingNetwork, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )
        
        # Initialize to more uniform distribution
        with torch.no_grad():
            self.gate[-1].bias.fill_(0.0)
            nn.init.xavier_uniform_(self.gate[-1].weight, gain=0.1)
            
        self.temperature = 1  # Start with higher temperature for more uniform distribution

    def forward(self, personality_feat):
        gate_logits = self.gate(personality_feat)
        gate_logits = gate_logits / self.temperature
        gate_logits = gate_logits - gate_logits.max(dim=1, keepdim=True)[0]
        gate_weights = F.softmax(gate_logits, dim=1)
        # 打印部分样本的权重分布，选前5个样本，避免输出太多
        gate_weights_np = gate_weights.detach().cpu().numpy()
        # 也可以打印最大权重，观察是否过于集中
        max_weights = gate_weights.max(dim=1)[0]
        #print(f"[SoftGatingNetwork] Max gate weight per sample (first 5): {max_weights[:5].detach().cpu().numpy()}")
        return gate_weights
class FullModel3(nn.Module):
    def __init__(self, K=10, p2k_hidden=32, p2k_out_dim=64,
                 iafn_embed_dim=128, iafn_heads=4, gate_embed_dim=128,
                 visual_feature_types=None, audio_feature_types=None,
                 num_classes=2, dropout=0.5, grad_clip_value=1.0):
        super(FullModel3, self).__init__()
        self.K = K  # 最大可能的组件数
        self.p2k_hidden = p2k_hidden
        self.p2k_out_dim = p2k_out_dim
        self.active_components = None  # 将在fit_dpgmm中设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.has_personality = True  # 标记是否使用性格特征
        self.dpgmm = None
        self.cluster_labels_dict = None 
        self.cluster_centers = None
        self.dropout = dropout
        self.grad_clip_value = grad_clip_value  # 保存梯度裁剪值
        
        # 设置默认特征类型
        if visual_feature_types is None:
            visual_feature_types = ['resnet', 'openface']
        if audio_feature_types is None:
            audio_feature_types = ['wav2vec', 'myopensmile']

        
        # 记录各特征的标准归一化参数(会在训练过程中更新)
        self.feature_stats = {
            'visual_deep': {'mean': None, 'std': None},
            'visual_predesigned': {'mean': None, 'std': None},
            'audio_deep': {'mean': None, 'std': None},
            'audio_predesigned': {'mean': None, 'std': None},
        }
            
        # P2K核列表
        self.p2k_models = nn.ModuleList([SimpleP2K(input_dim=5, hidden_dim=p2k_hidden, output_dim=p2k_out_dim) for _ in range(K)])
        
        # 添加软门控网络
        self.soft_gate = SoftGatingNetwork(input_dim=5, hidden_dim=p2k_hidden, num_experts=K)
        
        self.mv_transformer = MultiViewTransformer(embed_dim=p2k_out_dim, num_heads=4, num_layers=1, dropout=dropout)

        # 多模态特征处理 - 修改后的IAFN
        self.iafn = IAFN(
            visual_feature_types=visual_feature_types,
            audio_feature_types=audio_feature_types,
            embed_dim=iafn_embed_dim,
            num_heads=iafn_heads,
            dropout=dropout
        )
        
        self.p2k_to_gate = nn.Sequential(
            nn.Linear(p2k_out_dim, gate_embed_dim),
            nn.LayerNorm(gate_embed_dim),  # 添加LayerNorm
            nn.Dropout(dropout)
        )
        self.iafn_to_gate = nn.Sequential(
            nn.Linear(iafn_embed_dim, gate_embed_dim),
            nn.LayerNorm(gate_embed_dim),  # 添加LayerNorm
            nn.Dropout(dropout)
        )
        self.gate_fusion = PersonalityMultiModalGate(gate_embed_dim, dropout=dropout)
        
        # 用户级特征处理组件
        self.user_feat_proj = nn.Sequential(
            nn.Linear(iafn_embed_dim, gate_embed_dim),
            nn.LayerNorm(gate_embed_dim),  # 添加LayerNorm
            nn.Dropout(dropout)
        )
        self.two_way_fusion_mlp = nn.Sequential(
            nn.Linear(gate_embed_dim * 2, gate_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(gate_embed_dim)
        )
        

        # 初始化用户特征处理LSTM和投影层
        self.user_sequence_lstm = nn.LSTM(
            input_size=iafn_embed_dim,
            hidden_size=iafn_embed_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        for name, p in self.user_sequence_lstm.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
                
        self.lstm_projection = nn.Linear(
            2 * iafn_embed_dim, iafn_embed_dim
        )
        nn.init.xavier_uniform_(self.lstm_projection.weight)
        nn.init.zeros_(self.lstm_projection.bias)
        # 三路融合模块
        self.three_way_fusion = ThreeWayFusion(gate_embed_dim, dropout=dropout)

        # 直接处理多模态特征的线性层（用于无性格特征的情况）
        self.multimodal_only_fc = nn.Sequential(
            nn.Linear(iafn_embed_dim, gate_embed_dim),
            nn.LayerNorm(gate_embed_dim),  # 添加LayerNorm
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(gate_embed_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),  # 添加LayerNorm
            nn.Dropout(dropout),  
            nn.Linear(64, num_classes)
        )
        
        # DP-GMM聚类模型
        self.dpgmm = None
        self.cluster_id_to_indices = None
        self.personality_to_cluster_pos = None
        self.max_cluster_size = None
        self.multi_view_embedding = None
        self.active_components = None  # 存储实际活跃的聚类数量
        
        # 创建默认的性格特征表示
        self.default_personality_embedding = nn.Parameter(
            torch.zeros(p2k_out_dim), requires_grad=True
        )
        self.cluster_labels_dict = None
        
        # 存储用户级特征
        self.user_features_dict = None
        
        # 梯度监控相关
        self.gradient_norms = {}  # 用于存储各层梯度范数
        self.gradient_stats = {'max': 0, 'min': 0, 'mean': 0, 'median': 0}  # 梯度统计信息
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重，防止梯度爆炸"""
        for name, p in self.named_parameters():
            if p.dim() > 1:  # 权重矩阵
                if 'transformer' in name:
                    # Transformer需要更小的初始化值
                    nn.init.xavier_uniform_(p, gain=0.02)
                else:
                    nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
                
    def normalize_features(self, x, feature_type):
        """特征归一化，处理极端值"""
        if x is None:
            return None
            
        # 检查并替换NaN/Inf值
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 对于预设计音频特征使用特殊处理
        if feature_type == 'audio_predesigned':
            # 使用鲁棒的特征变换
            with torch.no_grad():
                try:
                    # 尝试在GPU上计算
                    q_high = torch.quantile(x.abs().view(-1), 0.99)
                except RuntimeError:
                    # 如果GPU内存不足，转到CPU上计算
                    # 采样部分数据计算分位数
                    sample_size = 10000
                    indices = torch.randperm(x.numel())[:sample_size]
                    sampled_data = x.view(-1)[indices]
                    q_high = torch.quantile(sampled_data.abs(), 0.99)

                # 避免阈值过小
                threshold = max(q_high.item(), 100.0)
                # 对数变换 + 归一化
                sign = torch.sign(x)
                x_abs = torch.clamp(x.abs(), 0, threshold)
                x_norm = torch.log1p(x_abs) / torch.log(torch.tensor(threshold + 1.0, device=x.device))
                return sign * x_norm
        else:
            # 标准归一化 - 按批次计算
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True) + 1e-5
            return (x - mean) / std
            
    def check_gradients(self):
        """检查并记录模型梯度"""
        self.gradient_norms = {}
        all_grads = []
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                # 检查梯度是否包含NaN或Inf
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    # 将NaN和Inf替换为0
                    param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                
                # 记录梯度范数
                grad_norm = param.grad.norm().item()
                self.gradient_norms[name] = grad_norm
                all_grads.append(grad_norm)
                
        if all_grads:
            all_grads = torch.tensor(all_grads)
            self.gradient_stats = {
                'max': all_grads.max().item(),
                'min': all_grads.min().item(),
                'mean': all_grads.mean().item(),
                'median': all_grads.median().item()
            }
        
        return self.gradient_stats
    
    def save_clustering_info(self, filepath):
        """
        保存聚类信息到文件
        
        Args:
            filepath: 保存路径
        """
        # 将张量转换为numpy数组
        cluster_centers = self.cluster_centers.cpu().numpy() if self.cluster_centers is not None else None
        scaler_mean = self.scaler_mean.cpu().numpy() if hasattr(self, 'scaler_mean') else None
        scaler_scale = self.scaler_scale.cpu().numpy() if hasattr(self, 'scaler_scale') else None
        
        # 准备保存的数据
        cluster_info = {
            'actual_k': self.actual_k if hasattr(self, 'actual_k') else self.K,
            'cluster_centers': cluster_centers,
            'cluster_labels_dict': self.cluster_labels_dict,
            'cluster_id_to_indices': self.cluster_id_to_indices,
            'scaler_mean': scaler_mean,
            'scaler_scale': scaler_scale
        }
        
        # 保存到文件
        with open(filepath, 'wb') as f:
            pickle.dump(cluster_info, f)
        
    def load_clustering_info(self, filepath, device=None):
        """
        从文件加载聚类信息
        
        Args:
            filepath: 聚类信息文件路径
            device: 模型运行的设备
        """
        if device is None:
            device = self.device
        
        with open(filepath, 'rb') as f:
            cluster_info = pickle.load(f)
        
        # 更新聚类数量
        self.actual_k = cluster_info['actual_k']
        self.K = self.actual_k
        
        # 重建P2K模型列表，确保大小匹配
        self.p2k_models = nn.ModuleList([
            SimpleP2K(input_dim=5, hidden_dim=self.p2k_hidden, output_dim=self.p2k_out_dim).to(device)
            for _ in range(self.K)
        ])
        
        # 同时更新软门控网络
        self.soft_gate = SoftGatingNetwork(input_dim=5, hidden_dim=self.p2k_hidden, num_experts=self.K).to(device)
        
        # 对新创建的P2K模型初始化权重
        for module in self.p2k_models:
            for name, p in module.named_parameters():
                if p.dim() > 1:  # 权重矩阵
                    nn.init.xavier_uniform_(p)
                elif 'bias' in name:
                    nn.init.zeros_(p)
        
        # 加载聚类中心和其他聚类信息
        if 'cluster_centers' in cluster_info and cluster_info['cluster_centers'] is not None:
            self.cluster_centers = torch.tensor(cluster_info['cluster_centers'], device=device)
        
        if 'cluster_labels_dict' in cluster_info:
            self.cluster_labels_dict = cluster_info['cluster_labels_dict']
        
        if 'cluster_id_to_indices' in cluster_info:
            self.cluster_id_to_indices = cluster_info['cluster_id_to_indices']
        
        if 'scaler_mean' in cluster_info and cluster_info['scaler_mean'] is not None:
            self.scaler_mean = torch.tensor(cluster_info['scaler_mean'], device=device)
        
        if 'scaler_scale' in cluster_info and cluster_info['scaler_scale'] is not None:
            self.scaler_scale = torch.tensor(cluster_info['scaler_scale'], device=device)
        
        # 标记模型已加载聚类信息
        self.has_personality = True if self.cluster_centers is not None else False
        
        return cluster_info
    
    def fit_dpgmm(self, all_personality_features, all_user_ids, user_multimodal_features=None):
        if all_personality_features is None or all_personality_features.shape[0] <= 1:
            self.has_personality = False
            print("警告: 没有足够的性格特征数据进行聚类，将使用纯多模态模式")
            return
        
        device = next(self.parameters()).device
        personality_np = all_personality_features.detach().cpu().numpy()
        
        # 检查数据是否包含NaN或Inf值
        if np.isnan(personality_np).any() or np.isinf(personality_np).any():
            print("警告: 性格特征数据包含NaN或Inf值，将被替换")
            personality_np = np.nan_to_num(personality_np, nan=0.0, posinf=1e6, neginf=-1e6)
        
        scaler = StandardScaler()
        personality_np = scaler.fit_transform(personality_np)
        self.scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
        self.scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)
        
        # 如果有用户级多模态特征，则与个人特征融合后再进行聚类
        if user_multimodal_features is not None:
            print("检测到用户级多模态特征，将与个人特征融合后再进行聚类")
            
            # 保存用户级特征字典
            self.user_features_dict = {user_id: feat for user_id, feat in zip(all_user_ids, user_multimodal_features)}
            
            mm_features_np = user_multimodal_features.detach().cpu().numpy()
            
            # 检查数据是否包含NaN或Inf值
            if np.isnan(mm_features_np).any() or np.isinf(mm_features_np).any():
                print("警告: 用户级多模态特征包含NaN或Inf值，将被替换")
                mm_features_np = np.nan_to_num(mm_features_np, nan=0.0, posinf=1e6, neginf=-1e6)
                
            mm_scaler = StandardScaler()
            mm_features_np = mm_scaler.fit_transform(mm_features_np)
            
            # 将用户级多模态特征与个人特征融合
            combined_features = np.concatenate([personality_np, mm_features_np], axis=1)
            
            print(f"融合特征维度: {combined_features.shape}")
            clustering_features = combined_features
        else:
            print("仅使用个人特征进行聚类")
            clustering_features = personality_np
        
        # 聚类逻辑
        metrics = {'silhouette': [], 'dbi': [], 'ch': []}
        cluster_labels_dict = {}
        for k in range(2, 9):
            hierarchical = AgglomerativeClustering(n_clusters=k, linkage='ward')
            cluster_labels = hierarchical.fit_predict(clustering_features)
            metrics['silhouette'].append(silhouette_score(clustering_features, cluster_labels))
            metrics['dbi'].append(davies_bouldin_score(clustering_features, cluster_labels))
            metrics['ch'].append(calinski_harabasz_score(clustering_features, cluster_labels))
            cluster_labels_dict[k] = cluster_labels
        
        silhouette_vals = np.array(metrics['silhouette'])
        dbi_vals = np.array(metrics['dbi'])
        ch_vals = np.array(metrics['ch'])
        norm_silhouette = (silhouette_vals + 1) / 2
        norm_dbi = 1 - (dbi_vals - dbi_vals.min()) / (dbi_vals.max() - dbi_vals.min() + 1e-10)
        norm_ch = (ch_vals - ch_vals.min()) / (ch_vals.max() - ch_vals.min() + 1e-10)
        w_silhouette, w_dbi, w_ch = 0.4, 0.3, 0.3
        scores = w_silhouette * norm_silhouette + w_dbi * norm_dbi + w_ch * norm_ch
        best_k = np.argmax(scores) + 2
        best_cluster_labels = cluster_labels_dict[best_k]
        print(f"选择最佳 K={best_k}, 综合得分={scores[best_k - 2]:.4f}")
        
        self.cluster_labels_dict = {user_id: int(label) for user_id, label in zip(all_user_ids, best_cluster_labels)}
        self.cluster_id_to_indices = {k: np.where(best_cluster_labels == k)[0] for k in range(best_k)}
        self.actual_k = best_k
        print(f"实际使用的聚类数量: {self.actual_k}")
        
        # 存储聚类中心（基于原始个人特征，而非融合特征）
        self.cluster_centers = []
        for k in range(self.actual_k):
            cluster_indices = self.cluster_id_to_indices[k]
            if len(cluster_indices) > 0:
                cluster_features = personality_np[cluster_indices]
                center = np.mean(cluster_features, axis=0)
            else:
                center = np.zeros(personality_np.shape[1])
            self.cluster_centers.append(center)
        self.cluster_centers = torch.tensor(self.cluster_centers, dtype=torch.float32, device=device)
        
        # 调整p2k_models并移至正确设备
        if self.actual_k != self.K:
            self.p2k_models = nn.ModuleList([
                SimpleP2K(input_dim=5, hidden_dim=self.p2k_hidden, output_dim=self.p2k_out_dim).to(device)
                for _ in range(self.actual_k)
            ])
            
            # 更新软门控网络以匹配新的K
            self.soft_gate = SoftGatingNetwork(input_dim=5, hidden_dim=self.p2k_hidden, num_experts=self.actual_k).to(device)
            
            self.K = self.actual_k
            
            # 对新创建的P2K模型初始化权重
            for module in self.p2k_models:
                for name, p in module.named_parameters():
                    if p.dim() > 1:  # 权重矩阵
                        nn.init.xavier_uniform_(p)
                    elif 'bias' in name:
                        nn.init.zeros_(p)
        
        # 聚类分析输出
        print("\n=== 融合特征层次聚类分析 ===")
        total_users = 0
        for k in range(best_k):
            user_indices = self.cluster_id_to_indices[k]
            user_count = len(user_indices)
            total_users += user_count
            user_ids = [all_user_ids[idx] for idx in user_indices]
            print(f"聚类 {k}:")
            print(f"  用户数量: {user_count}")
            print(f"  用户 ID: {user_ids[:10]}{'...' if len(user_ids) > 10 else ''}")
        print(f"总用户数量: {total_users} (应等于数据集大小: {len(all_user_ids)})")
        
        # 保存融合特征对应的聚类结果供后续分析
        if user_multimodal_features is not None:
            self.fused_clustering_features = clustering_features
            self.fused_cluster_labels = best_cluster_labels
    
    def predict_cluster(self, personality_feat):
        """
        使用层次聚类预测人格特征所属的聚类
        
        Args:
            personality_feat: 人格特征 [B, 5]
            
        Returns:
            cluster_ids: 聚类ID [B]
            pos_in_clusters: 簇内位置 [B]（层次聚类无序，设为0）
            cluster_probs: 伪聚类概率 [B, actual_k]（硬分配为 one-hot）
        """
        if not self.has_personality or self.cluster_centers is None:
            batch_size = personality_feat.size(0)
            return np.zeros(batch_size, dtype=int), np.zeros(batch_size, dtype=int), None
        
        with torch.no_grad():
            # 检查输入是否包含NaN或Inf
            if torch.isnan(personality_feat).any() or torch.isinf(personality_feat).any():
                personality_feat = torch.nan_to_num(personality_feat, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # 标准化输入特征
            personality_std = (personality_feat - self.scaler_mean) / self.scaler_scale
            distances = torch.cdist(personality_std, self.cluster_centers)
            cluster_ids = torch.argmin(distances, dim=1)
            cluster_probs = F.one_hot(cluster_ids, num_classes=self.actual_k).float()
            pos_in_clusters = torch.zeros_like(cluster_ids)  # 层次聚类无簇内排序
            return cluster_ids.cpu().numpy(), pos_in_clusters.cpu().numpy(), cluster_probs
    
    def forward(self, personality_feat, visual_deep, visual_predesigned, audio_deep, audio_predesigned, 
            visual_mask=None, audio_mask=None, user_id=None, user_history_data=None):

        device = personality_feat.device
        batch_size = personality_feat.size(0)

        # 改为检测，不再自动替换nan/inf
        if torch.isnan(personality_feat).any() or torch.isinf(personality_feat).any():
            personality_feat = torch.nan_to_num(personality_feat, nan=0.0, posinf=1e6, neginf=-1e6)
            
        # 输入特征归一化
        visual_deep = self.normalize_features(visual_deep, 'visual_deep')
        visual_predesigned = self.normalize_features(visual_predesigned, 'visual_predesigned')
        audio_deep = self.normalize_features(audio_deep, 'audio_deep')
        audio_predesigned = self.normalize_features(audio_predesigned, 'audio_predesigned')
        
        # 检查归一化后的特征
        if visual_deep is not None:
            if torch.isnan(visual_deep).any() or torch.isinf(visual_deep).any():
                visual_deep = torch.nan_to_num(visual_deep, nan=0.0, posinf=1e6, neginf=-1e6)
        if visual_predesigned is not None:
            if torch.isnan(visual_predesigned).any() or torch.isinf(visual_predesigned).any():
                visual_predesigned = torch.nan_to_num(visual_predesigned, nan=0.0, posinf=1e6, neginf=-1e6)
        if audio_deep is not None:
            if torch.isnan(audio_deep).any() or torch.isinf(audio_deep).any():
                audio_deep = torch.nan_to_num(audio_deep, nan=0.0, posinf=1e6, neginf=-1e6)
        if audio_predesigned is not None:
            if torch.isnan(audio_predesigned).any() or torch.isinf(audio_predesigned).any():
                audio_predesigned = torch.nan_to_num(audio_predesigned, nan=0.0, posinf=1e6, neginf=-1e6)

        # 处理多模态输入
        if all(x is not None for x in [visual_deep, visual_predesigned, audio_deep, audio_predesigned]):
            hm, modal_weights = self.iafn(visual_deep, visual_predesigned, audio_deep, audio_predesigned)
            hm_mapped = self.iafn_to_gate(hm)

            if torch.isnan(hm_mapped).any() or torch.isinf(hm_mapped).any():
                hm_mapped = torch.nan_to_num(hm_mapped, nan=0.0, posinf=1e6, neginf=-1e6)
        else:
            hm = None
            hm_mapped = None

        # 处理人格特征 - 使用软门控替换硬聚类分配
        if not self.has_personality or self.cluster_centers is None:
            default_personality = self.default_personality_embedding.expand(batch_size, -1)
            hp_mapped = self.p2k_to_gate(default_personality)
            
            if torch.isnan(hp_mapped).any() or torch.isinf(hp_mapped).any():
                hp_mapped = torch.nan_to_num(hp_mapped, nan=0.0, posinf=1e6, neginf=-1e6)
        else:
            # 为每个样本的人格特征生成门控权重
            gate_weights = self.soft_gate(personality_feat)  # [B, K]
            
            if torch.isnan(gate_weights).any() or torch.isinf(gate_weights).any():
                gate_weights = torch.nan_to_num(gate_weights, nan=0.0, posinf=1e6, neginf=-1e6)
                # 确保权重和为1
                gate_weights = gate_weights / gate_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
            
            # 对每个样本运行所有P2K模型并加权组合
            hp_all_models = []
            for i in range(batch_size):
                p_feat = personality_feat[i:i+1]  # [1, 5]
                model_outputs = []
                
                for k in range(self.K):
                    model_output = self.p2k_models[k](p_feat)  # [1, p2k_out_dim]
                    model_outputs.append(model_output)
                
                # 堆叠所有模型输出
                stacked_outputs = torch.cat(model_outputs, dim=0)  # [K, p2k_out_dim]
                
                # 使用门控权重对输出进行加权
                weighted_sum = torch.mm(
                    gate_weights[i:i+1],  # [1, K]
                    stacked_outputs  # [K, p2k_out_dim]
                )  # [1, p2k_out_dim]
                
                hp_all_models.append(weighted_sum)
            
            # 组合所有样本的结果
            hpTrans_all = torch.cat(hp_all_models, dim=0)  # [B, p2k_out_dim]
            
            if torch.isnan(hpTrans_all).any() or torch.isinf(hpTrans_all).any():
                hpTrans_all = torch.nan_to_num(hpTrans_all, nan=0.0, posinf=1e6, neginf=-1e6)
            
            hp_mapped = self.p2k_to_gate(hpTrans_all)
            
            if torch.isnan(hp_mapped).any() or torch.isinf(hp_mapped).any():
                hp_mapped = torch.nan_to_num(hp_mapped, nan=0.0, posinf=1e6, neginf=-1e6)

        # 处理用户级特征 - 直接在forward中提取
        hu_mapped = torch.zeros_like(hp_mapped)
        if user_id is not None and user_history_data is not None:
            # 遍历批次中的每个样本
            for i in range(batch_size):
                uid = user_id[i] if isinstance(user_id, list) else user_id
                # 获取当前用户的历史数据
                user_data = user_history_data.get(uid, {})
                if user_data:
                    # 提取当前用户的多模态特征序列
                    user_features = []
                    for sample_idx in range(len(user_data.get('visual_deep', []))):
                        # 提取每个历史样本的特征
                        vd = user_data['visual_deep'][sample_idx].to(device)
                        vp = user_data['visual_predesigned'][sample_idx].to(device)
                        ad = user_data['audio_deep'][sample_idx].to(device)
                        ap = user_data['audio_predesigned'][sample_idx].to(device)
                        
                        # 数值稳定性检查
                        vd = torch.nan_to_num(vd, nan=0.0, posinf=1e6, neginf=-1e6)
                        vp = torch.nan_to_num(vp, nan=0.0, posinf=1e6, neginf=-1e6)
                        ad = torch.nan_to_num(ad, nan=0.0, posinf=1e6, neginf=-1e6)
                        ap = torch.nan_to_num(ap, nan=0.0, posinf=1e6, neginf=-1e6)
                        
                        # 提取多模态特征
                        with torch.no_grad():
                            hist_feat, _ = self.iafn(vd.unsqueeze(0), vp.unsqueeze(0), 
                                                   ad.unsqueeze(0), ap.unsqueeze(0))
                        user_features.append(hist_feat)
                    
                    if user_features:
                        # 将用户历史特征堆叠为序列
                        sequence = torch.cat(user_features, dim=0)  # [num_samples, iafn_embed_dim]
                        
                        # 检查序列数值稳定性
                        sequence = torch.nan_to_num(sequence, nan=0.0, posinf=1e6, neginf=-1e6)
                        
                        # 如果历史样本足够多，使用LSTM处理
                        if sequence.size(0) >= 1:
                            sequence = sequence.unsqueeze(0)  # [1, num_samples, iafn_embed_dim]
                            
                            with torch.no_grad():
                                lstm_out, (h_n, c_n) = self.user_sequence_lstm(sequence)
                                
                            # 合并双向LSTM的最终状态
                            forward = h_n[0]  # [1, iafn_embed_dim]
                            backward = h_n[1]  # [1, iafn_embed_dim]
                            user_feat = torch.cat([forward, backward], dim=1)  # [1, 2*iafn_embed_dim]
                            
                            # 投影回原始维度
                            user_feat = self.lstm_projection(user_feat)  # [1, iafn_embed_dim]
                        else:
                            # 样本太少，使用平均池化
                            user_feat = sequence.mean(dim=0, keepdim=True)  # [1, iafn_embed_dim]
                        
                        # 数值稳定性检查
                        user_feat = torch.nan_to_num(user_feat, nan=0.0, posinf=1e6, neginf=-1e6)
                        
                        # 映射到门控表示空间
                        hu_i = self.user_feat_proj(user_feat)  # [1, gate_embed_dim]
                        hu_mapped[i:i+1] = hu_i
        
        # 检查hu_mapped的数值稳定性
        if torch.isnan(hu_mapped).any() or torch.isinf(hu_mapped).any():
            hu_mapped = torch.nan_to_num(hu_mapped, nan=0.0, posinf=1e6, neginf=-1e6)
        fusion_mode='3way'
         # 根据fusion_mode选择融合方式
        if fusion_mode == '3way':
            assert hp_mapped is not None and hm_mapped is not None and hu_mapped is not None, \
                "3way模式需要人格、多模态和用户特征"
            h_fusion = self.three_way_fusion(hp_mapped, hm_mapped, hu_mapped)

        elif fusion_mode == 'av+feat':
            assert hp_mapped is not None and hm_mapped is not None, "av+feat模式需要人格和多模态"
            # 复用你之前的gate_fusion模块融合两路
            h_fusion = self.gate_fusion(hp_mapped, hm_mapped)

        elif fusion_mode == 'user+feat':
            assert hp_mapped is not None and hu_mapped is not None, "user+feat模式需要人格和用户历史"
            # 简单拼接+MLP（需要你定义一个两路融合层或者用以下示范）
            h_fusion = self.gate_fusion(hp_mapped, hu_mapped)
    

        elif fusion_mode == 'av+user':
            assert hm_mapped is not None and hu_mapped is not None, "av+user模式需要多模态和用户历史"
            h_fusion = torch.cat([hm_mapped, hu_mapped], dim=1)
            h_fusion = self.two_way_fusion_mlp(h_fusion)

        elif fusion_mode == 'av':
            assert hm_mapped is not None, "av模式需要多模态特征"
            h_fusion = self.multimodal_only_fc(hm_mapped)

        elif fusion_mode == 'user':
            assert hu_mapped is not None, "user模式需要用户历史特征"
            h_fusion = self.user_feat_proj(hu_mapped)

        elif fusion_mode == 'feat':
            assert hp_mapped is not None, "feat模式需要人格特征"
            h_fusion = self.p2k_to_gate(hp_mapped)

        else:
            raise ValueError(f"未知fusion_mode: {fusion_mode}")
        
        # 检查融合表示的数值稳定性
        if torch.isnan(h_fusion).any() or torch.isinf(h_fusion).any():
            h_fusion = torch.nan_to_num(h_fusion, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 将融合表示通过分类器获得输出
        logits = self.classifier(h_fusion)
        
        return logits, h_fusion  # 返回logits、融合表示
    
    def predict_proba(self, *args, **kwargs):
        """预测概率函数"""
        logits, _, _ = self.forward(*args, **kwargs)
        if logits.size(1) > 1:
            # 多类分类
            probs = F.softmax(logits, dim=1)
        else:
            # 二分类
            probs = torch.sigmoid(logits)
        return probs
    
    def extract_multiview_embeddings(self, personality_feat, visual_deep, visual_predesigned, audio_deep, audio_predesigned):
        """提取多视角嵌入表示"""
        # 输入特征归一化
        visual_deep = self.normalize_features(visual_deep, 'visual_deep')
        visual_predesigned = self.normalize_features(visual_predesigned, 'visual_predesigned')
        audio_deep = self.normalize_features(audio_deep, 'audio_deep')
        audio_predesigned = self.normalize_features(audio_predesigned, 'audio_predesigned')
        
        # 多模态特征提取
        hm, _ = self.iafn(visual_deep, visual_predesigned, audio_deep, audio_predesigned)
        
        # 人格特征映射
        if not self.has_personality or self.cluster_centers is None:
            batch_size = personality_feat.size(0)
            hp = self.default_personality_embedding.expand(batch_size, -1)
        else:
            # 软门控方式处理人格特征
            batch_size = personality_feat.size(0)
            gate_weights = self.soft_gate(personality_feat)  # [B, K]
            
            if torch.isnan(gate_weights).any() or torch.isinf(gate_weights).any():
                gate_weights = torch.nan_to_num(gate_weights, nan=0.0, posinf=1e6, neginf=-1e6)
                gate_weights = gate_weights / gate_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
            
            # 对每个样本运行所有P2K模型并加权组合
            hp_all_models = []
            for i in range(batch_size):
                p_feat = personality_feat[i:i+1]  # [1, 5]
                model_outputs = []
                
                for k in range(self.K):
                    model_output = self.p2k_models[k](p_feat)  # [1, p2k_out_dim]
                    model_outputs.append(model_output)
                
                # 堆叠所有模型输出
                stacked_outputs = torch.cat(model_outputs, dim=0)  # [K, p2k_out_dim]
                
                # 使用门控权重对输出进行加权
                weighted_sum = torch.mm(
                    gate_weights[i:i+1],  # [1, K]
                    stacked_outputs  # [K, p2k_out_dim]
                )  # [1, p2k_out_dim]
                
                hp_all_models.append(weighted_sum)
            
            # 组合所有样本的结果
            hp = torch.cat(hp_all_models, dim=0)  # [B, p2k_out_dim]
        
        # 检查数值稳定性
        if torch.isnan(hp).any() or torch.isinf(hp).any():
            hp = torch.nan_to_num(hp, nan=0.0, posinf=1e6, neginf=-1e6)
        if torch.isnan(hm).any() or torch.isinf(hm).any():
            hm = torch.nan_to_num(hm, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 使用多视角Transformer融合特征
        multi_view_emb = self.mv_transformer(hp, hm)
        
        return multi_view_emb
