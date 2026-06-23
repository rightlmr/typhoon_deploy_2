"""
台风强度订正模型推理模块

包含：
  1. 模型定义 (ForecastHourEmbedding, HybridTemporalFusion, TyphoonIntensityCorrectionModelHistory)
  2. 数据预处理 (normalize_intensity, denormalize_intensity)
  3. 输入构建 (build_correction_input)
  4. 推理封装 (TyphoonCorrectionInference)

原训练框架中多余的 Lightning/Dataset/DataLoader 代码已剥离，仅保留推理必需的逻辑。
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------
# 注入 chaosbench 库路径（部署包内部 libs/）
# -------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEPLOY_ROOT = _SCRIPT_DIR.parent
_CORRECTION_LIB_DIR = _DEPLOY_ROOT / "libs"
if str(_CORRECTION_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_CORRECTION_LIB_DIR))

from chaosbench.models.encoder.fastervit.faster_vit import FasterViT

from aifs_data_utils import (
    AIFS_VAR_ORDER,
    crop_aifs_field_from_grib,
    extract_aifs_intensity,
    normalize_aifs_tensor,
    load_aifs_normalization_stats,
    latlon_to_grid_index,
)


# -------------------------------------------------
# IBTrACS 统计加载（用于强度标准化）
# -------------------------------------------------
def load_ibtracs_stats(stats_file: str) -> Dict:
    """加载 IBTrACS 标准化统计参数"""
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"IBTrACS stats file not found: {stats_file}")
    with open(stats_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_intensity(intensity: np.ndarray, stats: Dict) -> np.ndarray:
    """
    使用 IBTrACS 统计量对强度进行标准化 (log1p + Z-score)

    Args:
        intensity: [2] 原始强度 [max_wind (knots), min_pres (hPa)]
        stats: IBTrACS 统计参数字典

    Returns:
        normalized: [2] 标准化后的强度
    """
    pres_baseline = stats.get('pres_baseline', 1040)
    target_vars = ['USA_WIND', 'USA_PRES']
    normalized = np.zeros_like(intensity, dtype=np.float32)

    for i, var_name in enumerate(target_vars):
        value = intensity[i]
        transformer_cfg = stats['transformers'][var_name]

        if transformer_cfg.get('is_pressure_deficit', False):
            value = pres_baseline - value

        transformed = np.log1p(np.maximum(value, 0.0))
        mean = stats['stats'][var_name]['transformed_mean']
        std = stats['stats'][var_name]['transformed_std']
        normalized[i] = (transformed - mean) / std

    return normalized


def denormalize_intensity(normalized: np.ndarray, stats: Dict) -> np.ndarray:
    """
    反标准化强度到原始物理量纲

    Args:
        normalized: [..., 2] 标准化后的强度
        stats: IBTrACS 统计参数字典

    Returns:
        original: [..., 2] 原始强度
    """
    pres_baseline = stats.get('pres_baseline', 1040)
    target_vars = ['USA_WIND', 'USA_PRES']
    result = np.zeros_like(normalized)

    for i, var_name in enumerate(target_vars):
        cfg = stats['transformers'][var_name]
        mean = stats['stats'][var_name]['transformed_mean']
        std = stats['stats'][var_name]['transformed_std']

        transformed = normalized[..., i] * std + mean
        original = np.expm1(transformed)

        if cfg.get('is_pressure_deficit', False):
            original = pres_baseline - original

        result[..., i] = original

    return result


# ==============================================================================
# 模型定义（从 model_correction_historytimestamps.py 迁移，去除训练相关代码）
# ==============================================================================

class ForecastHourEmbedding(nn.Module):
    """
    预报时效嵌入层：将标量预报时效转换为特征向量
    使用可学习嵌入 (nn.Embedding)，适用于固定离散预报时效
    """
    def __init__(self, embed_dim=16, step=6, max_hour=240):
        super().__init__()
        self.embed_dim = embed_dim
        self.step = step
        self.max_hour = max_hour
        self.num_embeddings = max_hour // step + 1
        self.embedding = nn.Embedding(self.num_embeddings, embed_dim)

    def hour_to_index(self, forecast_hour):
        index = (forecast_hour / self.step).long()
        index = torch.clamp(index, 0, self.num_embeddings - 1)
        return index

    def forward(self, forecast_hour):
        if forecast_hour.dim() == 0:
            forecast_hour = forecast_hour.unsqueeze(0)
        index = self.hour_to_index(forecast_hour)
        embed = self.embedding(index)
        return embed


class HybridTemporalFusion(nn.Module):
    """
    混合时序融合模块 (空间保持 + 非对称 Cross-Attention + 门控残差)
    """
    def __init__(self, feature_dim, num_timesteps):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_timesteps = num_timesteps
        self.num_history = num_timesteps - 1

        self.history_fusion = nn.Sequential(
            nn.Conv2d(self.num_history * feature_dim, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
        )

        self.num_heads = 8
        assert feature_dim % self.num_heads == 0
        head_dim = feature_dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.kv_proj = nn.Conv2d(feature_dim, feature_dim * 2, kernel_size=1)
        self.out_proj = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.Sigmoid(),
        )

        self.norm = nn.GroupNorm(num_groups=8, num_channels=feature_dim)

    def forward(self, x):
        B, T, C, H, W = x.shape
        f_cur = x[:, -1]
        f_hist = x[:, :-1]

        f_hist = f_hist.reshape(B, (T - 1) * C, H, W)
        f_hist = self.history_fusion(f_hist)

        q = self.q_proj(f_cur)
        kv = self.kv_proj(f_hist)
        k, v = kv.chunk(2, dim=1)

        q = q.view(B, C, H * W).permute(0, 2, 1)
        k = k.view(B, C, H * W).permute(0, 2, 1)
        v = v.view(B, C, H * W).permute(0, 2, 1)

        q = q.reshape(B, H * W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, H * W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, H * W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(B, H * W, C)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        out = self.out_proj(out)

        gate = self.gate(torch.cat([f_cur, out], dim=1))
        out = self.norm(f_cur + gate * out)

        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return out


class TyphoonIntensityCorrectionModelHistory(nn.Module):
    """
    台风强度订正模型 - 历史时间步融合版本（推理专用，已去除训练逻辑）
    """

    def __init__(self, model_args: dict, data_args: dict):
        super().__init__()
        self.model_args = model_args
        self.data_args = data_args

        self.history_timesteps = data_args.get('history_timesteps', [0])
        self.num_history = len(self.history_timesteps)
        self.num_timesteps = self.num_history + 1

        self.share_weights = model_args.get('share_weights', False)
        self.use_forecast_hour = data_args.get('use_forecast_hour', False)
        self.forecast_hour_embed_dim = model_args.get('forecast_hour_embed_dim', 4)
        self.dataset_type = str(data_args.get('dataset_type', 'aifs')).lower()

        self.use_late_fh_embed = (self.dataset_type == 'aifs' and self.use_forecast_hour)

        if self.use_forecast_hour:
            self.forecast_hour_step = data_args.get('forecast_hour_step', 6)
            self.forecast_hour_max = data_args.get('forecast_hour_max', 240)
            self.forecast_hour_embedding = ForecastHourEmbedding(
                embed_dim=self.forecast_hour_embed_dim,
                step=self.forecast_hour_step,
                max_hour=self.forecast_hour_max,
            )
            num_hours = self.forecast_hour_max // self.forecast_hour_step + 1
            if self.use_late_fh_embed:
                extra_in_chans = 0
                print(f"Using LATE forecast_hour embedding: fh_embed_dim={self.forecast_hour_embed_dim}")
            else:
                extra_in_chans = self.forecast_hour_embed_dim
                print(f"Using EARLY forecast_hour embedding (dim={self.forecast_hour_embed_dim}, "
                      f"hours={num_hours}, step={self.forecast_hour_step})")
        else:
            self.forecast_hour_embedding = None
            self.forecast_hour_step = 6
            self.forecast_hour_max = 240
            extra_in_chans = 0
            print("Not using forecast_hour")

        dataset_type = str(data_args.get('dataset_type', 'aifs')).lower()
        if dataset_type == 'aifs':
            default_aifs_vars = ['u10', 'v10', 'mslp', 't2', 'u850', 'v850', 'q850', 't850',
                                 'u700', 'v700', 'q700', 't700', 'u500', 'v500', 'q500', 't500']
            base_in_chans = len(data_args.get('aifs_vars', default_aifs_vars))
        elif dataset_type == 'era5':
            base_in_chans = len(data_args.get('era5_vars', ['u10', 'v10', 'mslp', 't2']))
        else:
            raise ValueError(f"Unknown dataset_type '{dataset_type}', expected 'aifs' or 'era5'")

        total_in_chans = base_in_chans + extra_in_chans

        self.feature_dim = model_args.get('dim', 64) * (2 ** (len(model_args.get('depths', [2, 2, 6, 2])) - 1))
        print(f"Calculated feature_dim={self.feature_dim} based on FasterViT configuration")

        if self.share_weights:
            print(f"Using shared weights for {self.num_timesteps} timesteps")
            self.backbone = self._create_backbone(total_in_chans)
            self.backbones = None
        else:
            print(f"Using independent weights for {self.num_timesteps} timesteps")
            self.backbones = nn.ModuleList([
                self._create_backbone(total_in_chans)
                for _ in range(self.num_timesteps)
            ])
            self.backbone = None

        self.temporal_fusion = HybridTemporalFusion(
            feature_dim=self.feature_dim,
            num_timesteps=self.num_timesteps,
        )

        if self.use_late_fh_embed:
            self.fh_projection = nn.Linear(self.forecast_hour_embed_dim, self.feature_dim)
            print(f"  Added fh_projection: {self.forecast_hour_embed_dim} -> {self.feature_dim}")
        else:
            self.fh_projection = None

        hidden_dim = model_args.get('hidden_dim', 256)
        dropout = model_args.get('dropout', 0.3)
        self.wind_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.pres_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        print(f"TyphoonIntensityCorrectionModelHistory initialized")
        print(f"  History timesteps: {self.history_timesteps}")
        print(f"  Total timesteps: {self.num_timesteps}")
        print(f"  Share weights: {self.share_weights}")
        print(f"  Feature dim: {self.feature_dim}")

    def _create_backbone(self, in_chans):
        return FasterViT(
            dim=self.model_args.get('dim', 64),
            in_dim=self.model_args.get('in_dim', 64),
            depths=self.model_args.get('depths', [2, 2, 6, 2]),
            window_size=self.model_args.get('window_size', [2, 2, 2, 2]),
            ct_size=self.model_args.get('ct_size', 2),
            mlp_ratio=self.model_args.get('mlp_ratio', 4.),
            num_heads=self.model_args.get('num_heads', [4, 8, 16, 32]),
            resolution=self.model_args.get('img_size', 32),
            drop_path_rate=self.model_args.get('drop_path_rate', 0.1),
            in_chans=in_chans,
            num_classes=0,
            qkv_bias=self.model_args.get('qkv_bias', True),
            qk_scale=self.model_args.get('qk_scale', None),
            drop_rate=self.model_args.get('drop_rate', 0.1),
            attn_drop_rate=self.model_args.get('attn_drop_rate', 0.1),
            layer_scale=self.model_args.get('layer_scale', None),
            layer_scale_conv=self.model_args.get('layer_scale_conv', None),
            layer_norm_last=self.model_args.get('layer_norm_last', False),
            hat=self.model_args.get('hat', [True, True, True, False]),
            do_propagation=self.model_args.get('do_propagation', True),
        )

    def _prepare_forecast_hour(self, forecast_hour, batch_size: int, device: torch.device):
        if not self.use_forecast_hour or forecast_hour is None:
            return None
        if isinstance(forecast_hour, int):
            forecast_hour = torch.tensor(
                [forecast_hour] * batch_size,
                device=device,
                dtype=torch.float32
            )
        elif isinstance(forecast_hour, torch.Tensor):
            forecast_hour = forecast_hour.to(device=device, dtype=torch.float32)
            if forecast_hour.dim() == 0:
                forecast_hour = forecast_hour.unsqueeze(0).expand(batch_size)
            elif forecast_hour.size(0) != batch_size:
                forecast_hour = forecast_hour.expand(batch_size)
        return forecast_hour

    def _embed_forecast_hour(self, x, forecast_hour):
        if self.use_forecast_hour and forecast_hour is not None:
            hour_embed = self.forecast_hour_embedding(forecast_hour)
            B, _, H, W = x.shape
            hour_feat = hour_embed.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)
            x = torch.cat([x, hour_feat], dim=1)
        return x

    def extract_features(self, x, forecast_hour=None):
        x = self._embed_forecast_hour(x, forecast_hour)
        if self.share_weights:
            features = self.backbone.forward_features(x)
        else:
            raise NotImplementedError("Use extract_all_features for non-shared weights")
        features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return features

    def extract_all_features(self, current_field, history_fields=None, forecast_hour=None, history_forecast_hours=None):
        all_fields = [current_field]
        if history_fields is not None and len(history_fields) > 0:
            all_fields.extend(history_fields)

        all_hours = [forecast_hour]
        if history_forecast_hours is not None and len(history_forecast_hours) > 0:
            all_hours.extend(history_forecast_hours)

        all_features = []
        for i, field in enumerate(all_fields):
            if not self.use_late_fh_embed:
                hour = all_hours[i] if i < len(all_hours) else None
                field = self._embed_forecast_hour(field, hour)
            if self.share_weights:
                feat = self.backbone.forward_features(field)
            else:
                feat = self.backbones[i].forward_features(field)
            all_features.append(feat)

        all_features = torch.stack(all_features, dim=0).permute(1, 0, 2, 3, 4)
        return all_features

    def forward(self, current_field, aifs_intensity, history_fields=None, history_intensities=None,
                forecast_hour=None, history_forecast_hours=None, return_delta=False, return_both=False):
        all_features = self.extract_all_features(current_field, history_fields, forecast_hour, history_forecast_hours)
        fused_feature = self.temporal_fusion(all_features)

        if self.use_late_fh_embed and forecast_hour is not None:
            fh_embed = self.forecast_hour_embedding(forecast_hour)
            fh_proj = self.fh_projection(fh_embed)
            fused_feature = fused_feature + fh_proj

        wind_delta = self.wind_head(fused_feature)
        pres_delta = self.pres_head(fused_feature)
        pred_delta = torch.cat([wind_delta, pres_delta], dim=1)
        pred_obs = aifs_intensity + pred_delta

        if return_both:
            return pred_obs, pred_delta
        if return_delta:
            return pred_delta
        return pred_obs


# -------------------------------------------------
# 模型加载
# -------------------------------------------------
def load_correction_model(checkpoint_path: str,
                          model_args: Dict,
                          data_args: Dict,
                          device: str = "auto") -> nn.Module:
    """
    加载台风强度订正模型

    Args:
        checkpoint_path: checkpoint 文件路径（.ckpt 或 .pt）
        model_args: 模型参数字典
        data_args: 数据参数字典
        device: 计算设备

    Returns:
        加载好的模型
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TyphoonIntensityCorrectionModelHistory(
        model_args=model_args,
        data_args=data_args,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"[Correction] Model loaded from {checkpoint_path}")
    print(f"[Correction] Device: {device}")
    return model


# -------------------------------------------------
# 单样本推理输入构建
# -------------------------------------------------
def build_correction_input(grib_path: str,
                           forecast_hour: int,
                           lat: float,
                           lon: float,
                           aifs_stats: Dict,
                           ibtracs_stats: Dict,
                           var_order: List[str] = None,
                           field_size: int = 64,
                           history_info: Optional[List[Dict]] = None) -> Dict[str, torch.Tensor]:
    """
    为强度订正模型构建单个样本的输入字典

    Args:
        grib_path: AIFS GRIB2 文件路径
        forecast_hour: 当前预报时效
        lat, lon: 台风中心经纬度
        aifs_stats: AIFS 标准化统计参数
        ibtracs_stats: IBTrACS 标准化统计参数
        var_order: AIFS 变量顺序，None 则使用全部 16 个变量
        field_size: 裁剪区域大小
        history_info: 历史时间步信息列表，每个元素为 dict:
            {'grib_path': str, 'forecast_hour': int, 'lat': float, 'lon': float}
            如果为 None，则不使用历史时间步

    Returns:
        输入字典，可直接传入 model.forward():
            {
                'current_field': [1, C, H, W],
                'aifs_intensity': [1, 2],
                'forecast_hour': [1],
                'history_fields': List[[1, C, H, W]] (可选),
                'history_intensities': List[[1, 2]] (可选),
                'history_forecast_hours': List[int] (可选),
            }
    """
    if var_order is None:
        var_order = AIFS_VAR_ORDER

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载当前时刻的场数据
    current_data = crop_aifs_field_from_grib(
        grib_path, lat, lon, field_size, var_order
    )
    if current_data is None:
        raise RuntimeError(f"Failed to load AIFS field from {grib_path} fh={forecast_hour}")

    # 2. 提取当前 AIFS 强度（反标准化前）
    current_intensity_raw = np.array(extract_aifs_intensity(current_data, var_order), dtype=np.float32)
    current_intensity_norm = normalize_intensity(current_intensity_raw, ibtracs_stats)
    print(f"  [CORR] AIFS raw: wind={current_intensity_raw[0]:.1f}kt, pres={current_intensity_raw[1]:.1f}hPa")

    # 3. 标准化场数据
    current_tensor = torch.from_numpy(current_data)
    current_tensor = normalize_aifs_tensor(current_tensor, var_order, aifs_stats)

    # 4. 构建输入字典
    result = {
        'current_field': current_tensor.unsqueeze(0).to(device),  # [1, C, H, W]
        'aifs_intensity': torch.from_numpy(current_intensity_norm).unsqueeze(0).to(device),  # [1, 2]
        'forecast_hour': torch.tensor([forecast_hour], dtype=torch.float32, device=device),
    }

    # 5. 加载历史时间步（如果提供）
    if history_info is not None and len(history_info) > 0:
        history_fields = []
        history_intensities = []
        history_forecast_hours = []

        for hist in history_info:
            hist_data = crop_aifs_field_from_grib(
                hist['grib_path'],
                hist['lat'], hist['lon'], field_size, var_order
            )
            if hist_data is None:
                raise RuntimeError(f"Failed to load history field from {hist['grib_path']}")

            hist_intensity_raw = np.array(extract_aifs_intensity(hist_data, var_order), dtype=np.float32)
            hist_intensity_norm = normalize_intensity(hist_intensity_raw, ibtracs_stats)

            hist_tensor = torch.from_numpy(hist_data)
            hist_tensor = normalize_aifs_tensor(hist_tensor, var_order, aifs_stats)

            history_fields.append(hist_tensor.unsqueeze(0).to(device))
            history_intensities.append(torch.from_numpy(hist_intensity_norm).unsqueeze(0).to(device))
            history_forecast_hours.append(hist['forecast_hour'])

        result['history_fields'] = history_fields
        result['history_intensities'] = history_intensities
        result['history_forecast_hours'] = history_forecast_hours

    return result


# -------------------------------------------------
# 主推理类
# -------------------------------------------------
class TyphoonCorrectionInference:
    """
    台风强度订正推理封装类

    Usage:
        infer = TyphoonCorrectionInference(
            checkpoint_path=".../checkpoints/last.ckpt",
            model_args={...},
            data_args={...},
            aifs_stats_file=".../normalization_stats.json",
            ibtracs_stats_file=".../ibtracs_stats_correction_log1p.json",
        )
        sample_input = build_correction_input(
            grib_path=".../AIFS_2024_09_01_12_FCST_054h.grib2",
            forecast_hour=54, lat=25.0, lon=130.0,
            aifs_stats=infer.aifs_stats,
            ibtracs_stats=infer.ibtracs_stats,
        )
        pred_obs = infer.predict_single(sample_input)
        # pred_obs: [1, 2] tensor -> denormalize -> [wind (kt), pres (hPa)]
    """

    def __init__(self,
                 checkpoint_path: str,
                 model_args: Dict,
                 data_args: Dict,
                 aifs_stats_file: str,
                 ibtracs_stats_file: str,
                 device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # 加载模型
        self.model = load_correction_model(checkpoint_path, model_args, data_args, device)
        self.model_args = model_args
        self.data_args = data_args

        # 加载统计参数
        self.aifs_stats = load_aifs_normalization_stats(aifs_stats_file)
        self.ibtracs_stats = load_ibtracs_stats(ibtracs_stats_file)

        # 变量顺序
        dataset_type = str(data_args.get('dataset_type', 'aifs')).lower()
        if dataset_type == 'aifs':
            default_vars = ['u10', 'v10', 'mslp', 't2', 'u850', 'v850', 'q850', 't850',
                            'u700', 'v700', 'q700', 't700', 'u500', 'v500', 'q500', 't500']
            self.var_order = data_args.get('aifs_vars', default_vars)
        else:
            self.var_order = data_args.get('era5_vars', ['u10', 'v10', 'mslp', 't2'])

        self.field_size = data_args.get('field_size', 64)
        self.use_forecast_hour = data_args.get('use_forecast_hour', False)
        self.history_timesteps = data_args.get('history_timesteps', None)
        self.use_history = self.history_timesteps is not None and len(self.history_timesteps) > 0

        print(f"[Correction] Var order: {self.var_order}")
        print(f"[Correction] Field size: {self.field_size}")
        print(f"[Correction] Use forecast_hour: {self.use_forecast_hour}")
        print(f"[Correction] Use history: {self.use_history}")
        if self.use_history:
            print(f"[Correction] History timesteps: {self.history_timesteps}")

    def predict_single(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        对单个样本执行推理

        Args:
            inputs: 由 build_correction_input() 构建的输入字典

        Returns:
            pred_obs: [1, 2] 预测观测强度（标准化空间）
        """
        with torch.no_grad():
            pred_obs = self.model(
                current_field=inputs['current_field'],
                aifs_intensity=inputs['aifs_intensity'],
                history_fields=inputs.get('history_fields', None),
                history_intensities=inputs.get('history_intensities', None),
                forecast_hour=inputs.get('forecast_hour', None) if self.use_forecast_hour else None,
                history_forecast_hours=inputs.get('history_forecast_hours', None),
                return_delta=False,
            )
        return pred_obs

    def denormalize_prediction(self, pred_normalized: torch.Tensor) -> np.ndarray:
        """
        将标准化的预测结果反标准化为原始物理量纲

        Args:
            pred_normalized: [..., 2] 标准化预测值

        Returns:
            original: [..., 2] 原始值 [wind (kt), pres (hPa)]
        """
        pred_np = pred_normalized.cpu().numpy()
        return denormalize_intensity(pred_np, self.ibtracs_stats)

    def predict_and_denormalize(self, inputs: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        执行推理并反标准化

        Returns:
            [1, 2] numpy 数组 [wind (kt), pres (hPa)]
        """
        pred_norm = self.predict_single(inputs)
        pred_np = self.denormalize_prediction(pred_norm)

        # 计算修正量 (delta = predicted - AIFS raw)
        aifs_norm = inputs['aifs_intensity'].cpu().numpy()
        aifs_raw = denormalize_intensity(aifs_norm, self.ibtracs_stats)
        delta_wind = pred_np[0, 0] - aifs_raw[0, 0]
        delta_pres = pred_np[0, 1] - aifs_raw[0, 1]
        print(f"  [CORR] Predicted: wind={pred_np[0,0]:.1f}kt ({delta_wind:+.1f}), pres={pred_np[0,1]:.1f}hPa ({delta_pres:+.1f})")

        return pred_np
