"""
Tropical Cyclone Detection library — 推理精简版

仅保留 VGG_V3 模型和 StandardScaler，用于台风中心点检测推理。
训练相关的模块（dataset、trainer、tester、callbacks 等）已全部移除。
"""

from . import models
from . import scaling

__all__ = ['models', 'scaling']
