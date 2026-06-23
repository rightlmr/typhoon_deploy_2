from pathlib import Path
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ABS_PATH = Path(__file__).resolve().parent.parent




# 空间裁剪参数（度数）- WP区域
CROP_LAT_MIN = -13.5
CROP_LAT_MAX = 82.25     
CROP_LON_MIN = 92.25
CROP_LON_MAX = 188.0


# 台风存在类型
TYPHOON_NONE = 0    # 无台风
TYPHOON_SINGLE = 1  # 单台风
TYPHOON_MULTI = 2   # 多台风

# AIFS变量顺序（与NetCDF文件中的channel顺序一致）
VAR_ORDER = [
    'u10', 'v10', 'mslp', 't2',      # surface
    'u850', 'v850', 'q850', 't850',  # 850hPa
    'u700', 'v700', 'q700', 't700',  # 700hPa
    'u500', 'v500', 'q500', 't500',  # 500hPa
]

# 预报时效列表（0-240小时，步长6小时）
FORECAST_HOURS = list(range(0, 241, 6))

# 物理特征层次配置（用于计算涡度/散度/风切变）
PHYSICS_LEVEL_CONFIG = {
    'surface': {'u_idx': 0, 'v_idx': 1, 'level': 'surface'},
    '850hPa': {'u_idx': 4, 'v_idx': 5, 'level': '850hPa'},
    '700hPa': {'u_idx': 8, 'v_idx': 9, 'level': '700hPa'},
    '500hPa': {'u_idx': 12, 'v_idx': 13, 'level': '500hPa'},
}