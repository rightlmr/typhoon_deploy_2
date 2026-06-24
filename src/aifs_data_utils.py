"""
AIFS 数据共享工具模块（GRIB2 版本）

为台风中心点预测和强度订正模型提供统一的数据加载、预处理和裁剪功能。
数据源：AIFS GRIB2 单时效文件（pygrib 读取）
"""

import os
import re
import json
import torch
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


# -------------------------------------------------
# AIFS 全局网格参数
# -------------------------------------------------
AIFS_RESOLUTION = 0.25  # 度
AIFS_LAT_MIN = -90.0
AIFS_LAT_MAX = 90.0
AIFS_LON_MIN = 0.0
AIFS_LON_MAX = 360.0
AIFS_HEIGHT = 721
AIFS_WIDTH = 1440

LATS_GLOBAL = np.linspace(90, -90, 721)
LONS_GLOBAL = np.linspace(0, 359.75, 1440)

# -------------------------------------------------
# GRIB2 变量映射（shortName + typeOfLevel + level）
# -------------------------------------------------
# 格式: (shortName, typeOfLevel, level_value)
GRIB_VAR_MAP = {
    'u10':    ('10u', 'heightAboveGround', 10),
    'v10':    ('10v', 'heightAboveGround', 10),
    'mslp':   ('msl', 'meanSea', 0),
    't2':     ('2t',  'heightAboveGround', 2),
    'u850':   ('u',   'isobaricInhPa', 850),
    'v850':   ('v',   'isobaricInhPa', 850),
    'q850':   ('q',   'isobaricInhPa', 850),
    't850':   ('t',   'isobaricInhPa', 850),
    'u700':   ('u',   'isobaricInhPa', 700),
    'v700':   ('v',   'isobaricInhPa', 700),
    'q700':   ('q',   'isobaricInhPa', 700),
    't700':   ('t',   'isobaricInhPa', 700),
    'u500':   ('u',   'isobaricInhPa', 500),
    'v500':   ('v',   'isobaricInhPa', 500),
    'q500':   ('q',   'isobaricInhPa', 500),
    't500':   ('t',   'isobaricInhPa', 500),
}

# -------------------------------------------------
# AIFS 变量顺序（与模型输入 channel 顺序一致）
# -------------------------------------------------
AIFS_VAR_ORDER = [
    'u10', 'v10', 'mslp', 't2',      # surface
    'u850', 'v850', 'q850', 't850',  # 850hPa
    'u700', 'v700', 'q700', 't700',  # 700hPa
    'u500', 'v500', 'q500', 't500',  # 500hPa
]

# -------------------------------------------------
# 2-vars 检测模型专用变量
# -------------------------------------------------
DETECTION_VAR_ORDER = ['msl', 'vo_850']

# 2-vars scaler 参数（快速检验用，正式推理从 model_dir 加载）
DETECTION_SCALER_PARAMS = {
    "msl": {"mean": 101164.820312, "std": 609.284912},
    "vo_850": {"mean": 2.1169e-06, "std": 4.8166e-05},
}

OLD_AIFS_NAME_RE = re.compile(
    r"AIFS_(\d{4})_(\d{2})_(\d{2})_(\d{2})_FCST_(\d+)h\.grib2$"
)
OPER_FC_NAME_RE = re.compile(r"(\d{14})-(\d+)h-oper-fc\.grib2$")


def is_supported_aifs_grib_filename(filename: str) -> bool:
    """Return True for supported AIFS GRIB2 file naming schemes."""
    basename = os.path.basename(filename)
    if not basename.endswith(".grib2"):
        return False
    return bool(OLD_AIFS_NAME_RE.match(basename) or OPER_FC_NAME_RE.match(basename))


def build_aifs_candidate_filenames(init_time: datetime, forecast_hour: int) -> List[str]:
    """Build supported filename candidates for one init time and forecast hour."""
    candidates = [
        (
            f"AIFS_{init_time.year}_{init_time.month:02d}_{init_time.day:02d}_"
            f"{init_time.hour:02d}_FCST_{forecast_hour:03d}h.grib2"
        ),
        f"{init_time:%Y%m%d%H%M%S}-{forecast_hour}h-oper-fc.grib2",
        f"{init_time:%Y%m%d%H%M%S}-{forecast_hour:03d}h-oper-fc.grib2",
    ]
    return list(dict.fromkeys(candidates))


def iter_aifs_candidate_dirs(aifs_data_dir: str, init_time: datetime) -> List[str]:
    """Return common directories used by old and operational AIFS layouts."""
    return [
        aifs_data_dir,
        os.path.join(aifs_data_dir, f"{init_time.year}{init_time.month:02d}"),
        os.path.join(aifs_data_dir, f"{init_time:%Y%m%d%H}"),
        os.path.join(aifs_data_dir, f"{init_time:%Y%m%d%H%M%S}"),
    ]


def find_aifs_candidate_path(aifs_data_dir: str,
                             init_time: datetime,
                             forecast_hour: int) -> Optional[str]:
    """Find a GRIB2 file across supported names and directory layouts."""
    candidate_names = build_aifs_candidate_filenames(init_time, forecast_hour)

    for directory in iter_aifs_candidate_dirs(aifs_data_dir, init_time):
        for fname in candidate_names:
            fpath = os.path.join(directory, fname)
            if os.path.exists(fpath):
                return fpath

    candidate_set = set(candidate_names)
    for root, _, fnames in os.walk(aifs_data_dir):
        for fname in fnames:
            if fname in candidate_set:
                return os.path.join(root, fname)

    return None


# -------------------------------------------------
# 文件名解析
# -------------------------------------------------
def parse_aifs_filename(filename: str) -> Tuple[datetime, int]:
    """
    解析 AIFS 文件名，返回 (valid_time, lead_hour)

    支持格式:
        AIFS_YYYY_MM_DD_HH_FCST_xxxh.grib2
        YYYYMMDDHHMMSS-Hh-oper-fc.grib2
    """
    basename = os.path.basename(filename)
    m = OLD_AIFS_NAME_RE.match(basename)
    if m:
        year, mon, day, hour, lead = (int(m.group(i)) for i in range(1, 6))
        init_time = datetime(year, mon, day, hour)
        valid_time = init_time + timedelta(hours=lead)
        return valid_time, lead

    m = OPER_FC_NAME_RE.match(basename)
    if m:
        init_time = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
        lead = int(m.group(2))
        valid_time = init_time + timedelta(hours=lead)
        return valid_time, lead

    raise ValueError(f"Cannot parse filename: {basename}")


def get_init_time_from_filename(filename: str) -> Optional[datetime]:
    """从 AIFS 文件名中提取初始化时间"""
    try:
        valid_time, lead = parse_aifs_filename(filename)
        return valid_time - timedelta(hours=lead)
    except ValueError:
        return None


# -------------------------------------------------
# GRIB2 文件读取
# -------------------------------------------------
def _read_grib_var(grbs, short_name: str, type_of_level: str, level: int):
    """
    从 pygrib 消息列表中读取指定变量

    Args:
        grbs: pygrib.open() 返回的对象
        short_name: GRIB shortName
        type_of_level: typeOfLevel 字符串
        level: level 值

    Returns:
        values: [H, W] numpy 数组
    """
    matches = grbs.select(shortName=short_name, typeOfLevel=type_of_level, level=level)
    if not matches:
        raise ValueError(f"Variable not found: shortName={short_name}, "
                         f"typeOfLevel={type_of_level}, level={level}")
    return _canonicalize_grib_global(matches[0].values, matches[0])


def _canonicalize_grib_global(values: np.ndarray, message) -> np.ndarray:
    """Return GRIB values on north-to-south lat and 0-360 increasing lon axes."""

    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(LATS_GLOBAL), len(LONS_GLOBAL)):
        return values

    try:
        lats, lons = message.latlons()
    except Exception:
        return values

    if lats.shape != values.shape or lons.shape != values.shape:
        return values

    lat1d = np.asarray(lats[:, 0], dtype=np.float64)
    lon1d = np.mod(np.asarray(lons[0, :], dtype=np.float64), 360.0)
    lat_order = np.argsort(-lat1d)
    lon_order = np.argsort(lon1d)
    return values[np.ix_(lat_order, lon_order)].astype(np.float32, copy=False)


def _read_all_grib_vars(grib_path: str, var_list: List[str]) -> Dict[str, np.ndarray]:
    """
    从 GRIB2 文件读取一组变量

    Args:
        grib_path: GRIB2 文件路径
        var_list: AIFS 变量名列表（如 ['mslp', 'u850', ...]）

    Returns:
        Dict[var_name, values]，values 为 [721, 1440] numpy 数组
    """
    import pygrib
    result = {}
    with pygrib.open(grib_path) as grbs:
        for var_name in var_list:
            if var_name not in GRIB_VAR_MAP:
                raise ValueError(f"Unknown variable: {var_name}")
            short_name, tol, level = GRIB_VAR_MAP[var_name]
            result[var_name] = _read_grib_var(grbs, short_name, tol, level)
    return result


def grib_to_xarray(grib_path: str,
                   lat_min: float = 0.0,
                   lat_max: float = 70.0,
                   lon_min: float = 100.0,
                   lon_max: float = 320.0) -> xr.Dataset:
    """
    将 AIFS GRIB2 文件转换为 xarray Dataset（2-vars 版本: msl + vo_850）

    Args:
        grib_path: GRIB2 文件路径
        lat_min, lat_max: 纬度裁剪范围
        lon_min, lon_max: 经度裁剪范围

    Returns:
        xr.Dataset: 包含 'msl' 和 'vo_850' 变量
    """
    valid_time, _ = parse_aifs_filename(grib_path)

    # 读取 msl, u850, v850
    data = _read_all_grib_vars(grib_path, ['mslp', 'u850', 'v850'])
    msl = data['mslp']
    u_850 = data['u850']
    v_850 = data['v850']

    # GRIB 数据纬度通常从北到南，经度从 0 到 360
    # 裁剪
    lat_idx = np.where((LATS_GLOBAL >= lat_min) & (LATS_GLOBAL <= lat_max))[0]
    lon_idx = np.where((LONS_GLOBAL >= lon_min) & (LONS_GLOBAL <= lon_max))[0]

    lat_coords = LATS_GLOBAL[lat_idx].astype(np.float32)
    lon_coords = LONS_GLOBAL[lon_idx].astype(np.float32)

    msl_cropped = msl[lat_idx[0]:lat_idx[-1] + 1, lon_idx[0]:lon_idx[-1] + 1]
    u_cropped = u_850[lat_idx[0]:lat_idx[-1] + 1, lon_idx[0]:lon_idx[-1] + 1]
    v_cropped = v_850[lat_idx[0]:lat_idx[-1] + 1, lon_idx[0]:lon_idx[-1] + 1]

    vo_850 = calc_vo850(u_cropped, v_cropped, lat_coords, lon_coords)

    coords = {
        "time": [np.datetime64(valid_time, "ns")],
        "lat": lat_coords,
        "lon": lon_coords,
    }

    def _var(arr):
        return (["time", "lat", "lon"], arr[np.newaxis].astype(np.float32))

    ds = xr.Dataset(
        {"msl": _var(msl_cropped), "vo_850": _var(vo_850)},
        coords=coords,
    )
    return ds


def calc_vo850(u: np.ndarray, v: np.ndarray, lat1d: np.ndarray, lon1d: np.ndarray) -> np.ndarray:
    """
    计算 850hPa 相对涡度

    Args:
        u, v: [H, W] 风场
        lat1d, lon1d: 1D 坐标

    Returns:
        vo_850: [H, W] 相对涡度
    """
    R = 6371000.0
    dlon_rad = np.deg2rad(np.abs(float(lon1d[1]) - float(lon1d[0])))
    dlat_rad = np.deg2rad(np.abs(float(lat1d[1]) - float(lat1d[0])))

    cos_lat = np.cos(np.deg2rad(lat1d))[:, np.newaxis]
    dx = R * dlon_rad * cos_lat
    dy = R * dlat_rad

    dv_dx = np.gradient(v, axis=1) / dx
    du_dy = np.gradient(u, axis=0) / (-dy)

    return (dv_dx - du_dy).astype(np.float32)


# -------------------------------------------------
# GRIB2 文件查找（订正模型使用）
# -------------------------------------------------
def find_aifs_grib_for_time(aifs_data_dir: str, target_time: datetime) -> Optional[Tuple[str, datetime, int]]:
    """
    查找包含目标时间的 AIFS GRIB2 文件

    文件路径格式: {aifs_data_dir}/{YYYYMM}/AIFS_YYYY_MM_DD_HH_FCST_HHHh.grib2

    Returns:
        (fpath, init_time, forecast_hour) 或 None
    """
    for init_hour in [12, 0]:
        init_time = target_time.replace(hour=init_hour, minute=0, second=0, microsecond=0)
        if target_time.hour < init_hour:
            init_time = init_time - timedelta(days=1)

        forecast_hour = int((target_time - init_time).total_seconds() / 3600)
        if forecast_hour < 0 or forecast_hour > 240 or forecast_hour % 6 != 0:
            continue

        fpath = find_aifs_candidate_path(aifs_data_dir, init_time, forecast_hour)
        if fpath is not None:
            return fpath, init_time, forecast_hour

    return None


def find_aifs_grib_by_init(aifs_data_dir: str, init_time: datetime, forecast_hour: int) -> Optional[str]:
    """
    根据初始化时间和预报时效查找 AIFS GRIB2 文件

    文件路径格式: {aifs_data_dir}/{YYYYMM}/AIFS_YYYY_MM_DD_HH_FCST_HHHh.grib2

    Args:
        aifs_data_dir: AIFS 数据根目录
        init_time: 初始化时间
        forecast_hour: 预报时效（小时）

    Returns:
        文件路径 或 None
    """
    return find_aifs_candidate_path(aifs_data_dir, init_time, forecast_hour)


# -------------------------------------------------
# 经纬度与网格索引转换
# -------------------------------------------------
def latlon_to_grid_index(lat: float, lon: float) -> Tuple[int, int]:
    """
    将经纬度转换为 AIFS 全局网格索引

    Args:
        lat: 纬度 (-90 to 90)
        lon: 经度 (0 to 360)

    Returns:
        (row, col) 网格索引
    """
    if lon < 0:
        lon += 360.0

    row = int(round((AIFS_LAT_MAX - lat) / AIFS_RESOLUTION))
    col = int(round((lon - AIFS_LON_MIN) / AIFS_RESOLUTION))

    row = max(0, min(row, AIFS_HEIGHT - 1))
    col = max(0, min(col, AIFS_WIDTH - 1))
    return row, col


def grid_index_to_latlon(row: int, col: int) -> Tuple[float, float]:
    """将网格索引转换为经纬度"""
    lat = AIFS_LAT_MAX - row * AIFS_RESOLUTION
    lon = AIFS_LON_MIN + col * AIFS_RESOLUTION
    return lat, lon


# -------------------------------------------------
# 区域裁剪
# -------------------------------------------------
def crop_aifs_field(data: np.ndarray,
                    lat: float,
                    lon: float,
                    field_size: int = 64) -> np.ndarray:
    """
    以 (lat, lon) 为中心，从 AIFS 全球场中裁剪 field_size×field_size 区域

    Args:
        data: [C, H, W] AIFS 全球场数据
        lat, lon: 中心点经纬度
        field_size: 裁剪区域大小（偶数）

    Returns:
        cropped: [C, field_size, field_size]
    """
    half = field_size // 2
    center_row, center_col = latlon_to_grid_index(lat, lon)

    row_start = center_row - half
    row_end = center_row + half
    col_start = center_col - half
    col_end = center_col + half

    # 边界检查与填充
    pad_top = max(0, -row_start)
    pad_bottom = max(0, row_end - AIFS_HEIGHT)
    pad_left = max(0, -col_start)
    pad_right = max(0, col_end - AIFS_WIDTH)

    row_start = max(0, row_start)
    row_end = min(AIFS_HEIGHT, row_end)
    col_start = max(0, col_start)
    col_end = min(AIFS_WIDTH, col_end)

    cropped = data[:, row_start:row_end, col_start:col_end]

    if pad_top or pad_bottom or pad_left or pad_right:
        cropped = np.pad(
            cropped,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode='edge'
        )

    return cropped


def crop_aifs_field_from_grib(grib_path: str,
                              lat: float,
                              lon: float,
                              field_size: int = 64,
                              var_order: List[str] = None) -> Optional[np.ndarray]:
    """
    从 GRIB2 文件中读取并裁剪 AIFS 区域

    Args:
        grib_path: GRIB2 文件路径
        lat, lon: 中心点经纬度
        field_size: 裁剪区域大小
        var_order: 变量顺序列表，None 则使用全部 16 个变量

    Returns:
        [n_vars, field_size, field_size] numpy 数组，或 None（失败时）
    """
    if var_order is None:
        var_order = AIFS_VAR_ORDER

    try:
        data = _read_all_grib_vars(grib_path, var_order)

        # 堆叠为 [C, H, W]
        stacked = np.stack([data[v] for v in var_order], axis=0)

        # 裁剪
        cropped = crop_aifs_field(stacked, lat, lon, field_size)
        return cropped.astype(np.float32)

    except Exception as e:
        print(f"[ERROR] Failed to crop from {grib_path}: {e}")
        return None


# -------------------------------------------------
# 标准化工具
# -------------------------------------------------
def load_aifs_normalization_stats(stats_file: str) -> Dict:
    """加载 AIFS 标准化统计参数"""
    if not os.path.exists(stats_file):
        return {}
    with open(stats_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_aifs_tensor(tensor: torch.Tensor,
                          var_order: List[str],
                          normalization_stats: Dict) -> torch.Tensor:
    """
    对 AIFS 张量进行标准化

    Args:
        tensor: [C, H, W] 原始 AIFS 数据
        var_order: 变量名称列表
        normalization_stats: 标准化参数字典

    Returns:
        标准化后的张量 [C, H, W]
    """
    tensor = tensor.clone()
    for i, var_name in enumerate(var_order):
        if var_name not in normalization_stats:
            continue

        spec = normalization_stats[var_name]
        ch = tensor[i]

        if spec.get('method') == 'log1p+zscore':
            ch = torch.log1p(torch.clamp(ch, min=0.0))
            mean = spec['mean_log']
            std = spec['std_log']
        elif spec.get('method') == 'zscore':
            mean = spec['mean']
            std = spec['std']
        else:
            continue

        std = max(std, 1e-8)
        tensor[i] = (ch - mean) / std

    return tensor


def denormalize_aifs_tensor(tensor: torch.Tensor,
                            var_order: List[str],
                            normalization_stats: Dict) -> torch.Tensor:
    """反标准化 AIFS 张量"""
    tensor = tensor.clone()
    for i, var_name in enumerate(var_order):
        if var_name not in normalization_stats:
            continue

        spec = normalization_stats[var_name]
        if spec.get('method') == 'log1p+zscore':
            mean = spec['mean_log']
            std = spec['std_log']
        elif spec.get('method') == 'zscore':
            mean = spec['mean']
            std = spec['std']
        else:
            continue

        std = max(std, 1e-8)
        tensor[i] = tensor[i] * std + mean

        if spec.get('method') == 'log1p+zscore':
            tensor[i] = torch.expm1(tensor[i])

    return tensor


# -------------------------------------------------
# 强度提取
# -------------------------------------------------
def extract_aifs_intensity(data: np.ndarray, var_order: List[str]) -> Tuple[float, float]:
    """
    从 AIFS 场数据中提取台风强度指标

    Args:
        data: [n_vars, H, W] AIFS 原始数据（未归一化）
        var_order: 变量顺序列表

    Returns:
        (max_wind_knots, min_pres_hpa)
    """
    u10_idx = var_order.index('u10') if 'u10' in var_order else None
    v10_idx = var_order.index('v10') if 'v10' in var_order else None
    mslp_idx = var_order.index('mslp') if 'mslp' in var_order else None

    if u10_idx is None or v10_idx is None or mslp_idx is None:
        raise ValueError(f"Required variables (u10, v10, mslp) not found in {var_order}")

    u10 = data[u10_idx]
    v10 = data[v10_idx]

    wind_speed_ms = np.sqrt(u10**2 + v10**2)
    wind_speed_knots = wind_speed_ms * 1.94384
    max_wind = float(wind_speed_knots.max())

    # AIFS mslp 原始单位为 Pa，转换为 hPa
    min_pres = float(data[mslp_idx].min()) / 100.0

    return max_wind, min_pres


# -------------------------------------------------
# 批量文件扫描
# -------------------------------------------------
def scan_aifs_grib_files(aifs_dir: str) -> List[str]:
    """扫描目录下的所有 AIFS GRIB2 文件"""
    files = []
    for root, _, fnames in os.walk(aifs_dir):
        for fname in fnames:
            if is_supported_aifs_grib_filename(fname):
                files.append(os.path.join(root, fname))
    return sorted(files, key=lambda f: (*parse_aifs_filename(f), f))


def group_grib_files_by_init_time(files: List[str]) -> Dict[datetime, List[str]]:
    """将 GRIB2 文件列表按初始化时间 (init_time) 分组

    Args:
        files: GRIB2 文件路径列表

    Returns:
        Dict[init_time, 文件路径列表]，按 init_time 升序排列
    """
    groups: Dict[datetime, List[str]] = {}
    for f in files:
        init_time = get_init_time_from_filename(f)
        if init_time is None:
            continue
        if init_time not in groups:
            groups[init_time] = []
        groups[init_time].append(f)

    # 对每个组内的文件排序
    for init_time in groups:
        groups[init_time].sort(key=lambda f: (parse_aifs_filename(f)[1], f))

    # 按 init_time 升序返回
    return dict(sorted(groups.items()))


def find_latest_init_time_files(aifs_dir: str) -> Tuple[Optional[datetime], List[str]]:
    """查找目录中最新初始化时间的所有 GRIB2 文件

    Args:
        aifs_dir: AIFS GRIB2 数据目录

    Returns:
        (latest_init_time, 该时次的所有文件路径列表)，若无文件则返回 (None, [])
    """
    files = scan_aifs_grib_files(aifs_dir)
    if not files:
        return None, []

    groups = group_grib_files_by_init_time(files)
    if not groups:
        return None, []

    latest_init_time = max(groups.keys())
    return latest_init_time, groups[latest_init_time]


def find_files_by_init_date(
    aifs_dir: str,
    target_date,
) -> Tuple[Optional[datetime], List[str]]:
    """查找指定初始化日期的所有 GRIB2 文件

    支持传入 datetime.datetime 或 datetime.date。
    若传入 date，则匹配该日期 00:00 开始的所有时次。

    Args:
        aifs_dir: AIFS GRIB2 数据目录
        target_date: 目标日期 (datetime 或 date)

    Returns:
        (matched_init_time, 文件路径列表)。
        如果 target_date 是 date 类型且当天有多个 init_time，
        返回该日期最晚的一个 init_time 及其文件。
        无匹配则返回 (None, [])。
    """
    from datetime import date as _date

    files = scan_aifs_grib_files(aifs_dir)
    if not files:
        return None, []

    groups = group_grib_files_by_init_time(files)
    if not groups:
        return None, []

    # 如果传入的是 date 对象，匹配该日期的所有 init_time，取最晚的
    if isinstance(target_date, _date) and not isinstance(target_date, datetime):
        day_init_times = [t for t in groups.keys()
                          if t.year == target_date.year and
                          t.month == target_date.month and
                          t.day == target_date.day]
        if not day_init_times:
            return None, []
        matched = max(day_init_times)
        return matched, groups[matched]

    # 如果传入的是 datetime，精确匹配（忽略秒以下精度）
    target_date = target_date.replace(minute=0, second=0, microsecond=0)
    if target_date in groups:
        return target_date, groups[target_date]

    # 尝试找同一天同一小时
    for init_time, file_list in groups.items():
        if (init_time.year == target_date.year and
            init_time.month == target_date.month and
            init_time.day == target_date.day and
            init_time.hour == target_date.hour):
            return init_time, file_list

    return None, []
