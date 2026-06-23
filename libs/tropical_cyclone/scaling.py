from typing import List
import xarray as xr
import torch


class StandardScaler:
    """
    基于 NetCDF mean/std 文件的标准化器

    从独立的 mean.nc 和 std.nc 文件加载各驱动变量的统计量，
    对输入 tensor 进行 (x - mean) / std 标准化。
    """
    def __init__(self,
            mean_src: str,
            std_src: str,
            drivers: List[str],
            dtype = torch.float32
        ) -> None:
        self._mean_ds = xr.load_dataset(mean_src)[drivers]
        self._std_ds = xr.load_dataset(std_src)[drivers]
        self._mean = torch.as_tensor(self._mean_ds[drivers].to_array().data, dtype=dtype)
        self._std = torch.as_tensor(self._std_ds[drivers].to_array().data, dtype=dtype)

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        scaled_tensor = ((tensor - self._mean) / self._std)
        return scaled_tensor

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        rescaled_tensor = ((tensor * self._std) + self._mean)
        return rescaled_tensor

    def get_mean(self):
        return self._mean

    def get_std(self):
        return self._std
