import torch.nn as nn
import torch


class VGG_V3(nn.Module):
    """
    VGG 风格 CNN，用于台风中心点坐标回归。

    输入: [B, 2, H, W]   (msl + vo_850)
    输出: [B, 2]         (patch 内 lat/lon 坐标，负值表示无台风)
    """
    def __init__(self,
        in_channels: int,
        out_channels: int,
        activation: str = 'nn.Identity',
        out_activation: str = 'nn.Identity',
        kernel_size: int = 3,
        init_std: float = None,
        dropout: float = 0.0,
        noise: bool = False,
    ) -> None:
        super().__init__()
        self.noise = noise
        if noise:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.g_cuda = torch.Generator(device=device)
        activation = eval(activation)
        out_activation = eval(out_activation)
        self.vgg = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=kernel_size, padding='same'),
            activation(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding="same"),
            activation(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding="same"),
            activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same"),
            activation(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"),
            activation(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"),
            activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same"),
            activation(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same"),
            activation(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same"),
            activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, padding="same"),
            activation(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, padding="same"),
            activation(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, padding="same"),
            activation(),
            nn.Dropout(dropout),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, padding="valid"),
            activation(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, padding="valid"),
            activation(),
            nn.Dropout(dropout),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=2, padding="valid"),
            activation(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=2, padding="valid"),
            activation(),
            nn.Dropout(dropout),

            nn.Flatten(),

            nn.Linear(in_features=1024, out_features=1024),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(in_features=1024, out_features=512),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(in_features=512, out_features=512),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(in_features=512, out_features=256),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(in_features=256, out_features=out_channels),

            out_activation(),
        )
        if init_std is not None:
            self._init_normal(init_std)

    def forward(self, x: torch.Tensor):
        if self.noise and torch.is_grad_enabled():
            if torch.rand(1)[0] > 0.5:
                x += torch.randn_like(x) * 0.2
        x = self.vgg(x)
        return x

    def _init_normal(self, std: float = 0.05):
        with torch.no_grad():
            for module in self.vgg.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0, std=std)
                    nn.init.normal_(module.bias, mean=0, std=std)
