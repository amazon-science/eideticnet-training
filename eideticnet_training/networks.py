import torch.nn as nn

from .core import EideticNetwork
from .prune.bridge import bridge_prune, bridge_prune_residual


SUPPORTED = [nn.Linear, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d]


class ConvNet(EideticNetwork):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_layers: int = 2,
        width: int = 32,
        dropout: float = 0,
        bn=True,
    ):
        super().__init__()
        self.bn = bn
        self.dropout = dropout

        blocks = [self.conv_block(in_channels, width, pool=True)]
        blocks[0][0].first_layer = True

        blocks.append(self.conv_block(width, width * 4))
        for _ in range(num_layers - 1):
            blocks.append(self.conv_block(width * 4, width * 4))

        blocks.append(self.conv_block(width * 4, width * 8, pool=True))
        for _ in range(num_layers - 1):
            blocks.append(self.conv_block(width * 8, width * 8))

        blocks.append(self.conv_block(width * 8, width * 32, pool=True))
        for _ in range(num_layers - 1):
            blocks.append(self.conv_block(width * 32, width * 32))
        self.blocks = nn.ModuleList(blocks)
        self.formatter = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten())
        self.classifiers = nn.ModuleList(
            [nn.Linear(width * 32, nc) for nc in num_classes]
        )
        for c in self.classifiers:
            c.last_layer = True

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        flat = self.formatter(x)
        return [c(flat) for c in self.classifiers]

    def _bridge_prune(self, pct, pruning_type=2, score_threshold=None):
        for i in range(len(self.blocks) - 1):
            bridge_prune(
                self.blocks[i][0],
                self.blocks[i][1],
                self.blocks[i + 1][0],
                pct,
                pruning_type=pruning_type,
                score_threshold=score_threshold,
            )
        bridge_prune(
            self.blocks[-1][0],
            self.blocks[-1][1],
            self.classifiers[self.phase],
            pct,
            pruning_type=pruning_type,
        )

    def conv_block(self, in_channels, out_channels, pool=False):
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=not self.bn,
            ),
            nn.BatchNorm2d(out_channels) if self.bn else nn.Identity(),
            nn.Dropout(self.dropout) if self.dropout else nn.Identity(),
            # Inplace ReLU is incompatible with backward hooks.
            nn.ReLU(inplace=False),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)


class ResNet(EideticNetwork):
    def __init__(
        self,
        in_channels,
        num_classes,
        n_blocks,
        expansion=1,
        bn=True,
        low_res=True,
    ):
        super().__init__()
        self.bn = bn

        if len(n_blocks) != 4:
            raise ValueError("number of blocks should be 4")

        blocks = [self.conv_block(in_channels, 64, 64, pool=False)]
        if not low_res:
            blocks[0][0] = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                padding=3,
                stride=1,
                bias=not self.bn,
            )
        blocks[0][0].first_layer = True
        res = [nn.Conv2d(in_channels, 64, 1, bias=False, stride=1)]
        res[0].first_layer = True

        for _ in range(n_blocks[0] - 1):
            blocks.append(self.conv_block(64, 64 * expansion, 64))
            res.append(nn.Conv2d(64, 64, 1, bias=False))

        blocks.append(self.conv_block(64, 64 * expansion, 128, pool=True))
        res.append(nn.Conv2d(64, 128, 1, bias=False, stride=2))

        for _ in range(n_blocks[1] - 1):
            blocks.append(self.conv_block(128, 128 * expansion, 128))
            res.append(nn.Conv2d(128, 128, 1, bias=False))

        blocks.append(self.conv_block(128, 128 * expansion, 256, pool=True))
        res.append(nn.Conv2d(128, 256, 1, bias=False, stride=2))

        for _ in range(n_blocks[2] - 1):
            blocks.append(self.conv_block(256, 256 * expansion, 256))
            res.append(nn.Conv2d(256, 256, 1, bias=False))

        blocks.append(self.conv_block(256, 256 * expansion, 512, pool=True))
        res.append(nn.Conv2d(256, 512, 1, bias=False, stride=2))

        for _ in range(n_blocks[3]):
            blocks.append(self.conv_block(512, 512 * expansion, 512))
            res.append(nn.Conv2d(512, 512, 1, bias=False))

        blocks.append(self.conv_block(512, 512 * expansion, 512 * expansion))
        res.append(nn.Conv2d(512, 512 * expansion, 1, bias=False))

        self.res = nn.ModuleList(res)
        self.blocks = nn.ModuleList(blocks)

        self.formatter = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifiers = nn.ModuleList(
            [nn.Linear(512 * expansion, nc) for nc in num_classes]
        )
        for c in self.classifiers:
            c.last_layer = True

    def forward(self, x):
        for b, r in zip(self.blocks, self.res):
            x = b(x) + r(x)
        flat = self.formatter(x)
        return [c(flat) for c in self.classifiers]

    def _bridge_prune(self, pct, pruning_type=2, score_threshold=None):
        for i in range(len(self.blocks) - 1):
            bridge_prune(
                self.blocks[i][0],
                self.blocks[i][1],
                self.blocks[i][3],
                pct,
                pruning_type=pruning_type,
                score_threshold=score_threshold,
            )
            bridge_prune(
                self.blocks[i][3],
                self.blocks[i][4],
                self.blocks[i + 1][0],
                pct,
                pruning_type=pruning_type,
                score_threshold=score_threshold,
            )
            bridge_prune_residual(
                self.blocks[i][0], self.blocks[i][3], self.res[i]
            )

        bridge_prune(
            self.blocks[-1][0],
            self.blocks[-1][1],
            self.blocks[-1][3],
            pct,
            pruning_type=pruning_type,
            score_threshold=score_threshold,
        )
        bridge_prune(
            self.blocks[-1][3],
            self.blocks[-1][4],
            self.classifiers[self.phase],
            pct,
            pruning_type=pruning_type,
            score_threshold=score_threshold,
        )
        bridge_prune_residual(
            self.blocks[-1][0], self.blocks[-1][3], self.res[-1]
        )

    def conv_block(self, in_channels, mid_channels, out_channels, pool=False):
        layers = [
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                bias=not self.bn,
            ),
            nn.BatchNorm2d(mid_channels) if self.bn else nn.Identity(),
            # Inplace ReLU is incompatible with backward hooks.
            nn.ReLU(inplace=False),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=not self.bn,
            ),
            nn.BatchNorm2d(out_channels) if self.bn else nn.Identity(),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)


class MLP(EideticNetwork):

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        num_layers: int = 2,
        width: int = 4096,
        dropout: float = 0,
        bn: bool = True,
    ):
        super(MLP, self).__init__()
        self.bn = bn
        self.dropout = dropout
        self.blocks = nn.ModuleList(
            [self.conv_block(in_features, width)]
            + [self.conv_block(width, width) for _ in range(num_layers)]
        )
        self.blocks[0][0].first_layer = True
        self.classifiers = nn.ModuleList(
            [nn.Linear(width, nc) for nc in num_classes]
        )
        for i in range(len(self.classifiers)):
            self.classifiers[i].last_layer = True
        self.phase = None

    def forward(self, x):
        if self.phase is None:
            raise RuntimeError("Need to call `set_phase` first")
        x = x.flatten(1)
        for b in self.blocks:
            x = b(x)
        return [c(x) for c in self.classifiers]

    def _bridge_prune(self, pct, pruning_type=2, score_threshold=None):
        for i in range(len(self.blocks) - 1):
            bridge_prune(
                self.blocks[i][0],
                self.blocks[i][1],
                self.blocks[i + 1][0],
                pct,
                pruning_type=pruning_type,
                score_threshold=score_threshold,
            )
        bridge_prune(
            self.blocks[-1][0],
            self.blocks[-1][1],
            self.classifiers[self.phase],
            pct,
            pruning_type=pruning_type,
            score_threshold=score_threshold,
        )

    def conv_block(self, in_channels, out_channels):
        layers = [
            nn.Linear(in_channels, out_channels, bias=not self.bn),
            nn.BatchNorm1d(out_channels) if self.bn else nn.Identity(),
            nn.Dropout(self.dropout) if self.dropout else nn.Identity(),
            # Inplace ReLU is incompatible with backward hooks.
            nn.ReLU(inplace=False),
        ]
        return nn.Sequential(*layers)
