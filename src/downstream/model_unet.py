import torch
import torch.nn as nn

class DoubleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            DoubleConv1D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)


class Up1D(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
            mid_channels = in_channels // 2
            self.conv = DoubleConv1D(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv1D(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: низ, x2: skip
        x1 = self.up(x1)

        # подгоняем длину, если не совпадает
        diff = x2.size(-1) - x1.size(-1)
        if diff > 0:
            x1 = nn.functional.pad(x1, (diff // 2, diff - diff // 2))
        elif diff < 0:
            x2 = nn.functional.pad(x2, (-diff // 2, -diff - (-diff // 2)))

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet1D_Light(nn.Module):
    """
    U-Net 1D с 4 уровнями даунсемплинга, вдохновлён Hong et al., 2025.
    Каналы: 32 → 64 → 128 → 256, bottleneck 512.
    """
    def __init__(self, n_channels, n_classes, base_ch=32, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        ch1 = base_ch
        ch2 = base_ch * 2
        ch3 = base_ch * 4
        ch4 = base_ch * 8
        bottleneck_ch = base_ch * 16  # 512 при base_ch=32

        # encoder
        self.inc = DoubleConv1D(n_channels, ch1)
        self.down1 = Down1D(ch1, ch2)
        self.down2 = Down1D(ch2, ch3)
        self.down3 = Down1D(ch3, ch4)
        self.down4 = Down1D(ch4, bottleneck_ch)

        # decoder
        self.up1 = Up1D(bottleneck_ch + ch4, ch4, bilinear)
        self.up2 = Up1D(ch4 + ch3, ch3, bilinear)
        self.up3 = Up1D(ch3 + ch2, ch2, bilinear)
        self.up4 = Up1D(ch2 + ch1, ch1, bilinear)
        self.outc = OutConv1D(ch1, n_classes)

    def encode(self, x):
        x1 = self.inc(x)     # (N, ch1, L)
        x2 = self.down1(x1)  # (N, ch2, L/2)
        x3 = self.down2(x2)  # (N, ch3, L/4)
        x4 = self.down3(x3)  # (N, ch4, L/8)
        x5 = self.down4(x4)  # (N, bottleneck_ch, L/16)
        return x1, x2, x3, x4, x5

    def decode(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encode(x)
        logits = self.decode(x1, x2, x3, x4, x5)
        return logits, x5  # x5 — bottleneck

class UNet1DEncoder(nn.Module):
    """
    Обёртка над encoder-частью обученного U-Net.
    На вход:  x ∈ R^{B×C×L}
    На выход: bottleneck-фичи x5 ∈ R^{B×C_bottleneck×L_reduced}
    """
    def __init__(self, unet_model: nn.Module):
        super().__init__()
        # просто переиспользуем уже обученные блоки
        self.inc = unet_model.inc
        self.down1 = unet_model.down1
        self.down2 = unet_model.down2
        self.down3 = unet_model.down3
        self.down4 = unet_model.down4

    def forward(self, x):
        # это ровно то, что делал encode() в полном U-Net
        x1 = self.inc(x)    # (B, ch1, L)
        x2 = self.down1(x1) # (B, ch2, L/2)
        x3 = self.down2(x2) # (B, ch3, L/4)
        x4 = self.down3(x3) # (B, ch4, L/8)
        x5 = self.down4(x4) # (B, bottleneck_ch, L/16)
        return x5

class ERPHead(nn.Module):
    """
    Минимальная классификационная голова для P300.
    Вход:  (B, F, T)
    Выход: (B, 2) logits
    """
    def __init__(self, in_features=512, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, z):
        # z: (B, F, T)
        z = z.mean(dim=-1)   # global average pooling → (B, F)
        logits = self.fc(z)  # (B, 2)
        return logits
    
class P300Model(nn.Module):
    """
    Полная downstream модель:
    x → encoder → head → logits
    """
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        z = self.encoder(x)
        logits = self.head(z)
        return logits

def build_model(
    scenario: str,
    device: str,
    encoder_checkpoint: str = None,
):
    """
    Собирает downstream-модель для сценариев:
    - scratch
    - ssl_ft

    scratch:
        encoder случайный

    ssl_ft:
        encoder той же архитектуры + загруженные SSL-веса

    head всегда создаётся заново
    """
    valid_scenarios = {"scratch", "ssl_ft"}
    if scenario not in valid_scenarios:
        raise ValueError(f"Unknown scenario: {scenario}. Expected one of {valid_scenarios}")

    # создаём базовый U-Net только как контейнер encoder-блоков
    unet = UNet1D_Light(
        n_channels=14,
        n_classes=14,
        base_ch=32,
        bilinear=True,
    )
    encoder = UNet1DEncoder(unet)

    if scenario == "ssl_ft":
        if encoder_checkpoint is None:
            raise ValueError("encoder_checkpoint must be provided for scenario='ssl_ft'")
        encoder = load_encoder_checkpoint_into_model_encoder(
            model_encoder=encoder,
            encoder_checkpoint=encoder_checkpoint,
            device=device,
        )

    head = ERPHead(in_features=512, num_classes=2)
    model = P300Model(encoder=encoder, head=head)
    model = model.to(device)

    return model

# Загрузка весов
def load_encoder_checkpoint_into_model_encoder(model_encoder: nn.Module, encoder_checkpoint: str, device: str = "cpu"):
    """
    Загружает encoder_best.pt в encoder downstream-модели.

    Ожидаемый формат checkpoint:
    {
        'inc': state_dict(...),
        'down1': state_dict(...),
        'down2': state_dict(...),
        'down3': state_dict(...),
        'down4': state_dict(...),
    }
    """
    ckpt = torch.load(encoder_checkpoint, map_location=device)

    expected_keys = ["inc", "down1", "down2", "down3", "down4"]
    missing = [k for k in expected_keys if k not in ckpt]
    if len(missing) > 0:
        raise KeyError(f"Encoder checkpoint is missing keys: {missing}. Found keys: {list(ckpt.keys())}")

    model_encoder.inc.load_state_dict(ckpt["inc"], strict=True)
    model_encoder.down1.load_state_dict(ckpt["down1"], strict=True)
    model_encoder.down2.load_state_dict(ckpt["down2"], strict=True)
    model_encoder.down3.load_state_dict(ckpt["down3"], strict=True)
    model_encoder.down4.load_state_dict(ckpt["down4"], strict=True)

    return model_encoder


# Счётчик параметров
def count_all_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)