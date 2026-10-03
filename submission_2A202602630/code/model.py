"""model.py — MLP cho bài toán 7 lớp, shape cố định (xem README mục 3 và GUIDE, "Quy định kiến trúc"):

    x (B, 54) -> Linear(54, h1) -> ReLU -> [Dropout] -> Linear(h1, h2) -> ReLU -> [Dropout]
              -> ... -> Linear(h_last, 7) -> logits (B, 7)

Quy tắc:
  - Lớp cuối ra logit thô, KHÔNG softmax trong model (softmax nằm trong hàm mất mát).
  - Dropout chỉ đặt sau ReLU của lớp ẩn; không đặt trên đầu vào hay logit.
  - Mọi nn.Linear đều có bias. Không BatchNorm, không residual.
  - Số tham số phải khớp EXPECTED_PARAMS bên dưới.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Số tham số bắt buộc ứng với từng kiến trúc (in_features=54, num_classes=7)
EXPECTED_PARAMS = {
    (256, 128): 47_879,        # M-base  (baseline)
    (512, 256): 161_287,       # M-wide  (tuỳ chọn)
    (256, 128, 64): 55_687,    # M-deep  (tuỳ chọn)
}

INIT_CHOICES = ("zeros", "normal", "xavier", "he", "default")


class MLP(nn.Module):
    """MLP theo quy định ở đầu file.

    Args:
        hidden:   tuple số nơ-ron các lớp ẩn, ví dụ (256, 128)
        dropout:  xác suất TẮT nơ-ron q (nn.Dropout dùng p chính là xác suất tắt); 0.0 = không dùng
        init:     "zeros" | "normal" | "xavier" | "he" | "default"
    """

    def __init__(self, hidden=(256, 128), dropout: float = 0.0, init: str = "he",
                 in_features: int = 54, num_classes: int = 7):
        super().__init__()
        assert 0.0 <= dropout < 1.0, f"dropout phải trong [0, 1), nhận {dropout}"
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.init = init

        layers: list[nn.Module] = []
        n_in = in_features
        for h in self.hidden:
            layers += [nn.Linear(n_in, h), nn.ReLU()]
            if dropout > 0:  # q = 0 thì không thêm lớp nào, để in model ra gọn và đúng như baseline
                layers.append(nn.Dropout(p=dropout))
            n_in = h
        layers.append(nn.Linear(n_in, num_classes))  # lớp ra: logit thô, không ReLU/Dropout/softmax
        self.net = nn.Sequential(*layers)

        init_weights(self, init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 54) float32  ->  logits: (B, 7) float32."""
        return self.net(x)


def init_weights(model: nn.Module, init: str) -> None:
    """Khởi tạo tham số của MỌI nn.Linear (bias = 0, trừ "default").

    init:
        "zeros"   : W = 0
        "normal"  : W ~ N(0, 0.01^2)
        "xavier"  : nn.init.xavier_normal_, Var = 2/(n_in+n_out)  (công thức của PyTorch, không phải 1/n_in)
        "he"      : nn.init.kaiming_normal_(w, nonlinearity="relu"), Var = 2/n_in (mode="fan_in")
        "default" : giữ khởi tạo mặc định của nn.Linear (Kaiming-uniform với a=√5, Var ≈ 1/(3·n_in);
                    KHÔNG phải He). Bias cũng giữ mặc định.
    """
    if init not in INIT_CHOICES:
        raise ValueError(f"init phải thuộc {INIT_CHOICES}, nhận {init!r}")
    if init == "default":
        return
    for m in model.modules():
        if not isinstance(m, nn.Linear):
            continue
        w = m.weight
        if init == "zeros":
            nn.init.zeros_(w)
        elif init == "normal":
            nn.init.normal_(w, mean=0.0, std=0.01)
        elif init == "xavier":
            nn.init.xavier_normal_(w)
        elif init == "he":
            nn.init.kaiming_normal_(w, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(m.bias)


def count_params(model: nn.Module) -> int:
    """Tổng số tham số huấn luyện được. Dùng để assert với EXPECTED_PARAMS ngay sau khi tạo model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def activation_stats(model: nn.Module, x: torch.Tensor) -> list[float]:
    """Độ lệch chuẩn của kích hoạt SAU MỖI ReLU (tức đầu ra từng lớp ẩn), cộng thêm std của logits ở cuối.

    Dùng cho thí nghiệm khởi tạo: ở bước 0, một lô val. Trả về list dài len(hidden) + 1.
    Model được chuyển sang eval() (dropout tắt) rồi trả lại chế độ cũ.
    """
    was_training = model.training
    model.eval()
    stds: list[float] = []
    h = x
    for layer in model.net:
        h = layer(h)
        if isinstance(layer, nn.ReLU):
            stds.append(h.float().std().item())
    stds.append(h.float().std().item())  # logits
    model.train(was_training)
    return stds
