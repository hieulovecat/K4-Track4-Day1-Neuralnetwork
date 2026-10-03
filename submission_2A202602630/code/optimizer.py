"""optimizer.py — chọn bộ tối ưu, bộ lập lịch lr và cắt gradient.

Được dùng torch.optim.* và torch.nn.utils.clip_grad_norm_ (xem README mục 5).
File này gom việc chọn bộ tối ưu và cắt gradient để `train.py` gọn và mọi thí nghiệm công bằng.

Công thức cần hiểu (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import math

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")
SCHEDULERS = (None, "cosine", "warmup_cosine")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer.

    Chú ý: weight_decay của Adam (L2 cộng vào gradient, nên bị chia cho √v̂ như mọi gradient khác)
    khác weight_decay của AdamW (trừ thẳng lr·wd·w, không qua bước chuẩn hoá thích nghi).
    """
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer phải thuộc {OPTIMIZERS}, nhận {name!r}")
    if lr is None or lr <= 0:
        raise ValueError(f"lr phải là số dương, nhận {lr!r}")
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, warmup_steps: int = 0, **kwargs):
    """(Tuỳ chọn) Bộ lập lịch lr, gọi scheduler.step() SAU MỖI BƯỚC cập nhật (không phải mỗi epoch).

    name:
        None            : lr cố định (baseline)
        "cosine"        : lr giảm theo cosine từ lr gốc về 0 trong total_steps bước (CosineAnnealingLR)
        "warmup_cosine" : tăng tuyến tính từ ~0 lên lr gốc trong warmup_steps bước, rồi cosine về 0
                          (dùng khi thử quy tắc "batch ×k thì lr ×k, kèm khởi động")
    Dùng scheduler ở thí nghiệm nào thì ghi vào cột notes của bảng.
    """
    if name not in SCHEDULERS:
        raise ValueError(f"scheduler phải thuộc {SCHEDULERS}, nhận {name!r}")
    if name is None:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    warmup_steps = max(1, int(warmup_steps))

    def factor(step: int) -> float:  # hệ số nhân với lr gốc ở bước `step` (0-indexed)
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def clip_gradients(params, max_norm: float | None, as_tensor: bool = False):
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    max_norm=None: chỉ đo chuẩn, không cắt (clip_grad_norm_ với max_norm=inf không đổi gradient).
    Khi dùng FP16 + GradScaler: phải scaler.unscale_(optimizer) TRƯỚC khi gọi hàm này,
    nếu không chuẩn đo được bị nhân thêm hệ số scale s.
    as_tensor=True: trả tensor 0-chiều trên device thay vì float. float() buộc CPU chờ GPU
    ở mỗi bước; vòng huấn luyện dùng tensor rồi gom lại cuối epoch để không bị chậm.
    """
    params = [p for p in params if p.grad is not None]
    limit = float("inf") if max_norm is None else float(max_norm)
    total_norm = torch.nn.utils.clip_grad_norm_(params, limit)
    return total_norm.detach() if as_tensor else float(total_norm)
