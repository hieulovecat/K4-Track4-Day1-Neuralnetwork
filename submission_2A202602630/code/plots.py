"""plots.py — ảnh từng thí nghiệm (figures/<exp_id>.png) và ảnh chồng theo nhóm (figures/compare_<nhóm>.png).

Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
Hàm nhận được cả `result` vừa trả về từ run_experiment lẫn dict đọc lại từ results/<exp_id>.json.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

METRIC_LABELS = {
    "train_loss": "train loss (eval mode)",
    "val_loss": "val loss",
    "val_acc": "val accuracy",
    "val_macro_f1": "val macro-F1",
    "train_acc": "train accuracy",
    "grad_norm": "grad_norm trung bình (trước clip)",
    "grad_norm_max": "grad_norm lớn nhất (trước clip)",
    "epoch_time_s": "thời gian / epoch (s)",
    "lr": "learning rate",
}


def cfg_label(cfg: dict) -> str:
    """Tóm tắt cấu hình trên một dòng, dùng làm tiêu đề ảnh."""
    hidden = "-".join(str(h) for h in cfg.get("hidden", ()))
    parts = [f"{cfg.get('optimizer')} lr={cfg.get('lr'):g}" if cfg.get("lr") is not None else str(cfg.get("optimizer")),
             f"loss={cfg.get('loss')}", f"batch={cfg.get('batch')}", f"hidden={hidden}",
             f"init={cfg.get('init')}", f"drop={cfg.get('dropout')}", f"clip={cfg.get('clip_norm')}",
             f"{cfg.get('precision')}", f"seed={cfg.get('seed')}"]
    if cfg.get("weight_decay"):
        parts.insert(1, f"wd={cfg['weight_decay']:g}")
    if cfg.get("scheduler"):
        parts.insert(1, f"sched={cfg['scheduler']}")
    return ", ".join(parts)


def _arr(values) -> np.ndarray:
    """None (NaN đã lưu trong JSON) -> nan, để matplotlib bỏ qua điểm đó."""
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _fmt(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_run(result: dict, path: str) -> None:
    """Một thí nghiệm -> một ảnh PNG gồm 3 ô:
         (1) train_loss và val_loss theo epoch
         (2) val_acc và val_macro_f1 theo epoch
         (3) grad_norm (trung bình và lớn nhất trong epoch, đo TRƯỚC khi clip) + ngưỡng clip nếu có
    Đường đứt dọc = best_epoch (val_loss thấp nhất).
    """
    cfg, h, s = result["cfg"], result["history"], result["summary"]
    ep = np.array(h["epoch"])
    best = s.get("best_epoch") or 0

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    ax = axes[0]
    ax.plot(ep, _arr(h["train_loss"]), "o-", ms=3, label="train (eval mode, 50k mẫu cố định)")
    ax.plot(ep, _arr(h["val_loss"]), "s-", ms=3, label="val")
    ax.set_title("Loss")
    ax.set_ylabel(f"loss ({cfg.get('loss', 'ce').upper()})")

    ax = axes[1]
    ax.plot(ep, _arr(h["val_acc"]), "o-", ms=3, label="val accuracy")
    ax.plot(ep, _arr(h["val_macro_f1"]), "s-", ms=3, label="val macro-F1")
    ax.axhline(0.4876, color="gray", lw=0.8, ls=":", label="đoán lớp đa số (acc 0.4876)")
    ax.set_title("Val accuracy / macro-F1")
    ax.set_ylabel("giá trị")

    ax = axes[2]
    gn = _arr(h["grad_norm"])
    gmax = _arr(h["grad_norm_max"]) if "grad_norm_max" in h else None
    ax.plot(ep, gn, "o-", ms=3, label="trung bình epoch")
    if gmax is not None:
        ax.plot(ep, gmax, "^--", ms=3, alpha=0.7, label="lớn nhất epoch")
    if cfg.get("clip_norm"):
        ax.axhline(cfg["clip_norm"], color="red", lw=1, ls="--", label=f"ngưỡng clip c={cfg['clip_norm']:g}")
    finite = np.concatenate([gn[np.isfinite(gn)], gmax[np.isfinite(gmax)] if gmax is not None else []])
    if finite.size and finite.min() > 0 and finite.max() / finite.min() > 50:
        ax.set_yscale("log")  # gai gradient lớn: trục log để vẫn thấy được đường trung bình
    ax.set_title("Chuẩn L2 gradient (trước clip)")
    ax.set_ylabel("‖g‖₂")

    for ax in axes:
        if best:
            ax.axvline(best, color="green", lw=0.8, ls="--", label=f"best epoch = {best}")
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    status = "  [DIVERGED]" if s.get("diverged") else ""
    fig.suptitle(f"{cfg.get('exp_id')}{status} — {cfg_label(cfg)}\n"
                 f"best val_loss {_fmt(s.get('best_val_loss'))} | val_acc {_fmt(s.get('val_acc'))} | "
                 f"val_macro_f1 {_fmt(s.get('val_macro_f1'))} | loss bước 0 {_fmt(s.get('step0_loss'))}",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def plot_compare(results: list[dict], metric, path: str, title: str = "") -> None:
    """Vẽ chồng một (hoặc vài) chỉ số của nhiều thí nghiệm, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    metric: tên một chỉ số trong history (ví dụ "val_loss") hoặc list tên, mỗi chỉ số một ô.
    Dùng cho figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    metrics = [metric] if isinstance(metric, str) else list(metric)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.6 * len(metrics), 4.4), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        all_vals = []
        for r in results:
            vals = _arr(r["history"][m])
            all_vals.append(vals[np.isfinite(vals)])
            label = r["cfg"]["exp_id"] + (" (diverged)" if r["summary"].get("diverged") else "")
            ax.plot(r["history"]["epoch"], vals, "o-", ms=2.5, label=label)
        vals = np.concatenate(all_vals) if all_vals else np.array([])
        if m.startswith("grad_norm") and vals.size and vals.min() > 0 and vals.max() / vals.min() > 50:
            ax.set_yscale("log")
        ax.set_title(METRIC_LABELS.get(m, m))
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel(METRIC_LABELS.get(m, m))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    _save(fig, path)
