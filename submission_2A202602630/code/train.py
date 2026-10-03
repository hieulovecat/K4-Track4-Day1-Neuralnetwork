"""train.py — đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.

Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).
Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, build_scheduler, clip_gradients

N_CLASSES = 7

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    scheduler=None,            # None | "cosine" | "warmup_cosine" (không thuộc baseline)
    warmup_steps=0,
    seed=1,
)

TRAIN_LOSS_SUBSET = 50_000   # train_loss đo trên 50 000 mẫu train CỐ ĐỊNH (giống nhau ở mọi thí nghiệm)
SUBSET_SEED = 0              # tách riêng khỏi cfg["seed"] để mọi lần chạy dùng đúng một tập con
DIVERGE_CHECK_EVERY = 50     # kiểm tra NaN/inf mỗi 50 bước (kiểm tra mỗi bước sẽ buộc CPU chờ GPU)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, k: int = N_CLASSES) -> np.ndarray:
    """Ma trận nhầm lẫn (k, k), hàng = nhãn thật, cột = dự đoán (giống scripts/evaluate.py)."""
    idx = y_true.to(torch.int64) * k + y_pred.to(torch.int64)
    return torch.bincount(idx, minlength=k * k).reshape(k, k).cpu().numpy()


def per_class_scores(cm: np.ndarray):
    """precision, recall, F1 từng lớp; bằng 0 khi mẫu số bằng 0 (giống scripts/evaluate.py)."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return prec, rec, f1


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0."""
    return float(per_class_scores(cm)[2].mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits, ở chế độ eval() và FP32."""
    model.eval()
    preds = [model(X[i:i + batch_size]).argmax(dim=1) for i in range(0, len(X), batch_size)]
    return torch.cat(preds)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192, return_cm: bool = False) -> dict:
    """dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt), no_grad, FP32.

    loss = tổng loss từng mẫu / N (không phải trung bình của trung bình lô, vì lô cuối nhỏ hơn).
    """
    model.eval()
    loss_sum = torch.zeros((), dtype=torch.float64, device=X.device)
    preds = []
    for i in range(0, len(X), batch_size):
        logits = model(X[i:i + batch_size]).float()
        yb = y[i:i + batch_size]
        loss_sum += compute_loss(logits, yb, loss_name, reduction="sum").double()
        preds.append(logits.argmax(dim=1))
    pred = torch.cat(preds)
    cm = confusion_matrix(y, pred)
    out = {
        "loss": float(loss_sum) / len(X),
        "acc": float(np.trace(cm) / cm.sum()),
        "macro_f1": macro_f1_from_confusion(cm),
    }
    if return_cm:
        out["cm"] = cm
    return out


def compute_loss(logits, y, loss_name: str, reduction: str = "mean"):
    """"ce"  : cross-entropy trên logit thô và nhãn int64 (softmax nằm TRONG F.cross_entropy).
       "mse" : MSE giữa logit thô và one-hot(y), như nn.MSELoss: không có hệ số 1/2,
               reduction="mean" lấy trung bình trên MỌI phần tử (B·7), không phải chỉ trên B.
               reduction="sum" (dùng trong evaluate) trả tổng / 7, để evaluate chia tiếp cho N
               ra đúng cùng thang với loss lúc huấn luyện.
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    if loss_name == "mse":
        target = F.one_hot(y, N_CLASSES).to(logits.dtype)
        loss = F.mse_loss(logits, target, reduction=reduction)
        return loss / N_CLASSES if reduction == "sum" else loss
    raise ValueError(f"loss phải là 'ce' hoặc 'mse', nhận {loss_name!r}")


def _autocast_ctx(device_type: str, precision: str):
    """Chỉ bọc forward + loss. Tham số vẫn FP32; autocast chỉ hạ độ chính xác của phép toán."""
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device_type, dtype=dtype)


def _fixed_train_subset(n: int, device) -> torch.Tensor:
    g = torch.Generator().manual_seed(SUBSET_SEED)
    idx = torch.randperm(n, generator=g)[:min(TRAIN_LOSS_SUBSET, n)]
    return idx.to(device)


def _sync(device_type: str) -> None:
    if device_type == "cuda":
        torch.cuda.synchronize()


def run_experiment(cfg: dict, data: dict, verbose: bool = True) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG); khoá thiếu lấy từ DEFAULT_CFG
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": mỗi epoch một phần tử: train_loss (eval mode, tập con cố định), train_acc,
                    train_loss_running (trung bình loss các lô lúc huấn luyện, có dropout),
                    val_loss, val_acc, val_macro_f1, grad_norm (trung bình, TRƯỚC clip), grad_norm_max,
                    clip_frac (tỉ lệ bước bị cắt), lr, epoch_time_s,
         "summary": step0_loss, best_val_loss, best_epoch, final_train_loss, final_val_loss,
                    val_acc, val_macro_f1 (ở best_epoch), time_per_epoch_s, peak_mem_MB, diverged, ...,
         "best_state": state_dict (trên CPU) của epoch có val_loss thấp nhất}
    X_eval KHÔNG được dùng trong hàm này: mọi lựa chọn (best epoch, ...) chỉ dựa vào val.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    cfg["hidden"] = tuple(cfg["hidden"])
    X_tr, y_tr, X_val, y_val = data["X_tr"], data["y_tr"], data["X_val"], data["y_val"]
    device = X_tr.device
    dev_type = device.type
    precision = cfg["precision"]
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError(f"precision không hợp lệ: {precision!r}")
    if precision == "fp16" and dev_type != "cuda":
        raise RuntimeError("fp16 + GradScaler cần GPU CUDA")

    # ---- 0. model, optimizer
    set_seed(cfg["seed"])
    model = MLP(hidden=cfg["hidden"], dropout=cfg["dropout"], init=cfg["init"]).to(device)
    n_params = count_params(model)
    if cfg["hidden"] in EXPECTED_PARAMS:
        assert n_params == EXPECTED_PARAMS[cfg["hidden"]], f"{cfg['hidden']}: {n_params} tham số"

    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"], momentum=cfg["momentum"])
    steps_per_epoch = math.ceil(len(X_tr) / cfg["batch"])
    total_steps = steps_per_epoch * cfg["epochs"]
    scheduler = build_scheduler(optimizer, cfg["scheduler"], total_steps, warmup_steps=cfg["warmup_steps"])
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])  # thứ tự xáo lô phụ thuộc seed
    sub = _fixed_train_subset(len(X_tr), device)
    X_sub, y_sub = X_tr[sub], y_tr[sub]
    if dev_type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ---- 1. loss bước 0, TRƯỚC bước cập nhật đầu tiên
    step0 = evaluate(model, X_val, y_val, cfg["loss"])
    if verbose:
        print(f"[{cfg['exp_id']}] {n_params} tham số, {steps_per_epoch} bước/epoch, "
              f"loss bước 0 = {step0['loss']:.4f} (ln 7 = {math.log(N_CLASSES):.4f})")

    keys = ["epoch", "train_loss", "train_acc", "train_loss_running", "val_loss", "val_acc",
            "val_macro_f1", "grad_norm", "grad_norm_max", "clip_frac", "lr", "epoch_time_s"]
    hist = {k: [] for k in keys}
    best_val_loss, best_epoch, best_state = float("inf"), 0, None
    diverged, amp_skipped = False, 0
    all_gn = []

    # ---- 2. huấn luyện
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        lr_now = optimizer.param_groups[0]["lr"]
        _sync(dev_type)
        t0 = time.perf_counter()
        gns, losses = [], []
        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], gen):
            with _autocast_ctx(dev_type, precision):
                logits = model(xb)
            loss = compute_loss(logits.float(), yb, cfg["loss"])  # loss tính bằng FP32 cho ổn định

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # đưa gradient về thang thật TRƯỚC khi đo chuẩn/clip
            else:
                loss.backward()
            gn = clip_gradients(model.parameters(), cfg["clip_norm"], as_tensor=True)
            if scaler is not None:
                scaler.step(optimizer)      # tự bỏ qua bước nếu gradient có inf/NaN
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

            gns.append(gn)
            losses.append(loss.detach())
            if len(losses) % DIVERGE_CHECK_EVERY == 0 and not torch.isfinite(
                    torch.stack(losses[-DIVERGE_CHECK_EVERY:])).all():
                diverged = True
                break
        _sync(dev_type)
        epoch_time = time.perf_counter() - t0

        loss_t = torch.stack(losses)
        if not torch.isfinite(loss_t).all():
            diverged = True
        gn_np = torch.stack(gns).float().cpu().numpy()
        finite = np.isfinite(gn_np)
        if scaler is not None:
            amp_skipped += int((~finite).sum())  # bước FP16 bị GradScaler bỏ qua (gradient tràn số)
        all_gn.append(gn_np[finite])
        clip_frac = float((gn_np[finite] > cfg["clip_norm"]).mean()) if cfg["clip_norm"] and finite.any() else 0.0

        tr = evaluate(model, X_sub, y_sub, cfg["loss"])  # eval mode: so sánh được với val
        va = evaluate(model, X_val, y_val, cfg["loss"])
        row = dict(epoch=epoch, train_loss=tr["loss"], train_acc=tr["acc"],
                   train_loss_running=float(loss_t[torch.isfinite(loss_t)].mean()) if torch.isfinite(loss_t).any() else float("nan"),
                   val_loss=va["loss"], val_acc=va["acc"], val_macro_f1=va["macro_f1"],
                   grad_norm=float(gn_np[finite].mean()) if finite.any() else float("nan"),
                   grad_norm_max=float(gn_np[finite].max()) if finite.any() else float("nan"),
                   clip_frac=clip_frac, lr=lr_now, epoch_time_s=epoch_time)
        for k in keys:
            hist[k].append(row[k])

        if math.isfinite(va["loss"]) and va["loss"] < best_val_loss:
            best_val_loss, best_epoch = va["loss"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(f"  ep {epoch:2d}  train {tr['loss']:.4f}  val {va['loss']:.4f}  acc {va['acc']:.4f}  "
                  f"F1 {va['macro_f1']:.4f}  |g| {row['grad_norm']:.3f} (max {row['grad_norm_max']:.2f})  "
                  f"{epoch_time:.1f}s" + ("  DIVERGED" if diverged else ""))
        if diverged or not math.isfinite(va["loss"]):
            diverged = True
            break

    # ---- 3. tóm tắt (val_acc, val_macro_f1 lấy ở best_epoch: hoạt động như dừng sớm)
    bi = best_epoch - 1
    all_gn = np.concatenate(all_gn) if all_gn else np.array([])
    summary = dict(
        step0_loss=step0["loss"],
        best_val_loss=best_val_loss if best_epoch else float("nan"),
        best_epoch=best_epoch,
        final_train_loss=hist["train_loss"][-1],
        final_val_loss=hist["val_loss"][-1],
        val_acc=hist["val_acc"][bi] if best_epoch else float("nan"),
        val_macro_f1=hist["val_macro_f1"][bi] if best_epoch else float("nan"),
        final_val_acc=hist["val_acc"][-1],
        final_val_macro_f1=hist["val_macro_f1"][-1],
        time_per_epoch_s=float(np.mean(hist["epoch_time_s"])),
        peak_mem_MB=torch.cuda.max_memory_allocated(device) / 2**20 if dev_type == "cuda" else None,
        diverged=diverged,
        epochs_run=len(hist["epoch"]),
        n_params=n_params,
        steps_per_epoch=steps_per_epoch,
        grad_norm_median=float(np.median(all_gn)) if all_gn.size else float("nan"),
        grad_norm_p90=float(np.percentile(all_gn, 90)) if all_gn.size else float("nan"),
        clip_frac=float((all_gn > cfg["clip_norm"]).mean()) if cfg["clip_norm"] and all_gn.size else 0.0,
        amp_skipped_steps=amp_skipped,
        device=torch.cuda.get_device_name(device) if dev_type == "cuda" else dev_type,
    )
    if verbose:
        print(f"[{cfg['exp_id']}] best epoch {best_epoch}: val_loss {summary['best_val_loss']:.4f}  "
              f"val_acc {summary['val_acc']:.4f}  val_macro_f1 {summary['val_macro_f1']:.4f}  "
              f"({summary['time_per_epoch_s']:.1f}s/epoch)")
    return {"cfg": cfg, "history": hist, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`, nhãn 0..6."""
    row_id = np.asarray(row_id).astype(np.int64)
    preds = np.asarray(preds).astype(np.int64)
    assert row_id.shape == preds.shape, f"{row_id.shape} vs {preds.shape}"
    assert len(np.unique(row_id)) == len(row_id), "row_id bị trùng"
    assert preds.min() >= 0 and preds.max() <= N_CLASSES - 1, "pred phải trong 0..6"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame({"row_id": row_id, "pred": preds}).to_csv(path, index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str,
               repo_root: str | None = None, out_json: str | None = None) -> dict | None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Nếu có repo_root: chạy luôn `scripts/evaluate.py --pred <pred_path> [--out <out_json>]`,
    in kết quả và trả về dict đọc từ out_json (nếu có).
    """
    cfg = {**DEFAULT_CFG, **cfg}
    device = data["X_eval"].device
    model = MLP(hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"], init="default").to(device)
    model.load_state_dict(result["best_state"])
    preds = predict(model, data["X_eval"])  # FP32, eval mode
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    print(f"đã ghi {pred_path} ({len(preds)} dòng)")

    if repo_root is None:
        return None
    cmd = [sys.executable, "scripts/evaluate.py", "--pred", os.path.abspath(pred_path)]
    if out_json:
        cmd += ["--out", os.path.abspath(out_json)]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}  # script in tiếng Việt; tránh lỗi mã hoá trên Windows
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, encoding="utf-8", env=env)
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"evaluate.py lỗi:\n{proc.stdout}\n{proc.stderr}")
    if out_json:
        with open(out_json, encoding="utf-8") as f:
            return json.load(f)
    return None
