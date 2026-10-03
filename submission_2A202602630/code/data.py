"""data.py — nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)
N_FEATURES = 54
N_CLASSES = 7
N_TRAIN, N_EVAL = 464_809, 116_203


def _check_xy(X, y, n_expected: int, name: str) -> None:
    assert X.shape == (n_expected, N_FEATURES), f"{name}: X có shape {X.shape}"
    assert X.dtype == np.float32, f"{name}: X có dtype {X.dtype}"
    assert y.shape == (n_expected,), f"{name}: y có shape {y.shape}"
    assert y.dtype == np.int64, f"{name}: y có dtype {y.dtype}"
    assert y.min() >= 0 and y.max() <= N_CLASSES - 1, f"{name}: nhãn ngoài khoảng 0..6"


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    """
    tr = np.load(f"{processed_dir}/train.npz")
    ev = np.load(f"{processed_dir}/eval.npz")
    X_train, y_train = tr["X"], tr["y"]
    X_eval, y_eval, eval_row_id = ev["X"], ev["y"], ev["row_id"]

    _check_xy(X_train, y_train, N_TRAIN, "train")
    _check_xy(X_eval, y_eval, N_EVAL, "eval")
    assert len(np.unique(eval_row_id)) == N_EVAL, "eval row_id bị trùng"
    return X_train, y_train, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval), phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Mọi thí nghiệm dùng cùng seed và val_fraction để so sánh công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """mean và std của 10 cột số, tính CHỈ trên phần train còn lại (sau khi tách val).

    Tính trên val/eval là rò rỉ thông tin: phép chuẩn hoá (một phần của "mô hình")
    sẽ đã nhìn thấy dữ liệu mà ta dùng để đánh giá nó.
    """
    num = X_tr[:, :N_NUMERIC].astype(np.float64)  # float64 để tổng của ~370k mẫu không mất chính xác
    mean = num.mean(axis=0)
    std = num.std(axis=0)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Bản sao của X với 10 cột đầu = (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    out = X.copy()
    safe_std = np.where(std > 0, std, 1.0).astype(np.float32)  # cột hằng số: chỉ trừ mean, không chia 0
    out[:, :N_NUMERIC] = (out[:, :N_NUMERIC] - mean) / safe_std
    return out


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed", verbose: bool = True) -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device: X_tr, y_tr, X_val, y_val, X_eval, y_eval
    cùng các mảng numpy: eval_row_id, mean, std
    """
    X_train, y_train, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_train, y_train, val_fraction, seed)

    mean, std = fit_standardizer(X_tr)  # chỉ X_tr; X_val và X_eval không tham gia
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    def to_x(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    def to_y(a):
        return torch.tensor(a, dtype=torch.int64, device=device)

    data = {
        "X_tr": to_x(X_tr), "y_tr": to_y(y_tr),
        "X_val": to_x(X_val), "y_val": to_y(y_val),
        "X_eval": to_x(X_eval), "y_eval": to_y(y_eval),
        "eval_row_id": eval_row_id, "mean": mean, "std": std,
    }

    if verbose:
        print(f"X_tr   {tuple(X_tr.shape)}  X_val {tuple(X_val.shape)}  X_eval {tuple(X_eval.shape)}  (device={device})")
        print("lớp   train(%)   val(%)   eval(%)")
        c_tr = np.bincount(y_tr, minlength=N_CLASSES) / len(y_tr)
        c_val = np.bincount(y_val, minlength=N_CLASSES) / len(y_val)
        c_ev = np.bincount(y_eval, minlength=N_CLASSES) / len(y_eval)
        for c in range(N_CLASSES):
            print(f"{c:>3d}   {100*c_tr[c]:7.3f}  {100*c_val[c]:7.3f}  {100*c_ev[c]:7.3f}")
        # Mốc thấp nhất: lớp đa số chọn theo TRAIN, đo trên VAL
        majority = int(c_tr.argmax())
        print(f"Đoán luôn lớp đa số (lớp {majority}) -> val accuracy = {(y_val == majority).mean():.4f}")
        num = X_tr[:, :N_NUMERIC]
        print(f"10 cột số trên X_tr sau chuẩn hoá: |mean| max = {np.abs(num.mean(0)).max():.2e}, "
              f"std trong [{num.std(0).min():.4f}, {num.std(0).max():.4f}]")
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Batch cuối: GIỮ LẠI dù nhỏ hơn batch_size (371 847 = 726*512 + 135), để mọi mẫu train
    đều được dùng trong mỗi epoch. Với batch 512 thì đó là 727 bước/epoch.
    `generator` phải nằm trên cùng device với X (torch.Generator(device=X.device)).
    """
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
