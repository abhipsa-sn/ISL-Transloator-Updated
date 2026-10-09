"""
inspect_i_quality.py
====================
Inspection-only script.
Compares the quality of the 'I' dynamic dataset with the 'Sorry' dataset.

Read-only: this script never writes, modifies, or deletes any project file.

Usage:
    python inspect_i_quality.py
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Paths (read-only)
# ---------------------------------------------------------------------------
I_PATH     = r"data/raw/I/dynamic_data.npy"
SORRY_PATH = r"data/raw/Sorry/dynamic_data.npy"

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 65)
print("ISL Dataset Quality Inspector  —  I  vs  Sorry")
print("=" * 65)

try:
    i_data = np.load(I_PATH, allow_pickle=True)
    print(f"[OK] Loaded I      : {I_PATH}")
except Exception as e:
    print(f"[ERROR] Could not load I dataset: {e}")
    raise SystemExit(1)

try:
    sorry_data = np.load(SORRY_PATH, allow_pickle=True)
    print(f"[OK] Loaded Sorry  : {SORRY_PATH}")
except Exception as e:
    print(f"[ERROR] Could not load Sorry dataset: {e}")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# 2. Shapes
# ---------------------------------------------------------------------------
print("\n-- 2. Shapes --------------------------------------------------")
print(f"  I     shape : {i_data.shape}")
print(f"  Sorry shape : {sorry_data.shape}")

# Flatten each sample to a 1-D vector for similarity calculations.
# Handles both 2-D (samples x features) and 3-D (samples x frames x features).
def flatten_samples(arr: np.ndarray) -> np.ndarray:
    """Return array of shape (n_samples, n_features) as float64."""
    arr = arr.astype(np.float64)
    if arr.ndim == 1:
        # Array of objects (ragged) -- flatten each element individually.
        rows = []
        for sample in arr:
            rows.append(np.array(sample, dtype=np.float64).ravel())
        # Pad to the same length so we can stack.
        max_len = max(r.size for r in rows)
        padded  = np.zeros((len(rows), max_len), dtype=np.float64)
        for idx, r in enumerate(rows):
            padded[idx, : r.size] = r
        return padded
    elif arr.ndim == 2:
        return arr
    else:
        # e.g. (samples, frames, features) -> (samples, frames*features)
        return arr.reshape(arr.shape[0], -1)

i_flat     = flatten_samples(i_data)
sorry_flat = flatten_samples(sorry_data)

n_i, feat_i         = i_flat.shape
n_sorry, feat_sorry = sorry_flat.shape
print(f"\n  I     flattened : {n_i} samples x {feat_i} features")
print(f"  Sorry flattened : {n_sorry} samples x {feat_sorry} features")

# ---------------------------------------------------------------------------
# 3. Zero / blank frames
# ---------------------------------------------------------------------------
print("\n-- 3. Zero / Blank Frame Count --------------------------------")

def count_blank_frames(arr: np.ndarray) -> dict:
    """
    Works on either the raw (possibly 3-D) array or an object array.
    A 'blank frame' is a row of all zeros (or near-zero, tol=1e-6).
    """
    tol = 1e-6
    if arr.ndim == 3:
        # shape: (samples, frames, features)
        norms      = np.linalg.norm(arr, axis=-1)          # (samples, frames)
        blank_mask = norms < tol
        total_frames = arr.shape[0] * arr.shape[1]
        blank_count  = int(blank_mask.sum())
        blank_samples = int((blank_mask.any(axis=1)).sum())
    elif arr.ndim == 2:
        norms         = np.linalg.norm(arr, axis=-1)        # treat each row as a frame
        blank_count   = int((norms < tol).sum())
        total_frames  = arr.shape[0]
        blank_samples = blank_count                          # each row = one sample
    else:
        # object array -- check flattened
        flat = flatten_samples(arr)
        norms         = np.linalg.norm(flat, axis=-1)
        blank_count   = int((norms < tol).sum())
        total_frames  = flat.shape[0]
        blank_samples = blank_count
    return {
        "total_frames"   : total_frames,
        "blank_frames"   : blank_count,
        "blank_pct"      : 100.0 * blank_count / max(total_frames, 1),
        "samples_w_blank": blank_samples if arr.ndim == 3 else blank_count,
    }

i_blank     = count_blank_frames(i_data)
sorry_blank = count_blank_frames(sorry_data)

print(f"  I     -- total frames : {i_blank['total_frames']:>6}  |  "
      f"blank : {i_blank['blank_frames']:>4}  ({i_blank['blank_pct']:.1f}%)")
print(f"  Sorry -- total frames : {sorry_blank['total_frames']:>6}  |  "
      f"blank : {sorry_blank['blank_frames']:>4}  ({sorry_blank['blank_pct']:.1f}%)")

# ---------------------------------------------------------------------------
# 4. Within-class cosine similarity of I sequences to their mean
# ---------------------------------------------------------------------------
print("\n-- 4. Within-class Cosine Similarity  (I -> mean-I) -----------")

def safe_cosine_to_mean(flat: np.ndarray):
    """Return per-sample cosine similarity to the class mean."""
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)           # avoid /0
    normed   = flat / norms
    mean_vec = flat.mean(axis=0, keepdims=True)
    mean_norm = np.linalg.norm(mean_vec)
    if mean_norm == 0:
        mean_norm = 1e-10
    mean_vec_normed = mean_vec / mean_norm
    sims = (normed @ mean_vec_normed.T).ravel()           # (n,)
    return sims

i_self_sims = safe_cosine_to_mean(i_flat)

print(f"  Mean  : {i_self_sims.mean():.4f}")
print(f"  Std   : {i_self_sims.std():.4f}")
print(f"  Min   : {i_self_sims.min():.4f}  (sample {i_self_sims.argmin()})")
print(f"  Max   : {i_self_sims.max():.4f}  (sample {i_self_sims.argmax()})")

# ---------------------------------------------------------------------------
# 5. Cosine similarity between each I sequence and the mean Sorry sequence
# ---------------------------------------------------------------------------
print("\n-- 5. I vs Mean-Sorry Cosine Similarity -----------------------")

def align_features(a: np.ndarray, b: np.ndarray):
    """Pad the shorter dimension with zeros to make both the same width."""
    fa, fb = a.shape[1], b.shape[1]
    if fa == fb:
        return a, b
    target = max(fa, fb)
    def pad(arr, to):
        if arr.shape[1] < to:
            return np.hstack([arr, np.zeros((arr.shape[0], to - arr.shape[1]))])
        return arr[:, :to]
    return pad(a, target), pad(b, target)

i_aligned, sorry_aligned = align_features(i_flat, sorry_flat)

sorry_mean = sorry_aligned.mean(axis=0, keepdims=True)    # (1, features)
i_vs_sorry = cosine_similarity(i_aligned, sorry_mean).ravel()  # (n_i,)

print(f"  Mean  : {i_vs_sorry.mean():.4f}")
print(f"  Std   : {i_vs_sorry.std():.4f}")
print(f"  Min   : {i_vs_sorry.min():.4f}  (sample {i_vs_sorry.argmin()})")
print(f"  Max   : {i_vs_sorry.max():.4f}  (sample {i_vs_sorry.argmax()})")

# ---------------------------------------------------------------------------
# 6. Top-10 most Sorry-like I sequences
# ---------------------------------------------------------------------------
print("\n-- 6. Top-10 Most Sorry-like I Sequences ----------------------")
top10_idx  = np.argsort(i_vs_sorry)[::-1][:10]
print(f"  {'Rank':<5}  {'Sample':>8}  {'Sim to Sorry':>14}  {'Sim to Mean-I':>15}")
print("  " + "-" * 50)
for rank, idx in enumerate(top10_idx, 1):
    print(f"  {rank:<5}  {idx:>8}  {i_vs_sorry[idx]:>14.4f}  {i_self_sims[idx]:>15.4f}")

# ---------------------------------------------------------------------------
# 7. Top-5 least internally consistent I sequences
# ---------------------------------------------------------------------------
print("\n-- 7. Top-5 Least Internally Consistent I Sequences ----------")
print("   (lowest cosine similarity to the mean-I vector)")
bottom5_idx = np.argsort(i_self_sims)[:5]
print(f"  {'Rank':<5}  {'Sample':>8}  {'Sim to Mean-I':>15}  {'Sim to Sorry':>14}")
print("  " + "-" * 50)
for rank, idx in enumerate(bottom5_idx, 1):
    print(f"  {rank:<5}  {idx:>8}  {i_self_sims[idx]:>15.4f}  {i_vs_sorry[idx]:>14.4f}")

# ---------------------------------------------------------------------------
# 8. Hand-slot usage for I
# ---------------------------------------------------------------------------
print("\n-- 8. Hand-Slot Usage for I -----------------------------------")
print("   (assumes feature vector layout: [left_hand(21*3), right_hand(21*3), ...])")
print("   A slot is considered 'active' if its L2-norm > 1e-6 for that sample.")

# Standard MediaPipe hand landmark layout (21 keypoints x 3 coords = 63 features).
HAND_FEATURES = 63

def slot_activity(flat: np.ndarray, hand_feat: int = HAND_FEATURES):
    """
    Detect which hand slots are non-zero.
    Returns arrays of booleans: left_active, right_active per sample.
    """
    n, f = flat.shape
    tol  = 1e-6

    if f < 2 * hand_feat:
        # Feature vector too small to contain two hands -- report unknown.
        print(f"   [WARN] Feature dim ({f}) < 2x{hand_feat}; "
              f"cannot determine hand slots reliably.")
        # Treat entire vector as 'right hand only' as a fallback.
        norms = np.linalg.norm(flat, axis=1)
        return np.zeros(n, dtype=bool), norms > tol

    left_slice  = flat[:, :hand_feat]
    right_slice = flat[:, hand_feat: 2 * hand_feat]

    left_active  = np.linalg.norm(left_slice,  axis=1) > tol
    right_active = np.linalg.norm(right_slice, axis=1) > tol
    return left_active, right_active

left_active, right_active = slot_activity(i_flat)

only_left  = left_active  & ~right_active
only_right = ~left_active &  right_active
both       = left_active  &  right_active
neither    = ~left_active & ~right_active

print(f"  Left  hand only  : {only_left.sum():>4}  ({100*only_left.mean():.1f}%)")
print(f"  Right hand only  : {only_right.sum():>4}  ({100*only_right.mean():.1f}%)")
print(f"  Both hands       : {both.sum():>4}  ({100*both.mean():.1f}%)")
print(f"  Neither (blank)  : {neither.sum():>4}  ({100*neither.mean():.1f}%)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n-- Summary ----------------------------------------------------")
print(f"  I samples                    : {n_i}")
print(f"  Sorry samples                : {n_sorry}")
print(f"  I blank-frame rate           : {i_blank['blank_pct']:.1f}%")
print(f"  I self-consistency (mean sim): {i_self_sims.mean():.4f} +/- {i_self_sims.std():.4f}")
print(f"  I vs Sorry similarity (mean) : {i_vs_sorry.mean():.4f} +/- {i_vs_sorry.std():.4f}")

flag_sorry = (i_vs_sorry > 0.9).sum()
flag_incon = (i_self_sims < 0.5).sum()
if flag_sorry > 0:
    print(f"\n  [!] {flag_sorry} I sample(s) have >0.90 similarity to mean-Sorry -- "
          f"consider reviewing them.")
if flag_incon > 0:
    print(f"  [!] {flag_incon} I sample(s) have <0.50 self-consistency -- "
          f"consider re-recording.")
if flag_sorry == 0 and flag_incon == 0:
    print("\n  [OK] No obvious outliers detected.")

print("\n[Done] Inspection complete. No files were modified.\n")
