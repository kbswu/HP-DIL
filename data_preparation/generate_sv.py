import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path
from time import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from skimage.filters.rank import entropy as entropy_2d
from skimage.morphology import disk
from skimage.segmentation import slic
from scipy.stats import skew, kurtosis
from tqdm import tqdm


COMPACT = 0.1
TARGET_SV_NUM = 1000
ENTROPY_R = 3
INPUT_ROOT = Path(r'../Origin_NII_Data')
RES_DIR = Path(rf'..\Liver_SV_C{COMPACT}_N{TARGET_SV_NUM}')
RES_DIR.mkdir(exist_ok=True, parents=True)
SP_DIR = RES_DIR / 'Superpixels'
SP_DIR.mkdir(exist_ok=True)
VOXEL_SPACING = [1, 1, 5]
MAX_WORKERS = max(1, os.cpu_count() - 2)


def load_nii(path: Path):
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32), img.header


def find_mod(case_dir: Path, base: str) -> Path:
    for e in ('.nii.gz', '.nii', '.NII.GZ', '.NII'):
        p = case_dir / f'{base}{e}'
        if p.exists(): return p
    raise FileNotFoundError(f"{base} not found in {case_dir.name}")


def bbox_3d_indices(mask):
    if mask.sum() == 0: return None
    dims = np.nonzero(mask)
    slices = []
    starts = []
    for d in dims:
        slices.append(slice(d.min(), d.max() + 1))
        starts.append(d.min())
    return tuple(slices), tuple(starts)


def min_max_norm(img: np.ndarray) -> np.ndarray:
    v_min, v_max = img.min(), img.max()
    if v_max - v_min < 1e-8:
        return np.zeros_like(img)
    return (img - v_min) / (v_max - v_min)


def robust_norm_to_01(x, m):
    vals = x[m]
    if vals.size == 0: return x
    p1, p99 = np.percentile(vals, (1, 99))
    x_clip = np.clip(x, p1, p99)
    return min_max_norm(x_clip)


def entropy_3d_paper_strict(img_01: np.ndarray, r: int) -> np.ndarray:
    img_u8 = (img_01 * 255).astype(np.uint8)
    ker = disk(r)
    out = np.empty_like(img_01, dtype=np.float32)
    z_dim = img_01.shape[2]
    for z in range(z_dim):
        out[:, :, z] = entropy_2d(img_u8[:, :, z], footprint=ker)
    return min_max_norm(out)


def calc_stats_10(v: np.ndarray) -> np.ndarray:
    if len(v) == 0: return np.zeros(10, dtype=np.float32)
    v = v.astype(np.float32)
    mean = np.mean(v)
    std = np.std(v)
    q1, q2 = np.quantile(v, (0.25, 0.75))
    return np.array([
        skew(v, bias=False),
        kurtosis(v, bias=False, fisher=False),
        mean, np.median(v), q1, q2, q2 - q1,
        std, np.var(v), np.sum(v ** 2) / len(v)
    ], dtype=np.float32)


def process_single_case(case_dir: Path):
    cid = case_dir.name
    out_npz = SP_DIR / f'{cid}.npz'
    try:
        label_path = find_mod(case_dir, 'LABEL')
        label_img, _ = load_nii(label_path)
        mask = (label_img == 6)
        if mask.sum() == 0: mask = (label_img == 1)
        if mask.sum() == 0: return None, 0, f"{cid}: Empty Mask"
        bbox_res = bbox_3d_indices(mask)
        if bbox_res is None: return None, 0, f"{cid}: Mask empty"
        bb_slices, bb_starts = bbox_res
        msk_crop = mask[bb_slices]
        t1, _ = load_nii(find_mod(case_dir, 'T1WI'))
        t2, _ = load_nii(find_mod(case_dir, 'T2FS'))
        t1_crop = t1[bb_slices]
        t2_crop = t2[bb_slices]
        t1n = robust_norm_to_01(t1_crop, msk_crop)
        t2n = robust_norm_to_01(t2_crop, msk_crop)
        ent1 = entropy_3d_paper_strict(t1n, ENTROPY_R)
        ent2 = entropy_3d_paper_strict(t2n, ENTROPY_R)
        img_4channel = np.stack([t1n, t2n, ent1, ent2], axis=-1)
        lbl = slic(
            img_4channel,
            n_segments=TARGET_SV_NUM,  # 论文设定 1000
            mask=msk_crop,
            compactness=COMPACT,  # 论文设定 0.1
            sigma=0.5,  # 保持轻微平滑
            spacing=VOXEL_SPACING,
            start_label=1,
            enforce_connectivity=True,
            channel_axis=-1
        )
        n_sp = lbl.max()
        valid_indices = np.where(msk_crop)
        valid_feats = img_4channel[valid_indices]
        valid_labels = lbl[valid_indices]
        sp40 = np.zeros((n_sp, 40), dtype=np.float32)
        for i in range(1, n_sp + 1):
            mask_i = (valid_labels == i)
            if not np.any(mask_i): continue
            cluster_vals = valid_feats[mask_i]
            feats = []
            for ch in range(4):
                feats.append(calc_stats_10(cluster_vals[:, ch]))
            sp40[i - 1] = np.hstack(feats)
        gx = valid_indices[0] + bb_starts[0]
        gy = valid_indices[1] + bb_starts[1]
        gz = valid_indices[2] + bb_starts[2]
        full_shape = mask.shape
        flat_indices = np.ravel_multi_index((gx, gy, gz), full_shape)
        np.savez_compressed(
            out_npz,
            features=sp40,
            idxCL=valid_labels,
            indices=flat_indices,
            label=np.arange(1, n_sp + 1, dtype=np.int16),
            imgSize=np.array(full_shape),
            pixelDim=np.array(VOXEL_SPACING)
        )
        return sp40, n_sp, cid

    except Exception as e:
        return None, 0, f"{cid} Err: {str(e)}"


if __name__ == '__main__':
    if not INPUT_ROOT.exists(): sys.exit("ROOT dir not found")
    case_paths = sorted([d for d in INPUT_ROOT.iterdir() if d.is_dir()], key=lambda p: p.name)
    print(f"Processing: {len(case_paths)} cases")
    print(f"Compact={COMPACT}, TargetSV={TARGET_SV_NUM}")
    all40_list, count_list, cid_list, errors = [], [], [], []
    t0 = time()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futs = {executor.submit(process_single_case, p): p for p in case_paths}
        for f in tqdm(as_completed(futs), total=len(case_paths)):
            res, n, info = f.result()
            if res is not None:
                all40_list.append(res)
                count_list.append(n)
                cid_list.append(info)
            else:
                errors.append(info)
    if all40_list:
        np.save(RES_DIR / 'allSuperpixels.npy', np.vstack(all40_list))
        np.save(RES_DIR / 'idx_count.npy', np.array(count_list))
        np.save(RES_DIR / 'case_list.npy', np.array(cid_list))
        # print(f" Done in {(time() - t0) / 60:.1f} min.")
    if errors:
        print(f" {len(errors)} failures.")
        print(errors)