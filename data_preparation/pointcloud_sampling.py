"""
    Author: Wu Nan, Time: 2025-03-21
    CUDA accelerated PC sampling from habitat .npz maps
"""

from pathlib import Path
import numpy as np
import nibabel as nib
import logging
import traceback
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from torch_cluster import fps
import torch
TORCH_AVAILABLE = True
USE_CUDA = True
EXP_NUMBER = 23
KNN_K = 30
RESOLUTION = 1.1
K = EXP_NUMBER
N = 4096
MODALITIES = ('T1WI', 'T2FS')
SPACING = np.array([1.0, 1.0, 5.0], dtype=np.float32)
ORIGIN_ROOT = Path(r'..\Origin_NII_Data')
LOUVAIN_RES_DIR = Path(
    rf'..\Habitat_Louvain_K{KNN_K}_Res{RESOLUTION}_EXP{EXP_NUMBER}')
HAB_NII_DIR = LOUVAIN_RES_DIR / 'NIfTI_Maps'
OUT_DIR = LOUVAIN_RES_DIR / f'PointCloud_Data_K{K}_N{N}'
OUT_DIR.mkdir(exist_ok=True, parents=True)
MAX_WORKERS = max(1, os.cpu_count() - 2)
logging.basicConfig(level=logging.ERROR, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S')


def find_case_dir(cid: str) -> Path:
    p = ORIGIN_ROOT / cid
    if p.exists(): return p
    candidates = list(ORIGIN_ROOT.glob(f'*{cid}*'))
    if candidates: return candidates[0]
    raise FileNotFoundError(f"Cannot find raw data for {cid}")


def load_modalities(case_dir: Path, ref_shape: tuple, mask_idx):
    feats = []
    missing = []
    exts = ['.nii.gz', '.nii']
    for m in MODALITIES:
        f = None
        for ext in exts:
            cands = list(case_dir.glob(f'*{m}*{ext}'))
            if cands:
                f = cands[0];
                break
        if f is None:
            feats.append(np.zeros(len(mask_idx[0]), dtype=np.float32))
            missing.append(1)
        else:
            try:
                arr = nib.load(str(f)).get_fdata().astype(np.float32)
                if arr.shape != ref_shape:
                    if arr.ndim == 4 and arr.shape[:3] == ref_shape:
                        arr = arr[..., 0]
                    else:
                        feats.append(np.zeros(len(mask_idx[0]), dtype=np.float32))
                        missing.append(1)
                        continue
                feats.append(arr[mask_idx])
                missing.append(0)
            except:
                feats.append(np.zeros(len(mask_idx[0]), dtype=np.float32))
                missing.append(1)
    return np.stack(feats, axis=1), np.array(missing, dtype=np.uint8)


def fps_sampling(coords_phys, n_samples):
    P = len(coords_phys)
    if P >= n_samples:
        if TORCH_AVAILABLE:
            try:
                tensor = torch.from_numpy(coords_phys)
                if USE_CUDA and torch.cuda.is_available():
                    tensor = tensor.cuda()
                ratio = n_samples / P
                idx = fps(tensor, ratio=ratio, random_start=True)
                idx = idx.cpu().numpy()
                if len(idx) >= n_samples:
                    return idx[:n_samples]
                else:
                    pad = np.random.choice(idx, n_samples - len(idx), replace=True)
                    return np.concatenate([idx, pad])
            except Exception:
                pass
        sel = np.zeros(n_samples, dtype=np.int64)
        sel[0] = np.random.randint(P)
        dist = np.full(P, np.inf, dtype=np.float32)
        points = coords_phys
        for i in range(1, n_samples):
            last_pt = points[sel[i - 1]]
            d2 = np.sum((points - last_pt) ** 2, axis=1)
            dist = np.minimum(dist, d2)
            sel[i] = np.argmax(dist)
        return sel
    else:
        pad = np.random.choice(P, n_samples - P, replace=True)
        return np.concatenate([np.arange(P), pad])


def process_patient_worker(nii_path):
    try:
        cid = nii_path.name.split('_Habitats')[0]
        out_file = OUT_DIR / f'{cid}.npz'
        if out_file.exists():
            return None
        hab_img = nib.load(str(nii_path))
        hab_arr = hab_img.get_fdata().astype(np.int16)
        mask_idx = np.nonzero(hab_arr)
        if mask_idx[0].size == 0: return f"{cid}: Empty"
        feats_mod, missing_mod = load_modalities(find_case_dir(cid), hab_arr.shape, mask_idx)
        voxel_xyz = np.vstack(mask_idx).T.astype(np.float32)
        labels_all = hab_arr[mask_idx].astype(np.int16)
        phys_xyz = voxel_xyz * SPACING
        clouds_tensor = np.zeros((K, N, 3 + 1 + len(MODALITIES)), dtype=np.float32)
        centroids_vox = np.zeros((K, 3), dtype=np.float32)
        centroids_phys = np.zeros((K, 3), dtype=np.float32)
        missing_hab = np.ones(K, dtype=np.uint8)
        for hid in range(1, K + 1):
            sel = np.where(labels_all == hid)[0]
            if sel.size == 0: continue
            missing_hab[hid - 1] = 0
            v_coords = voxel_xyz[sel]
            p_coords = phys_xyz[sel]
            f_vals = feats_mod[sel]
            c_v = v_coords.mean(0)
            c_p = p_coords.mean(0)
            centroids_vox[hid - 1] = c_v
            centroids_phys[hid - 1] = c_p
            bb_min = v_coords.min(0)
            bb_max = v_coords.max(0)
            center = (bb_min + bb_max) / 2
            scale = (bb_max - bb_min).max() + 1e-8
            norm_xyz = (v_coords - center) / (scale / 2)
            sample_idx = fps_sampling(p_coords, N)
            cloud = np.concatenate([
                norm_xyz[sample_idx],
                np.full((N, 1), hid, dtype=np.float32),
                f_vals[sample_idx]
            ], axis=1)
            clouds_tensor[hid - 1] = cloud
        voxel_dist = np.full((K, K), -1, dtype=np.float32)
        phys_dist = np.full((K, K), -1, dtype=np.float32)
        present = np.where(missing_hab == 0)[0]
        if len(present) > 0:
            cv = centroids_vox[present]
            cp = centroids_phys[present]
            dv = np.sqrt(np.sum((cv[:, None, :] - cv[None, :, :]) ** 2, axis=2))
            dp = np.sqrt(np.sum((cp[:, None, :] - cp[None, :, :]) ** 2, axis=2))
            idx_i, idx_j = np.meshgrid(present, present, indexing='ij')
            voxel_dist[idx_i, idx_j] = dv
            phys_dist[idx_i, idx_j] = dp
            for idx in present:
                voxel_dist[idx, idx] = 0
                phys_dist[idx, idx] = 0
        np.savez_compressed(out_file,
                            clouds=clouds_tensor,
                            centroids_vox=centroids_vox,
                            centroids_phys=centroids_phys,
                            voxel_dist_mat=voxel_dist,
                            phys_dist_mat=phys_dist,
                            missing_habitats=missing_hab,
                            missing_modalities=missing_mod,
                            modalities=np.array(MODALITIES)
                            )
        return None
    except Exception as e:
        return f"{cid}: {str(e)}"


if __name__ == '__main__':
    nii_files = list(HAB_NII_DIR.glob('*.nii.gz'))
    if not nii_files:
        print(f"No NIfTI files in {HAB_NII_DIR}")
        exit()
    print(f" Found {len(nii_files)} maps.")
    print(f" Max Workers: {MAX_WORKERS}")
    print(f"⚡ CUDA Enabled: {USE_CUDA}")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(
            executor.map(process_patient_worker, nii_files),
            total=len(nii_files),
            desc="Building PointClouds"
        ))
    errors = [r for r in results if r is not None]
    if errors:
        print(f"\n  {len(errors)} errors occurred:")
        for e in errors[:5]: print(e)
    else:
        print("\n  All Done!")