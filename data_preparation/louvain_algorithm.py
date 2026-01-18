"""
    Author: Wu Nan, Time: 2025-03-16
    Louvain Clustering for Habitat Generation from Supervoxels
"""


import os
import sys
import numpy as np
import nibabel as nib
import pandas as pd
from sknetwork.clustering import Louvain
from pathlib import Path
from tqdm import tqdm
from time import time
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.impute import SimpleImputer
from scipy.ndimage import median_filter

# SV parameters
COMPACT = 0.1
TARGET_SV_SIZE = 1000
# Louvain parameters
RESOLUTION = 1.1
EXP_NUMBER = 999  # a random number to distinguish experiments, u'd better change it to real habitat number
KNN_K = 30
SPATIAL_WEIGHT = 10.0
PCA_VARIANCE = 0.95
SV_DIR = Path(rf'..\Liver_SV_C{COMPACT}_N{TARGET_SV_SIZE}')
SP_DIR = SV_DIR / 'Superpixels'
RES_DIR = Path(rf'../Habitat_Louvain_K{KNN_K}_Res{RESOLUTION}_EXP{EXP_NUMBER}')
RES_DIR.mkdir(exist_ok=True, parents=True)
NII_DIR = RES_DIR / 'NIfTI_Maps'
NII_DIR.mkdir(exist_ok=True)
VIS_REF_DIR = Path(r'..\Origin_NII_Data')
MAX_WORKERS = max(1, os.cpu_count() - 2)


def calculate_centroids(args):
    npz_path, cid = args
    try:
        data = np.load(npz_path)
        indices = data['indices']
        idxCL = data['idxCL']
        shape = data['imgSize']
        x_arr, y_arr, z_arr = np.unravel_index(indices, shape)
        df = pd.DataFrame({'sv': idxCL, 'x': x_arr, 'y': y_arr, 'z': z_arr})
        centroids_df = df.groupby('sv').mean()
        n_sp = data['label'].shape[0]
        centroids_df = centroids_df.reindex(range(1, n_sp + 1))
        return centroids_df[['x', 'y', 'z']].values.astype(np.float32), cid
    except Exception as e:
        return None, f"{cid}: {e}"


def prepare_data():
    feat_file = SV_DIR / 'allSuperpixels.npy'
    if not feat_file.exists():
        sys.exit(f"No features found at {feat_file}\n")

    X_feats = np.load(feat_file)
    if X_feats.shape[1] == 40:
        print(f"Feature Shape: {X_feats.shape}")
    else:
        print(f"Warning: Feature dim is {X_feats.shape[1]}. Expected 40.")
    centroid_cache = SV_DIR / 'allCentroids.npy'
    if centroid_cache.exists():
        print(" Loading cached centroids...")
        X_coords = np.load(centroid_cache)
    else:
        print(" Calculating centroids (One-time)...")
        case_ids = np.load(SV_DIR / 'case_list.npy')
        npz_files = [SP_DIR / f'{cid}.npz' for cid in case_ids]
        coords_list = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(tqdm(executor.map(calculate_centroids, zip(npz_files, case_ids)), total=len(case_ids)))
        for res, err in results:
            if res is None: sys.exit(f"Centroid Error: {err}")
            coords_list.append(res)
        X_coords = np.vstack(coords_list)
        np.save(centroid_cache, X_coords)
    return X_feats, X_coords


def run_louvain_pipeline(X_feats, X_coords):
    print("\n Graph Construction (PCA + Spatial)...")
    imputer = SimpleImputer(strategy='median')
    X_filled = imputer.fit_transform(X_feats)
    X_scaled = RobustScaler().fit_transform(X_filled)
    pca = PCA(n_components=PCA_VARIANCE, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f" PCA Reduced: {X_feats.shape[1]} dims -> {X_pca.shape[1]} dims")
    scaler_coord = MinMaxScaler()
    X_coords_norm = scaler_coord.fit_transform(X_coords)
    X_final = np.hstack([X_pca, X_coords_norm * SPATIAL_WEIGHT])
    print(f"   Building k-NN Graph (k={KNN_K})...")
    t0 = time()
    A = kneighbors_graph(X_final, KNN_K, mode='distance', include_self=False, n_jobs=-1)
    max_dist = A.max()
    A.data = max_dist - A.data
    A = A.maximum(A.T).tocsr()
    A.eliminate_zeros()
    # print(f"   Graph built in {time() - t0:.1f}s. Nodes: {A.shape[0]}")
    print(f"\n Running Louvain (Resolution={RESOLUTION})...")
    t0 = time()
    louvain = Louvain(
        resolution=RESOLUTION,
        modularity="newman",
        shuffle_nodes=False,
        return_probs=False,
        return_aggregate=False,
        random_state=42
    )
    labels = louvain.fit_predict(A).astype(np.int32)
    u, c = np.unique(labels, return_counts=True)
    print("   Cluster Counts:", dict(zip(u, c)))
    return labels


def save_nifti_worker(args):
    cid, start_idx, count, labels_global = args
    npz_path = SP_DIR / f'{cid}.npz'
    if not npz_path.exists(): return
    ref_dir = VIS_REF_DIR / cid
    if not ref_dir.exists():
        cands = [d for d in VIS_REF_DIR.iterdir() if d.name.lower() == cid.lower()]
        if cands:
            ref_dir = cands[0]
        else:
            return f"No ref directory for {cid}"
    ref_file = next(ref_dir.glob('*LABEL*.nii*'), None) or next(ref_dir.glob('*T1*.nii*'), None)
    if not ref_file:
        return f"No reference nii found in {ref_dir}"
    try:
        ref = nib.load(str(ref_file))
        data = np.load(npz_path)
        idxCL = data['idxCL']
        indices = data['indices']
        shape = data['imgSize']
        case_labels = labels_global[start_idx: start_idx + count]
        mapper = np.zeros(count + 1, dtype=np.int16)
        mapper[1:] = case_labels + 1
        vol = np.zeros(shape, dtype=np.int16)
        vol.ravel()[indices] = mapper[idxCL]
        mask_liver = (vol > 0)
        if mask_liver.any():
            vol_smooth = median_filter(vol, size=3)
            vol = np.where(mask_liver, vol_smooth, 0)
        out_name = f'{cid}_Habitats_K{KNN_K}_Res{RESOLUTION}.nii.gz'
        nib.save(nib.Nifti1Image(vol, ref.affine, ref.header), str(NII_DIR / out_name))
        return None
    except Exception as e:
        import traceback
        traceback.print_exc()
        return str(e)


def calculate_and_print_stats(labels, idx_counts, case_ids):
    print("\n" + "=" * 60)
    print(f" Habitat Statistics Report (Including Entropy Features)")
    print("=" * 60)
    unique_labels, sv_counts = np.unique(labels, return_counts=True)
    total_svs = len(labels)
    n_patients = len(case_ids)
    patient_occurrence = {lbl: 0 for lbl in unique_labels}
    current_ptr = 0
    for count in idx_counts:
        patient_labels = labels[current_ptr: current_ptr + count]
        unique_in_patient = np.unique(patient_labels)
        for lbl in unique_in_patient:
            patient_occurrence[lbl] += 1
        current_ptr += count
    stats_data = []
    for i, lbl in enumerate(unique_labels):
        sv_cnt = sv_counts[i]
        pat_cnt = patient_occurrence[lbl]
        pat_pct = (pat_cnt / n_patients) * 100
        stats_data.append({
            'Habitat_ID': lbl + 1,
            'Total_SVs': sv_cnt,
            'SV_Ratio': f"{sv_cnt / total_svs * 100:.2f}%",
            'Patient_Count': pat_cnt,
            'Prevalence': f"{pat_pct:.1f}%"
        })
    df = pd.DataFrame(stats_data)
    df = df.sort_values(by='Total_SVs', ascending=False)
    print(df.to_string(index=False))
    print("-" * 60)
    csv_path = RES_DIR / 'habitat_stats.csv'
    df.to_csv(csv_path, index=False)
    print(f"Statistics saved to: {csv_path}")


# ================= 主程序 =================
if __name__ == '__main__':
    if not SV_DIR.exists():
        print(f" Error: Input directory not found: {SV_DIR}")
        sys.exit(1)
    X_feats, X_coords = prepare_data()
    labels = run_louvain_pipeline(X_feats, X_coords)
    np.save(RES_DIR / 'louvain_labels.npy', labels)
    print("\n Writing Smoothed NIfTI Maps...")
    idx_counts = np.load(SV_DIR / 'idx_count.npy')
    case_ids = np.load(SV_DIR / 'case_list.npy')
    tasks = []
    curr = 0
    for cid, cnt in zip(case_ids, idx_counts):
        tasks.append((cid, curr, cnt, labels))
        curr += cnt
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(save_nifti_worker, tasks), total=len(tasks)))
    errors = [e for e in results if e is not None]
    if errors:
        print(f" {len(errors)} errors during NIfTI saving:")
        for e in errors[:5]: print(f"  - {e}")
    calculate_and_print_stats(labels, idx_counts, case_ids)
    print(f"\n All Done! Results in: {NII_DIR}")