"""
    from habitat nii to npz dense files for Radiomics analysis.
"""

import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# file path
KNN_K = 30
RESOLUTION = 1.1
EXP_NUMBER = 23  # better indicate real number of hab.
TARGET_SV_SIZE = 1000
COMPACT = 0.1


PREV_STEP_DIR = Path(rf'..\Habitat_Louvain_K{KNN_K}_Res{RESOLUTION}_EXP{EXP_NUMBER}')
MASK_INPUT_DIR = PREV_STEP_DIR / 'NIfTI_Maps'
ORIGIN_ROOT = Path(r'..\HP_DIL_OPENSOURSE\Origin_NII_Data')
OUT_DIR = PREV_STEP_DIR / 'Pixel_Data_NPZ'
OUT_DIR.mkdir(exist_ok=True, parents=True)
HABITAT_SUFFIX = f"_Habitats_K{KNN_K}_Res{RESOLUTION}.nii.gz"
MAX_WORKERS = max(1, os.cpu_count() - 2)


def find_origin_nii(cid, modality):
    case_dir = ORIGIN_ROOT / cid
    if not case_dir.exists(): return None
    cands = [p for p in case_dir.iterdir() if modality.lower() in p.name.lower() and p.name.endswith('.nii.gz')]
    if cands:
        cands.sort(key=lambda x: len(x.name))
        return cands[0]
    return None


def process_single_case(mask_file_path: Path):
    try:
        filename = mask_file_path.name
        if "_Habitats" not in filename:
            return f"Skipped (Naming format): {filename}"
        cid = filename.split('_Habitats')[0]
        t1_path = find_origin_nii(cid, 'T1')
        t2_path = find_origin_nii(cid, 'T2')
        if not t1_path or not t2_path:
            return f"Missing Origin MRI: {cid}"
        img_mask = nib.load(str(mask_file_path))
        img_t1 = nib.load(str(t1_path))
        img_t2 = nib.load(str(t2_path))
        mask_data = img_mask.get_fdata().astype(np.int16)
        t1_data = img_t1.get_fdata().astype(np.float32)
        t2_data = img_t2.get_fdata().astype(np.float32)
        if mask_data.shape != t1_data.shape:
            return f"Shape Mismatch: {cid} Mask{mask_data.shape} vs MRI{t1_data.shape}"
        unique_habs = np.unique(mask_data)
        unique_habs = unique_habs[unique_habs > 0]
        save_dict = {}
        save_dict['spacing'] = np.array(img_mask.header.get_zooms()[:3], dtype=np.float32)
        save_dict['shape'] = np.array(mask_data.shape, dtype=np.int16)
        for h_id in unique_habs:
            indices = np.where(mask_data == h_id)
            coords = np.stack(indices, axis=-1).astype(np.int16)
            vals_t1 = t1_data[indices].reshape(-1, 1)
            vals_t2 = t2_data[indices].reshape(-1, 1)
            combined = np.hstack([coords, vals_t1, vals_t2])
            save_dict[f'Habitat_{h_id}'] = combined
        out_path = OUT_DIR / f'{cid}.npz'
        np.savez_compressed(out_path, **save_dict)
        return None
    except Exception as e:
        return f"Error {mask_file_path.name}: {str(e)}"



if __name__ == '__main__':
    if not MASK_INPUT_DIR.exists():
        print(f"PATH: {MASK_INPUT_DIR}")
        sys.exit(1)
    print(f"Start Pixel Extraction Pipeline")
    print(f"Source Masks : {MASK_INPUT_DIR}")
    print(f"Origin MRI   : {ORIGIN_ROOT}")
    print(f"Output NPZ   : {OUT_DIR}")
    mask_files = list(MASK_INPUT_DIR.glob(f"*{HABITAT_SUFFIX}"))
    if not mask_files:
        print(f" Not find: {HABITAT_SUFFIX}")
        mask_files = list(MASK_INPUT_DIR.glob("*.nii.gz"))
        print(f"   Do all .nii.gz file ({len(mask_files)} )")
    print(f"   >>> Total cases: {len(mask_files)}")
    errors = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(process_single_case, mask_files), total=len(mask_files)))
    for res in results:
        if res: errors.append(res)
    if errors:
        print(f"\n  finish,  {len(errors)} errors:")
        for e in errors[:5]: print(f"  - {e}")
        if len(errors) > 5: print("  ... ")
    else:
        print("\n all success")