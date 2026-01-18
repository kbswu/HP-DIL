from functools import partial
from itertools import chain
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from typing import List, Dict, Union, Any
from werkzeug.debug.repr import missing


def get_label_scalar(df, pid: str, col_name) -> int:
    lab = df.loc[pid, col_name]
    return int(lab)


def select_patients(
        file_list: List[Path],
        label_df: pd.DataFrame,
        center_names: List[str],
        center_col: str = "Training_id",
) -> List[Path]:
    keep = []
    for f in file_list:
        pid = f.stem
        if pid not in label_df.index:
            continue
        center = label_df.loc[pid, center_col]
        if center in center_names:
            keep.append(f)
    return keep


def make_split(
        excel_path: Path,
        file_path: Path,
        sheet: str,
        split_cfg: Dict[str, List[str]] = None,
) -> tuple[dict[str, Any], DataFrame]:
    df = pd.read_excel(excel_path, sheet_name=sheet)
    df["image_id"] = df["image_id"].astype(str)
    df["npz_path"] = df["image_id"].apply(lambda x: file_path / f"{x}.npz")
    split_dict = {}
    if split_cfg:
        for split_name, hosp_list in split_cfg.items():
            mask = df["Training_id"].isin(hosp_list)
            split_dict[split_name] = df[mask]["npz_path"].tolist()
    df_clean = df.groupby("image_id").first()
    return split_dict, df_clean


class PatientBagDataset(Dataset):
    def __init__(
            self,
            file_list: List[Path],
            label_df: pd.DataFrame,
            label_cols: Union[str, List[str]] = "label",
    ):
        self.file_list = file_list
        self.label_dict = dict(zip(label_df.index, label_df.to_dict('records')))
        self.label_cols = [label_cols] if isinstance(label_cols, str) else label_cols

    def __len__(self):
        return len(self.file_list)

    def _load_npz(self, f: Path):
        with np.load(f, allow_pickle=True) as npz:
            clouds_raw = npz["clouds"]  # (20, N, C)
            mask_h = (npz["missing_habitats"] == 0)  # (20,) bool
            phys_dist_raw = npz["phys_dist_mat"]  # (20, 20)
            voxel_dist_raw = npz["voxel_dist_mat"]  # (20, 20)
            centroids_v_raw = npz["centroids_vox"]  # (20, 3)
            centroids_p_raw = npz["centroids_phys"]  # (20, 3)
            missing_mod = npz["missing_modalities"]  # (num_mod,)
            valid_clouds = []
            valid_indices = []
            for i, (c, m) in enumerate(zip(clouds_raw, mask_h)):
                if m and c is not None:
                    valid_clouds.append(c)
                    valid_indices.append(i)
            if len(valid_clouds) > 0:
                clouds = np.stack(valid_clouds, axis=0)  # (H_valid, N, C)
                phys_dist = phys_dist_raw[valid_indices][:, valid_indices]
                voxel_dist = voxel_dist_raw[valid_indices][:, valid_indices]
                centroids_vox = centroids_v_raw[valid_indices]
                centroids_phys = centroids_p_raw[valid_indices]
            else:
                print(f"[Warning] No valid habitats in {f.name}, using dummy data.")
                clouds = np.zeros((1, 8192, 11), dtype=np.float32)
                phys_dist = np.zeros((1, 1), dtype=np.float32)
                voxel_dist = np.zeros((1, 1), dtype=np.float32)
                centroids_vox = np.zeros((1, 3), dtype=np.float32)
                centroids_phys = np.zeros((1, 3), dtype=np.float32)
            clouds_feat = np.delete(clouds, 3, axis=-1)
            habitat_id = clouds[:, 0, 3].astype(int)  # 获取生境ID
            clouds_feat = clouds_feat[:, :, :5]  # (H_valid, N, 5)
            valid_h = clouds_feat.shape[0]
        return {
            "clouds": clouds_feat,  # float32
            "habitat_id": habitat_id,  # int
            "phys_dist": phys_dist,  # float32
            "voxel_dist": voxel_dist,  # float32
            "centroids_vox": centroids_vox,  # float32
            "centroids_phys": centroids_phys,  # float32
            "valid_h": valid_h,  # int
            "missing_mod": missing_mod  # int/bool
        }

    def __getitem__(self, idx):
        path = self.file_list[idx]
        pid = path.stem
        row_data = self.label_dict.get(pid)
        if row_data is None: raise KeyError(f"{pid} not found")
        data_dict = self._load_npz(path)
        if len(self.label_cols) == 1:
            y_val = row_data[self.label_cols[0]]
        else:
            y_val = [row_data[c] for c in self.label_cols]
        label = torch.tensor(y_val, dtype=torch.long if isinstance(y_val, int) else torch.float)
        p_label = torch.tensor(row_data["label"], dtype=torch.long)
        mask_h = torch.ones(data_dict["valid_h"], dtype=torch.bool)
        return {
            "pid": pid,
            "label": label,
            "ord_patient_label": p_label,
            "clouds": torch.from_numpy(data_dict["clouds"]).float(),
            "habitat_id": torch.from_numpy(data_dict["habitat_id"]).long(),
            "dist_mat": torch.from_numpy(data_dict["phys_dist"]).float(),
            "hab_mask": mask_h,
            "voxel_dist": torch.from_numpy(data_dict["voxel_dist"]).float(),
            "centroids_vox": torch.from_numpy(data_dict["centroids_vox"]).float(),
            "centroids_phys": torch.from_numpy(data_dict["centroids_phys"]).float(),
            "missing_mod": torch.from_numpy(data_dict["missing_mod"])
        }


def patient_collate(batch, num_habitats):
    max_H = num_habitats
    clouds_padded_list = []
    bag_mask_list = []
    phys_dist_padded_list = []
    voxel_dist_padded_list = []
    centroids_vox_padded_list = []
    centroids_phys_padded_list = []
    missing_mod_list = []
    for sample in batch:
        clouds = sample["clouds"]
        H_curr, N, C = clouds.shape
        pad_size = max_H - H_curr
        if pad_size > 0:
            pad_c = torch.zeros(pad_size, N, C, dtype=clouds.dtype)
            c_padded = torch.cat([clouds, pad_c], dim=0)
            mask = torch.cat([torch.ones(H_curr, dtype=torch.bool), torch.zeros(pad_size, dtype=torch.bool)], dim=0)
        else:
            c_padded = clouds[:max_H]
            mask = torch.ones(max_H, dtype=torch.bool)
            H_curr = max_H
        clouds_padded_list.append(c_padded)
        bag_mask_list.append(mask)
        p_dist = sample["dist_mat"]
        p_d_padded = torch.zeros((max_H, max_H), dtype=p_dist.dtype)
        p_d_padded[:H_curr, :H_curr] = p_dist[:H_curr, :H_curr]
        phys_dist_padded_list.append(p_d_padded)
        v_dist = sample["voxel_dist"]
        v_d_padded = torch.zeros((max_H, max_H), dtype=v_dist.dtype)
        v_d_padded[:H_curr, :H_curr] = v_dist[:H_curr, :H_curr]
        voxel_dist_padded_list.append(v_d_padded)
        c_vox = sample["centroids_vox"]
        c_v_padded = torch.zeros((max_H, 3), dtype=c_vox.dtype)
        c_v_padded[:H_curr] = c_vox[:H_curr]
        centroids_vox_padded_list.append(c_v_padded)
        c_phys = sample["centroids_phys"]
        c_p_padded = torch.zeros((max_H, 3), dtype=c_phys.dtype)
        c_p_padded[:H_curr] = c_phys[:H_curr]
        centroids_phys_padded_list.append(c_p_padded)
        missing_mod_list.append(sample["missing_mod"])
    out = {
        "clouds": torch.stack(clouds_padded_list, 0),  # (B, 16, N, 5)
        "bag_mask": torch.stack(bag_mask_list, 0),  # (B, 16)
        "dist_mat": torch.stack(phys_dist_padded_list, 0),  # (B, 16, 16)
        "voxel_dist": torch.stack(voxel_dist_padded_list, 0),  # (B, 16, 16)
        "centroids_vox": torch.stack(centroids_vox_padded_list, 0),  # (B, 16, 3)
        "centroids_phys": torch.stack(centroids_phys_padded_list, 0),  # (B, 16, 3)
        "missing_mod": torch.stack(missing_mod_list, 0),  # (B, Modalities)
        "habitat_id": [s["habitat_id"] for s in batch],  # List
        "ord_patient_label": torch.stack([s["ord_patient_label"] for s in batch], 0),
        "label": torch.stack([s["label"] for s in batch], 0),
        "pid": [s["pid"] for s in batch],
    }
    return out


def get_data_loaders(cfg: dict):
    data_path = Path(cfg["data_path"])
    print(f"[Info] Loading data from: {data_path}")
    split, df = make_split(
        excel_path=Path(cfg["excel_path"]),
        file_path=data_path,
        sheet=cfg["sheet_name"],
        split_cfg=cfg["split_cfg"]
    )
    collate_fn_with_param = partial(patient_collate, num_habitats=cfg["num_habitat"])
    task_col = cfg["task"]

    train_files = split["train"]
    train_labels = [get_label_scalar(df, f.stem, task_col) for f in train_files]
    sample_weights = [cfg["neg_weight"] if l == 0 else cfg["pos_weight"] for l in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    train_ds = PatientBagDataset(train_files, df, label_cols=task_col)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn_with_param,
        sampler=sampler,
        pin_memory=True
    )
    val_loaders = []
    val_groups = cfg["val_groups"]
    for group in val_groups:
        paths_in_group = select_patients(split["val"], df, group)
        if len(paths_in_group) == 0:
            print(f"[Warning] Group {group} is empty!")
            continue
        ds = PatientBagDataset(paths_in_group, df, label_cols=task_col)
        loader = DataLoader(
            ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            collate_fn=collate_fn_with_param,
            pin_memory=True
        )
        group_name = "+".join(group)
        val_loaders.append((group_name, loader))
    return train_loader, val_loaders


if __name__ == "__main__":
    pass


