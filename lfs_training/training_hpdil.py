import os
import torch
from itertools import chain
from tqdm import tqdm
from torch.optim import Adam
from lfs_dataset.lfs_3d_habitat_dataset import get_data_loaders
from lfs_model.hp_dil import HabitatGraph, Discriminator, MI_Est
from utils_eval_ckpt import tr_eval, save_best_ckpt

TASK_LIST = [
    ("TASK01", "task_1_01vs234"),
    ("TASK02", "task_2_012vs34"),
    ("TASK03", "task_3_0123vs4"),
]
START_VAL_EPOCH = 5
FOLDS = ["Tr_Fold_1", "Tr_Fold_2", "Tr_Fold_3", "Tr_Fold_4", "Tr_Fold_5"]
RANDOM_SEED = 42
REPORT_TRAIN_GROUP = True


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    for task_tag, task_col_name in TASK_LIST:
        NUM_HABITAT = 23
        EXP_NUM = NUM_HABITAT
        NUM_POINT = 4096
        BASE_NAME = f"HP_DIL_{task_tag}"
        TASK = task_col_name
        NOTE = "main_exam"
        BATCH_SIZE = 6
        EPOCHS = 120
        LR = 3e-5
        INNER_LOOP = 50
        POS_WEIGHT = 1.0
        NEG_WEIGHT = 10.0
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for VAL_FOLD in FOLDS:
            TRAIN_FOLDS = [f for f in FOLDS if f != VAL_FOLD]
            MODEL_NAME = f"{BASE_NAME}_{NUM_HABITAT}_habitat_{NUM_POINT}_{NOTE}_{VAL_FOLD}"
            CKPT_DIR = f"checkpoints/{MODEL_NAME}"
            print("\n" + "=" * 70)
            print(f"Task: {task_tag} | Split: val={VAL_FOLD}")
            print(f"Task Col: {TASK}")
            print(f"Save Dir: {CKPT_DIR}")
            print("=" * 70 + "\n")
            config = {
                "root_dir": r"C:\Users\wunan\Desktop\HP_DIL_OPENSOURSE",
                "excel_path": r"C:\Users\wunan\Desktop\HP_DIL_OPENSOURSE\all_patient_list_training_test.xlsx",
                "sheet_name": "all_patient",
                "data_path": rf"C:\Users\wunan\Desktop\HP_DIL_OPENSOURSE\Habitat_Louvain_K30_Res1.1_EXP{NUM_HABITAT}\PointCloud_Data_K{NUM_HABITAT}_N{NUM_POINT}",
                # "sv_volume": 1000,
                # "compact": 0.075,
                "exp_number": EXP_NUM,
                # "knn_k": 30,
                # "resolution": 1.1,
                "num_point": NUM_POINT,
                "num_habitat": NUM_HABITAT,
                "batch_size": BATCH_SIZE,
                "num_workers": 4,
                "task": TASK,
                "pos_weight": POS_WEIGHT,
                "neg_weight": NEG_WEIGHT,
            }
            if REPORT_TRAIN_GROUP:
                config["val_groups"] = [
                    TRAIN_FOLDS,
                    [VAL_FOLD],
                ]
            else:
                config["val_groups"] = [
                    [VAL_FOLD],
                ]
            config["split_cfg"] = {
                "train": TRAIN_FOLDS,
                "val": list(chain.from_iterable(config["val_groups"])),
                "test_ext": [],
            }
            train_loader, val_loaders = get_data_loaders(config)
            print("=" * 40)
            print(f"Train Samples: {len(train_loader.sampler)}")
            for name, loader in val_loaders:
                print(f"Val Group [{name}]: {len(loader.dataset)} samples")
            print("=" * 40 + "\n")
            model = HabitatGraph(
                feat_dim=120,
                pe_dim=8,
                num_habitat=NUM_HABITAT,
                in_channel=2,
                num_classes=2
            ).to(DEVICE)
            discriminator = Discriminator(hidden_size=128).to(DEVICE)
            optimizer = Adam(model.parameters(), lr=LR)
            optimizer_mi = Adam(discriminator.parameters(), lr=5e-4)
            criterion = torch.nn.CrossEntropyLoss(
                weight=torch.tensor([NEG_WEIGHT, POS_WEIGHT], device=DEVICE)
            )
            best_acc, best_auc = 0.0, 0.0
            for epoch in range(1, EPOCHS + 1):
                model.train()
                total_loss = 0.0

                pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
                for batch in pbar:
                    clouds = batch["clouds"].to(DEVICE)
                    dist_mat = batch["dist_mat"].to(DEVICE)
                    bag_mask = batch["bag_mask"].to(DEVICE)
                    label = batch["label"].to(DEVICE).float()
                    logits, pos_embed, all_embed, loss_conn, loss_ent, node_choice = model(
                        clouds, dist_mat, mask=bag_mask
                    )
                    for _ in range(INNER_LOOP):
                        optimizer_mi.zero_grad()
                        local_loss = -MI_Est(discriminator, all_embed.detach(), pos_embed.detach())
                        local_loss.backward()
                        optimizer_mi.step()
                    logits = logits.squeeze(-1)
                    loss_cls = criterion(logits, label.long())
                    mi_loss = MI_Est(discriminator, all_embed, pos_embed)
                    loss = loss_cls + 0.1 * mi_loss + 0.01 * (loss_conn + loss_ent)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss = loss.item()
                    total_loss += train_loss
                    pbar.set_postfix(
                        loss=f"{train_loss:.4f}",
                        cls=f"{loss_cls.item():.4f}",
                        conn=f"{loss_conn.item():.4f}",
                        ent=f"{loss_ent.item():.4f}",
                        mi=f"{mi_loss.item():.4f}",
                    )
                avg_loss = total_loss / max(1, len(train_loader))
                print(f"Epoch {epoch} Done. Avg Loss: {avg_loss:.4f}")
                if epoch >= START_VAL_EPOCH:
                    group_metrics = {}
                    for gname, loader in val_loaders:
                        auc_g, acc_g = tr_eval(
                            model, loader, epoch, phase=f"Val-{gname}", device=DEVICE
                        )
                        group_metrics[gname] = (auc_g, acc_g)
                    if len(val_loaders) == 1:
                        avg_val_auc, avg_val_acc = list(group_metrics.values())[0]
                    else:
                        key = VAL_FOLD if VAL_FOLD in group_metrics else list(group_metrics.keys())[-1]
                        avg_val_auc, avg_val_acc = group_metrics[key]
                    print("=" * 60)
                    print(f"Epoch {epoch} Report [{task_tag}] | val={VAL_FOLD}")
                    print(f"AUC: {avg_val_auc:.4f} | ACC: {avg_val_acc:.4f}")
                    for gname, (g_auc, g_acc) in group_metrics.items():
                        print(f"{gname}: AUC {g_auc:.4f} | ACC {g_acc:.4f}")
                    print("=" * 60)

                    best_acc, best_auc = save_best_ckpt(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        base_name=BASE_NAME,
                        total_auc=avg_val_auc,
                        total_acc=avg_val_acc,
                        group_metrics=group_metrics,
                        best_acc_so_far=best_acc,
                        best_auc_so_far=best_auc,
                        ckpt_dir=CKPT_DIR,
                    )
            print(f"\nDone: {BASE_NAME} | val={VAL_FOLD}\n")
