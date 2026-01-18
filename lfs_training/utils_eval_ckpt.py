import os
import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

from torchmetrics.classification import Accuracy, Precision, Recall, AUROC, ROC, AveragePrecision


def build_cls_metrics(device):
    accuracy_func = Accuracy(task="binary").to(device)
    precision_func = Precision(task="binary").to(device)
    recall_func = Recall(task="binary").to(device)
    auc_func = AUROC(task="binary").to(device)
    roc_curve_func = ROC(task="binary").to(device)
    average_precision_func = AveragePrecision(task="binary").to(device)
    return accuracy_func, precision_func, recall_func, auc_func, roc_curve_func, average_precision_func


def tr_eval(trained_model, loader, epoch, phase="Validation", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.eval()
    preds, labels, prob_list = [], [], []
    with torch.no_grad():
        for batch in loader:
            clouds = batch["clouds"].to(device)
            dist_mat = batch["dist_mat"].to(device)
            bag_mask = batch["bag_mask"].to(device)
            label = batch["label"].to(device).float()

            logits, _, _, _, _, _ = trained_model(clouds, dist_mat, mask=bag_mask)
            probs_positive = torch.softmax(logits, dim=1)[:, 1]
            pred = (probs_positive >= 0.5).long()

            prob_list.append(probs_positive.detach().cpu())
            preds.append(pred.detach().cpu())
            labels.append(label.detach().cpu())
    preds = torch.cat(preds, dim=0)
    probs_positive = torch.cat(prob_list, dim=0)
    labels = torch.cat(labels, dim=0)
    metric_device = preds.device
    accuracy_func, precision_func, recall_func, auc_func, roc_curve_func, auprc_func = build_cls_metrics(metric_device)
    acc = accuracy_func(preds, labels.long())
    prec = precision_func(preds, labels.long())
    rec = recall_func(preds, labels.long())
    auc_score = auc_func(probs_positive, labels.long())
    auprc_score = auprc_func(probs_positive, labels.long())

    print(f"\n{phase} Results at Epoch {epoch}:")
    print(f"ACC: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | AUROC: {auc_score:.4f}")
    print(f"AUPRC: {auprc_score:.4f}")

    return float(auc_score.item()), float(acc.item())


def _pct(x: float) -> str:
    return f"{x * 100:.2f}"


def _sanitize(tag: str) -> str:
    return tag.replace("+", "_").replace("/", "_").replace(" ", "_")


def save_best_ckpt(
    model,
    optimizer,
    epoch: int,
    base_name: str,
    total_auc: float,
    total_acc: float,
    group_metrics: Dict[str, Tuple[float, float]],
    best_acc_so_far: float,
    best_auc_so_far: float,
    ckpt_dir: str = "checkpoints",
):
    improved = False
    updated_acc = best_acc_so_far
    updated_auc = best_auc_so_far
    if (total_acc > best_acc_so_far) or (total_auc > best_auc_so_far):
        improved = True
        os.makedirs(ckpt_dir, exist_ok=True)
        name_parts = [
            f"Epoch_{epoch}",
            f"TotalAUC_{_pct(total_auc)}",
            f"TotalACC_{_pct(total_acc)}",
        ]
        for gname, (auc_g, acc_g) in group_metrics.items():
            name_parts.append(f"{_sanitize(gname)}_AUC_{_pct(auc_g)}_ACC_{_pct(acc_g)}")
        ckpt_path = Path(ckpt_dir) / ("_".join(name_parts) + ".pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_acc": max(best_acc_so_far, total_acc),
                "best_auc": max(best_auc_so_far, total_auc),
                "metrics": {
                    "total_auc": total_auc,
                    "total_acc": total_acc,
                    "group_metrics": group_metrics,
                },
            },
            ckpt_path,
        )
        print(f"Saved best model: {ckpt_path}")
        updated_acc = max(best_acc_so_far, total_acc)
        updated_auc = max(best_auc_so_far, total_auc)
    return updated_acc, updated_auc
