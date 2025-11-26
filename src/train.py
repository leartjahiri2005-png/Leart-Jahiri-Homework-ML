import os
import time
import torch
import torch.nn as nn
from src.data_loader import get_dataloaders
from src.sod_model import SODModel, iou_score, compute_prf1

def soft_iou(pred, target, eps=1e-6):
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean()

def train(
    num_epochs=25,
    batch_size=8,
    lr=5e-4,
    patience=7,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs("models", exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

    model = SODModel().to(device)
    bce_loss = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        model.train()
        train_loss_sum = 0.0
        train_iou_sum = 0.0
        n_train_batches = 0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            preds = model(images)

            bce = bce_loss(preds, masks)
            s_iou = soft_iou(preds, masks)
            loss = bce + 0.5 * (1.0 - s_iou)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_iou_sum += iou_score(preds.detach(), masks.detach()).item()
            n_train_batches += 1

        train_loss = train_loss_sum / max(1, n_train_batches)
        train_iou = train_iou_sum / max(1, n_train_batches)

        model.eval()
        val_loss_sum = 0.0
        val_iou_sum = 0.0
        val_prec_sum = 0.0
        val_rec_sum = 0.0
        val_f1_sum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                preds = model(images)

                bce = bce_loss(preds, masks)
                s_iou = soft_iou(preds, masks)
                loss = 0.7 * bce + 0.3 * (1.0 - s_iou) 


                val_loss_sum += loss.item()
                val_iou_sum += iou_score(preds, masks).item()

                p, r, f1 = compute_prf1(preds, masks)
                val_prec_sum += p.item()
                val_rec_sum += r.item()
                val_f1_sum += f1.item()

                n_val_batches += 1

        val_loss = val_loss_sum / max(1, n_val_batches)
        val_iou = val_iou_sum / max(1, n_val_batches)
        val_prec = val_prec_sum / max(1, n_val_batches)
        val_rec = val_rec_sum / max(1, n_val_batches)
        val_f1 = val_f1_sum / max(1, n_val_batches)

        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch}/{num_epochs}  (time: {epoch_time:.1f}s)")
        print(f"  Train: loss={train_loss:.4f}, IoU={train_iou:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, IoU={val_iou:.4f},"
            f"Prec={val_prec:.4f}, Rec={val_rec:.4f}, F1={val_f1:.4f}"
        )
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join("models", "last_checkpoint.pth"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join("models", "best_model.pth"))
            print("Saved new best model.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    print("\nTraining finished.")
if __name__ == "__main__":
    train(num_epochs=25, batch_size=8, lr=5e-4, patience=7)
