import os
import torch
import matplotlib.pyplot as plt
from src.data_loader import get_dataloaders
from src.sod_model import SODModel, iou_score, compute_prf1

def evaluate_on_test(model, test_loader, device):
    model.eval()

    iou_sum = 0.0
    prec_sum = 0.0
    rec_sum = 0.0
    f1_sum = 0.0
    mae_sum = 0.0
    n = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)

            iou = iou_score(preds, masks)
            p, r, f1 = compute_prf1(preds, masks)
            mae = torch.abs(preds - masks).mean()

            iou_sum += iou.item()
            prec_sum += p.item()
            rec_sum += r.item()
            f1_sum += f1.item()
            mae_sum += mae.item()
            n += 1

    print("\n--- Test set metrics ---")
    print(f"IoU:       {iou_sum / n:.4f}")
    print(f"Precision: {prec_sum / n:.4f}")
    print(f"Recall:    {rec_sum / n:.4f}")
    print(f"F1-score:  {f1_sum / n:.4f}")
    print(f"MAE:       {mae_sum / n:.4f}")

def save_example_predictions(
    model,
    test_loader,
    device,
    output_dir="results",
    num_batches=2,
):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    batch_idx = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            preds_bin = (preds > 0.4).float()

            for i in range(images.size(0)):
                img = images[i].cpu().permute(1, 2, 0).numpy()
                gt_mask = masks[i].cpu().squeeze(0).numpy()
                pr_mask = preds_bin[i].cpu().squeeze(0).numpy()

                overlay = img.copy()
                overlay[..., 2] = overlay[..., 2] + pr_mask * 0.8
                overlay = overlay.clip(0, 1)

                fig, axes = plt.subplots(1, 4, figsize=(10, 3))
                axes[0].imshow(img)
                axes[0].set_title("Input")
                axes[0].axis("off")

                axes[1].imshow(gt_mask, cmap="gray")
                axes[1].set_title("GT mask")
                axes[1].axis("off")

                axes[2].imshow(pr_mask, cmap="gray")
                axes[2].set_title("Pred mask")
                axes[2].axis("off")

                axes[3].imshow(overlay)
                axes[3].set_title("Overlay")
                axes[3].axis("off")

                fig.tight_layout()
                out_path = os.path.join(
                    output_dir, f"example_b{batch_idx}_i{i}.png"
                )
                plt.savefig(out_path, dpi=150)
                plt.close(fig)

            batch_idx += 1
            if batch_idx >= num_batches:
                break

    print(f"Saved example predictions to: {output_dir}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    _, _, test_loader = get_dataloaders(batch_size=8)

    model = SODModel().to(device)
    model.load_state_dict(
        torch.load("models/best_model.pth", map_location=device)
    )
    evaluate_on_test(model, test_loader, device)
    save_example_predictions(model, test_loader, device)