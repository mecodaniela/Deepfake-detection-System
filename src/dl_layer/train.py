"""
train.py — Loop trainimi për EfficientNet-B0 (dl_layer).
Ekzekutim: python src/dl_layer/train.py

Mbështet RESUME: nëse gjendet checkpoint i fundit, vazhdon nga epoka
ku ka mbetur, me gjendjen e plotë të optimizer/scheduler.
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

# Import nga modulet e tjera të projektit
sys.path.append(str(Path(__file__).resolve().parents[2]))  # rrënja e projektit
from src.dl_layer.model import build_model, count_trainable_params
from src.data_pipeline.dataset_loader import get_dataloader



NUM_EPOCHS = 5
NUM_UNFROZEN_BLOCKS = 1     # ishte 3
LEARNING_RATE = 5e-5        # ishte 1e-4
WEIGHT_DECAY = 3e-4         # ishte 1e-4
EARLY_STOP_PATIENCE = 2

MODELS_DIR = Path("models")
BEST_MODEL_PATH = MODELS_DIR / "dl_layer_best.pt"
CHECKPOINT_PATH = MODELS_DIR / "dl_layer_last_checkpoint.pt"

DEVICE = torch.device("cpu")  # eksplicit, siç e konfirmuam

# NJË EPOKË TRAINIMI

def train_one_epoch(model, loader, optimizer, criterion, epoch_num):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 50 == 0:
            elapsed = time.time() - start
            print(f"  [Epoka {epoch_num}] batch {batch_idx}/{len(loader)} "
                  f"— loss: {loss.item():.4f} — {elapsed:.0f}s")

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ============================================================
# VLERËSIM (mbi val, pa gradient)
# ============================================================

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ============================================================
# NDIHMËSE — CHECKPOINT (PËR RESUME)
# ============================================================

def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, epochs_no_improve):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "epochs_no_improve": epochs_no_improve,
        "num_unfrozen_blocks": NUM_UNFROZEN_BLOCKS,
    }, CHECKPOINT_PATH)


def load_checkpoint(model, optimizer, scheduler):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["best_val_loss"]
    epochs_no_improve = checkpoint["epochs_no_improve"]

    print(f"Checkpoint u gjet — rifillim nga epoka {start_epoch} "
          f"(best_val_loss deri tani: {best_val_loss:.4f})")

    return start_epoch, best_val_loss, epochs_no_improve


# ============================================================
# MAIN
# ============================================================

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Duke ngarkuar datasets...")
    train_loader = get_dataloader("train", shuffle=True)
    val_loader = get_dataloader("val", shuffle=False)

    print("Duke ndërtuar modelin...")
    model = build_model(num_unfrozen_blocks=NUM_UNFROZEN_BLOCKS).to(DEVICE)
    trainable, total = count_trainable_params(model)
    print(f"Parametra trainueshëm: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.1f}%)")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    # Resume automatik nëse ekziston checkpoint i mëparshëm
    if CHECKPOINT_PATH.exists():
        start_epoch, best_val_loss, epochs_no_improve = load_checkpoint(
            model, optimizer, scheduler
        )
    else:
        start_epoch = 1
        best_val_loss = float("inf")
        epochs_no_improve = 0

    if start_epoch > NUM_EPOCHS:
        print("Trainimi ishte tashmë kompletuar sipas NUM_EPOCHS. Asgjë për të bërë.")
        return

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n=== Epoka {epoch}/{NUM_EPOCHS} ===")
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        epoch_time = time.time() - epoch_start
        print(f"Epoka {epoch} përfundoi në {epoch_time:.0f}s")
        print(f"  Train — loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")
        print(f"  Val   — loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Learning rate aktual: {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "num_unfrozen_blocks": NUM_UNFROZEN_BLOCKS,
            }, BEST_MODEL_PATH)
            print(f"  ✓ Model i ri më i mirë u ruajt (val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  Pa përmirësim te val_loss ({epochs_no_improve}/{EARLY_STOP_PATIENCE})")

        # Ruaj checkpoint të plotë PAS ÇDO epoke (jo vetëm kur përmirësohet) — për resume
        save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, epochs_no_improve)

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping te epoka {epoch} — val_loss s'u përmirësua "
                  f"prej {EARLY_STOP_PATIENCE} epokash radhazi.")
            break

    print(f"\nTrainimi përfundoi. Modeli më i mirë: val_loss = {best_val_loss:.4f}")
    print(f"Ruajtur te: {BEST_MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()