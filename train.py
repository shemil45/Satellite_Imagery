import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from your other files
from model import UNET
from dataloader import BuildingDataset

# --- Hyperparameters & Configuration ---
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 # Adjust based on your GPU memory
NUM_EPOCHS = 5 # Start with a small number to test
NUM_WORKERS = 0 # Set to 0 for Windows compatibility
PIN_MEMORY = True
LOAD_MODEL = False # Set to True to load a pre-trained model

# --- Dataset Paths ---
# Make sure these point to your preprocessed patch folders
TRAIN_IMG_DIR = "data_processed/train/"
TRAIN_MASK_DIR = "data_processed/train_mask/"
VAL_IMG_DIR = "data_processed/val/"
VAL_MASK_DIR = "data_processed/val_mask/"

def train_fn(loader, model, optimizer, loss_fn):
    """
    Runs one epoch of training.
    """
    loop = tqdm(loader, desc="Training")

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # Forward pass
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        # Backward pass (backpropagation)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())

def check_accuracy(loader, model, device="cuda"):
    """
    Checks accuracy on the validation set.
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval() # Set model to evaluation mode

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    accuracy = num_correct / num_pixels * 100
    avg_dice_score = dice_score / len(loader)
    print(f"Validation Accuracy: {accuracy:.2f}%")
    print(f"Dice Score: {avg_dice_score:.4f}")
    
    model.train() # Set model back to training mode
    return avg_dice_score

def main():
    """Main function to run the training and validation."""
    print(f"Using device: {DEVICE}")
    
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    
    # Using Binary Cross-Entropy with Logits for better numerical stability
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Adam is a popular and effective optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Setup DataLoaders ---
    train_dataset = BuildingDataset(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )

    val_dataset = BuildingDataset(image_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )

    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_fn(train_loader, model, optimizer, loss_fn)

        # Check accuracy on validation set
        check_accuracy(val_loader, model, device=DEVICE)

    # --- Save Model ---
    print("\nTraining complete. Saving model...")
    torch.save(model.state_dict(), "unet_buildings.pth")
    print("Model saved to unet_buildings.pth")


if __name__ == "__main__":
    main()
