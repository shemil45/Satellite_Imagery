import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class BuildingDataset(Dataset):
    """
    Custom Dataset class for the Massachusetts Buildings dataset.
    This class will load image and mask patches from the disk.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (string): Directory with all the input image patches.
            mask_dir (string): Directory with all the corresponding mask patches.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Get a sorted list of image filenames
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Fetches a single data point (image and mask) from the dataset.
        
        Args:
            idx (int): Index of the data point to retrieve.
        
        Returns:
            tuple: (image, mask) where both are PyTorch tensors.
        """
        # Construct the file paths
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name) # Mask has the same filename

        # Open image and mask using Pillow
        # Images are RGB, masks are grayscale
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Convert images to numpy arrays
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # Normalize pixel values to be between 0 and 1
        image_np = image_np / 255.0
        mask_np = mask_np / 255.0
        
        # Add a channel dimension to the mask (H, W) -> (1, H, W)
        # This is needed for the loss function later
        mask_np = np.expand_dims(mask_np, axis=0)

        # PyTorch expects images in the format [Channels, Height, Width]
        # We need to transpose the image array from (H, W, C) to (C, H, W)
        image_np = image_np.transpose((2, 0, 1))
        
        # Convert numpy arrays to PyTorch tensors
        image_tensor = torch.from_numpy(image_np).float()
        mask_tensor = torch.from_numpy(mask_np).float()
        
        # Note: We are not using the 'transform' argument in this basic version,
        # but it's good practice to include it for future data augmentation.

        return image_tensor, mask_tensor

# --- Example of how to use the Dataset and DataLoader ---
if __name__ == '__main__':
    # Define the paths to your processed (patched) data
    # IMPORTANT: Update these paths if your folders are named differently
    TRAIN_IMG_DIR = "data_processed/train/"
    TRAIN_MASK_DIR = "data_processed/train_mask/"

    print("Setting up the Dataset...")
    # 1. Create an instance of our custom Dataset
    train_dataset = BuildingDataset(
        image_dir=TRAIN_IMG_DIR,
        mask_dir=TRAIN_MASK_DIR
    )

    print(f"Dataset created with {len(train_dataset)} samples.")
    
    print("\nSetting up the DataLoader...")
    # 2. Create an instance of the DataLoader
    # This will handle batching, shuffling, and loading data in parallel.
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=4,  # How many samples per batch
        shuffle=True, # Shuffle data at the beginning of each epoch
        num_workers=0 # Use 2 background processes to load data
    )
    
    print("DataLoader is ready. Fetching one batch to test...")

    # 3. Iterate over the DataLoader to get a batch
    try:
        images, masks = next(iter(train_loader))

        # Print the shape of the tensors to verify everything is correct
        print("\n--- Verification ---")
        print(f"Shape of one batch of images: {images.shape}")
        print(f"Shape of one batch of masks: {masks.shape}")
        print("--------------------")
        print("\nExplanation of shapes:")
        print(f"[Batch Size, Channels, Height, Width]")
        print("Image channels is 3 (R, G, B).")
        print("Mask channels is 1 (Grayscale).")

    except StopIteration:
        print("DataLoader is empty. Make sure your data directories are correct and not empty.")
    except Exception as e:
        print(f"An error occurred: {e}")
