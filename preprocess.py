import os
from PIL import Image
import numpy as np

# --- Configuration ---
# Set the desired size for the patches
PATCH_SIZE = 256
# Set the stride (how many pixels to move for the next patch). Can be same as PATCH_SIZE for no overlap.
STRIDE = 256
# Minimum percentage of building pixels required to save a patch (e.g., 0.01 = 1%)
MIN_BUILDING_PERCENTAGE = 0.01

# --- Paths ---
# Adjust these paths to match your folder structure
# Assumes you have folders like: ./data/train/ and ./data/train_mask/
INPUT_IMAGE_DIR = './data/train/'
INPUT_MASK_DIR = './data/train_mask/'

# Output directories will be created if they don't exist
OUTPUT_IMAGE_DIR = './data_processed/train/'
OUTPUT_MASK_DIR = './data_processed/train_mask/'

def create_patches():
    """
    Reads large images and masks and creates smaller patches from them.
    """
    print("Starting patch creation process...")

    # Create output directories if they don't exist
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

    # Get a list of the image filenames
    image_files = [f for f in os.listdir(INPUT_IMAGE_DIR) if f.endswith('.tiff')]
    
    total_patches_saved = 0

    for filename in image_files:
        try:
            # Construct full paths for the image and its corresponding mask
            image_path = os.path.join(INPUT_IMAGE_DIR, filename)
            # The dataset uses .tif for masks, so we replace the extension
            mask_filename = filename.replace('.tiff', '.tif')
            mask_path = os.path.join(INPUT_MASK_DIR, mask_filename)

            # Open the image and mask
            image = Image.open(image_path)
            mask = Image.open(mask_path)
            
            print(f"Processing {filename}...")

            # Get image dimensions
            width, height = image.size

            # Iterate over the image with a sliding window
            for y in range(0, height - PATCH_SIZE + 1, STRIDE):
                for x in range(0, width - PATCH_SIZE + 1, STRIDE):
                    # Define the crop box
                    box = (x, y, x + PATCH_SIZE, y + PATCH_SIZE)

                    # Crop the mask first to check if it's worth saving
                    mask_patch = mask.crop(box)
                    mask_patch_np = np.array(mask_patch)

                    # Calculate the percentage of building pixels (white pixels)
                    # White pixels have a value of 255
                    building_pixels = np.sum(mask_patch_np == 255)
                    total_pixels = PATCH_SIZE * PATCH_SIZE
                    building_percentage = building_pixels / total_pixels
                    
                    # Only save patches that meet the minimum threshold
                    if building_percentage >= MIN_BUILDING_PERCENTAGE:
                        # Crop the corresponding image patch
                        image_patch = image.crop(box)

                        # Create a unique filename for the patch
                        patch_filename = f"{os.path.splitext(filename)[0]}_{y}_{x}.tif"
                        
                        # Save the patches
                        image_patch.save(os.path.join(OUTPUT_IMAGE_DIR, patch_filename))
                        mask_patch.save(os.path.join(OUTPUT_MASK_DIR, patch_filename))
                        
                        total_patches_saved += 1

        except FileNotFoundError:
            print(f"Warning: Mask for {filename} not found. Skipping.")
        except Exception as e:
            print(f"An error occurred with {filename}: {e}")

    print(f"\nPatch creation complete!")
    print(f"Total patches saved: {total_patches_saved}")


if __name__ == '__main__':
    create_patches()