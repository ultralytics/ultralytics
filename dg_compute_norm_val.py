
import numpy as np
from PIL import Image
import os

def compute_normalization_parameters_rgb_only(dataset_folder, max_num_images=30000, batch_size=100):
    """
    Compute the mean and standard deviation for normalization of a large dataset,
    processing only RGB images.

    Args:
    dataset_folder (str): Path to the dataset folder containing images.
    batch_size (int): Number of images to process in each batch.

    Returns:
    tuple: mean and standard deviation of the dataset.
    """

    sum_array, sum_squared_array, num_images = None, None, 0

    image_files = [f for f in os.listdir(dataset_folder) if os.path.isfile(os.path.join(dataset_folder, f))]

    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_pixels = []
        if i > max_num_images:
            break
        for image_name in batch_files:
            image_path = os.path.join(dataset_folder, image_name)
            
            try:
                with Image.open(image_path) as img:
                    # Skip images that are not RGB
                    if img.mode != 'RGB':
                        print(f"Skipping non-RGB image: {image_name}")
                        continue

                    img_array = np.array(img).astype(np.float32)
                    img_array /= 255.
                    batch_pixels.append(img_array.reshape(-1, img_array.shape[-1]))
            except Exception as e:
                print(f"Error processing {image_name}: {e}")

        if len(batch_pixels) == 0:
            continue

        batch_pixels = np.concatenate(batch_pixels, axis=0)
        
        if sum_array is None:
            sum_array = np.sum(batch_pixels, axis=0)
            sum_squared_array = np.sum(batch_pixels ** 2, axis=0)
        else:
            sum_array += np.sum(batch_pixels, axis=0)
            sum_squared_array += np.sum(batch_pixels ** 2, axis=0)

        num_images += batch_pixels.shape[0]

    mean = sum_array / num_images
    std = np.sqrt(sum_squared_array / num_images - mean ** 2)

    return mean, std


# Example usage
dataset_folder = '/data/ml-data/UTKFace_datasets/Images'
mean, std = compute_normalization_parameters_rgb_only(dataset_folder)
print("Mean:", mean)
print("Standard Deviation:", std)

# Mean: [0.31194286 0.31191927 0.31099383]
# Standard Deviation: [0.45492784 0.41749189 0.39646529]
