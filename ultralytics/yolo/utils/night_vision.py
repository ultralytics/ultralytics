import cv2
import numpy as np
import torch


def night_vision_core(img, image_gamma='auto', min_gamma=0.6, max_gamma=1.0, min_normalized_intensity=0.25):
    """
    Night vision mode core function.

    Args:
        img (numpy.ndarray): The image to apply night vision mode to.

    Returns:
        numpy.ndarray: The image with night vision mode applied.
    """

    # Night vision mode

    # Split the image into separate color channels
    # b, g, r = cv2.split(img)

    # --------------------------------------------------------

    # -- Histogram Equalization --

    # Apply histogram equalization to each color channel
    # eq_b = cv2.equalizeHist(b)
    # eq_g = cv2.equalizeHist(g)
    # eq_r = cv2.equalizeHist(r)

    # eq_img = cv2.merge((eq_b, eq_g, eq_r))     # Merge the color channels back into a single image

    # --------------------------------------------------------

    # -- Contrast Limited Adaptive Histogram Equalization --

    # Apply CLAHE to each color channel
    # clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
    # cl_b = clahe.apply(b)
    # cl_g = clahe.apply(g)
    # cl_r = clahe.apply(r)

    # cl_img = cv2.merge((cl_b, cl_g, cl_r))    # Merge the color channels back into a single image

    # --------------------------------------------------------

    # -- Contrast Stretching --

    # Compute the minimum and maximum intensity values
    # min_intensity = np.min(img)
    # max_intensity = np.max(img)

    # Stretch the intensity values
    # a = 255 / (max_intensity - min_intensity)
    # b = -a * min_intensity

    # img_stretched = np.clip((a * img + b), 0, 255).astype(np.uint8)

    # --------------------------------------------------------

    # -- Gamma Correction --

    gamma = image_gamma

    if image_gamma == 'auto':
        intensity = measure_intensity(img)  # intensity of the image, value range (0-255)
        normaized_intensity = intensity / 255.0

        if normaized_intensity < min_normalized_intensity:  # if the image is dark
            scaled_intensity = ((normaized_intensity - 0) / (1 - 0)) * (max_gamma - min_gamma) + min_gamma
        else:   # if the image is bright
            scaled_intensity = 1.0  # no gamma correction
            
        gamma = scaled_intensity
    elif type(gamma) == float or type(gamma) == int:
        gamma = image_gamma
    else:
        raise ValueError("image_gamma must be either 'auto' or a float or int")

    # Apply gamma correction
    img_gamma_corrected = np.power(img / 255.0, gamma) * 255.0
    img_gamma_corrected = img_gamma_corrected.astype(np.uint8)

    # --------------------------------------------------------

    # b, g, r = cv2.split(img_gamma_corrected)

    # # Apply unsharp masking
    # b_unsharp = cv2.addWeighted(b, 1.5, b, -0.5, 0)

    # g_unsharp = cv2.addWeighted(g, 1.5, g, -0.5, 0)

    # r_unsharp = cv2.addWeighted(r, 1.5, r, -0.5, 0)

    # sharp_img = cv2.merge((b_unsharp, g_unsharp, r_unsharp))

    # Display the resulting images
    # cv2.imshow('Original Image', img)
    # cv2.imshow('Equalized Image', eq_img)
    # cv2.imshow('CLAHE Image', cl_img)

    # cv2.waitKey(0)     # Wait for a key press
    # cv2.destroyAllWindows()    # Close all windows

    return img_gamma_corrected


def convert_to_cv2(img):
    """
    Converts an image from torch to cv2 format.

    Args:
        img (torch.Tensor): The image in torch format. (shape: (3, H, W))

    Returns:
        numpy.ndarray: The image in cv2 format. (shape: (H, W, 3))
    """
    img = img.permute(1, 2, 0).numpy()  # convert to numpy
    img = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)  # normalize the image to 0-255 UINT8
    return img


def convert_to_torch(img):
    """
    Converts an image from torch to cv2 format.

    Args:
        img (numpy.ndarray): The image in cv2 format. (shape: (H, W, 3))

    Returns:
        torch.Tensor: The image in torch format. (shape: (3, H, W))
    """
    # normalize the image to 0-1 FLOAT32
    img = img / 255.0
    img = img.astype(np.float32)

    img = torch.from_numpy(img).permute(2, 0, 1)  # convert back to tensor
    return img


def apply_night_vision(img, image_gamma='auto', min_gamma=0.6, max_gamma=1.0, min_normalized_intensity=0.25):
    """
    Applies night vision mode to an image.
    (applies night_vision_core() function to the image)

    Args:
        img (torch.Tensor): The image to apply night vision mode to. (shape: (3, H, W))

    Returns:
        torch.Tensor: The image with night vision mode applied. (shape: (3, H, W))
    """
    img = convert_to_cv2(img)  # convert to cv2
    img = night_vision_core(img, image_gamma, min_gamma, max_gamma, min_normalized_intensity)  # apply night vision
    img = convert_to_torch(img)  # convert back to torch
    return img


def apply_night_vision_on_batch(batch, image_gamma='auto', min_gamma=0.6, max_gamma=1.0, min_normalized_intensity=0.25):
    """
    Applies night vision mode to a batch of images.
    (applies night_vision_core() function to each image in the batch)

    Args:
        batch (torch.Tensor): The batch of images to apply night vision mode to. (shape: (N, 3, H, W))

    Returns:
        torch.Tensor: The batch of images with night vision mode applied. (shape: (N, 3, H, W))
    """
    # batch = batch.cpu()
    for i in range(batch.shape[0]):
        img = batch[i]
        img = apply_night_vision(img, image_gamma, min_gamma, max_gamma, min_normalized_intensity) * 255.0
        batch[i] = img
    return batch


def measure_intensity(img):
    """
    Measures the intensity of an image.

    Args:
        img (numpy.ndarray): The image to measure the intensity of. (shape: (H, W, 3))

    Returns:
        float: The intensity of the image.
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    intensity = np.mean(gray_img)
    return intensity


# Load the image
# img = cv2.imread(r'D:\ME\Downloads\New folder\Downloads\dark_stop_6.png')

# print("img data type: ", img.dtype)
# print("img: ", img)
# print("img type: ", type(img), "img shape: ", img.shape)

# display the image
# cv2.imshow('Original Image', img)

# Apply night vision mode
# img = night_vision_core(img)

# Display the resulting image
# cv2.imshow('Enhanced Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
