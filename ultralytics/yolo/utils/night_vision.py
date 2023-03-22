import cv2
import numpy as np
import torch


def night_vision_core(img):
    """
    Night vision mode core function.
    Applies histogram equalization the image.

    Args:
        img (numpy.ndarray): The image to apply night vision mode to.

    Returns:
        numpy.ndarray: The image with night vision mode applied.
    """

    # Night vision mode

    # Split the image into separate color channels
    b, g, r = cv2.split(img)

    # --------------------------------------------------------

    # -- Histogram Equalization --

    # Apply histogram equalization to each color channel
    eq_b = cv2.equalizeHist(b)
    eq_g = cv2.equalizeHist(g)
    eq_r = cv2.equalizeHist(r)

    eq_img = cv2.merge((eq_b, eq_g, eq_r))  # Merge the color channels back into a single image

    # --------------------------------------------------------

    # -- Contrast Limited Adaptive Histogram Equalization --

    # Apply CLAHE to each color channel
    # clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
    # cl_b = clahe.apply(b)
    # cl_g = clahe.apply(g)
    # cl_r = clahe.apply(r)

    # cl_img = cv2.merge((cl_b, cl_g, cl_r))    # Merge the color channels back into a single image

    # --------------------------------------------------------

    # Display the resulting images
    # cv2.imshow('Original Image', img)
    # cv2.imshow('Equalized Image', eq_img)
    # cv2.imshow('CLAHE Image', cl_img)

    # cv2.waitKey(0)     # Wait for a key press
    # cv2.destroyAllWindows()    # Close all windows

    return eq_img


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


def apply_night_vision(img):
    """
    Applies night vision mode to an image.
    (applies night_vision_core() function to the image)

    Args:
        img (torch.Tensor): The image to apply night vision mode to. (shape: (3, H, W))

    Returns:
        torch.Tensor: The image with night vision mode applied. (shape: (3, H, W))
    """
    img = convert_to_cv2(img)  # convert to cv2
    img = night_vision_core(img)  # apply night vision
    img = convert_to_torch(img)  # convert back to torch
    return img


# Load the image
# img = cv2.imread(r'D:\ME\Downloads\New folder\Downloads\dark_stop_3.jpg')
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
