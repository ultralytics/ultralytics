import numpy as np
import torch
import torchvision
import torchvision.transforms as T

def fliplr(img):
    """ flip a batch of images horizontally """
    # N x C x H x W
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    img_flip = img.index_select(3, inv_idx)
    return img_flip

def extract_image_patch(image, bbox, patch_shape=None):
    """Extract an image patch (defined by a bounding box) from an image
    Parameters
    ----------
    image : torch.tensor | ndarray
        The full image (3 dimensions, with an RGB or grayscale channel).
    bbox : array_like
        The bounding box in format (tx, ty, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.
    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.
    """

    if len(image.shape) != 3:
        raise NotImplementedError(
            "Only image arrays of 3 dimensions are supported.")

    if image.shape[0] <= 3:
        ch, h, w = image.shape
    elif image.shape[2] <= 3:
        h, w, ch = image.shape
    else:
        raise ValueError(
            "Input image does not contain an RGB or gray channel.")

    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int32)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum([w, h], bbox[2:])

    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox

    if image.shape[0] <= 3:
        image = image[:, sy:ey, sx:ex]
    else:
        image = image[sy:ey, sx:ex, :]
    return image


def create_extractor(extractor_class, batch_size=32, image_shape=(224, 224), **kwargs):
    image_encoder = extractor_class(**kwargs)
    img_transform = T.Compose([T.ToTensor(), T.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    version = list(map(int, torchvision.__version__.split(".")[:2]))
    if version[0] == 0 and version[1] < 13:
        patch_transform = T.Resize(image_shape, interpolation=3)
    else:
        patch_transform = T.Resize(
            image_shape, interpolation=T.InterpolationMode.BICUBIC)

    def encoder(image, boxes):
        if len(boxes) == 0:
            return np.array([])
        image = img_transform(image)
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box)
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = torch.rand((3, image_shape[0], image_shape[1]))
            patch = patch_transform(patch)
            image_patches.append(patch)

        image_patches = torch.stack(image_patches)
        return image_encoder(image_patches, batch_size)

    return encoder


def fliplr(img):
    """ flip a batch of images horizontally """
    # N x C x H x W
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    img_flip = img.index_select(3, inv_idx)
    return img_flip


class FeatureExtractor:
    def __init__(self, model, device, feature_dim="infer"):
        self.device = device
        self.model = model
        self.model.eval()
        self.model.to(self.device)
        self.model = torch.compile(self.model)
        self.device = next(iter(model.parameters())).device
        self.dtype = next(iter(model.parameters())).dtype
        self.feature_dim = feature_dim

    def __call__(self, X, batch_size=32):
        X = X.type(self.dtype)
        if self.feature_dim == "infer":
            dummy = X[0].unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(dummy)
            self.feature_dim = output.shape[1]

        out = np.zeros((len(X), self.feature_dim), np.float32)

        for i in range(0, len(X), batch_size):
            imax = min(len(X), i + batch_size)
            X_in = X[i:imax]
            X_in_flip = fliplr(X_in)
            with torch.no_grad():
                Y = self.model(X_in.to(self.device))
                Y_flip = self.model(X_in_flip.to(self.device))
            Y += Y_flip
            Y_norm = torch.norm(Y, p=2, dim=1, keepdim=True)
            Y = Y.div(Y_norm.expand_as(Y)).to("cpu")
            out[i:imax, :] = Y
        return out
