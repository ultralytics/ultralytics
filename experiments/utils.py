from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt


def show_image(data, title="Image", path: str = None, scale: float = 7, dpi=300, inline=True):
    sizes = np.shape(data)
    print(sizes)
    fig = plt.figure()
    fig.set_size_inches(scale * sizes[1] / sizes[0], scale)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data)
    plt.suptitle(title, fontweight='bold')

    if inline:
        plt.show(dpi=dpi)
    else:
        data.show()

    if Path is not None:
        print(f"saving to {Path(path)}")
        title = title.replace(".", "_")
        fig.savefig(Path(path).joinpath(f"{title}.png"))


# FGSM attack code
def fgsm_attack(img, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = img + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # perturbed_image = torch.clamp(perturbed_image, 0, 255)
    # Return the perturbed image
    return perturbed_image


# restores the tensors to their original scale
def denorm(batch, mean=[0.1307], std=[0.3081], device="cpu"):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def test(model, device, test_loader, epsilon, batch_size):
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = torch.max(output, 1)  # get the index of the max log-probability

        # If the initial prediction is wrong, don't bother attacking, just move on
        # if init_pred != target:
        #     continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect ``datagrad``
        data_grad = data.grad.data

        # Restore the data to its original scale
        # data_denorm = denorm(data_grad)

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Reapply normalization
        # perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
        perturbed_data_normalized = perturbed_data

        # Re-classify the perturbed image
        output = model(perturbed_data_normalized)

        # Check for success
        final_pred = torch.max(output, 1)  # get the index of the max log-probability
        for k in range(batch_size):
            fp = final_pred.indices
            ip = init_pred.indices
            tp = target

            if fp == tp:
                correct += 1
                # Special case for saving 0 epsilon examples
                if epsilon == 0 and len(adv_examples) < 8:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((ip, fp, adv_ex))
            else:
                # Save some adv examples for visualization later
                if len(adv_examples) < 8:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((ip, fp, adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")
    # Return the accuracy and an adversarial example
    return final_acc, adv_examples
