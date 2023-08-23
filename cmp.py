import numpy as np

# Load the saved .npy files
loaded_tensor1 = np.load("pred_off_0.npy")
loaded_tensor2 = np.load("preds_decoded.npy")

# Compare the values of the loaded tensors
if np.allclose(loaded_tensor1, loaded_tensor2, atol=1e-6):
    print("The values of the loaded tensors match.")
else:
    print("The values of the loaded tensors do not match.")