#!/usr/bin/env python3
"""
Convert MobileCLIP2 model to TorchScript format for optimized inference.

Usage:
    python convert_mobileclip2_to_torchscript.py
"""

import os
import torch
import open_clip
import copy


def reparameterize_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Reparameterize model by converting multi-branched structure to single branch.
    
    Args:
        model: Model in train mode with multi-branch structure.
    
    Returns:
        Model in inference mode with reparameterized single branch.
    """
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    return model


class TextEncoderWrapper(torch.nn.Module):
    """Wrapper module for text encoder to enable TorchScript tracing."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, text):
        return self.model.encode_text(text)


def convert_to_torchscript(
    pretrained_path: str = "mobileclip2_b.pt",
    output_path: str = "mobileclip2_b.ts",
    device: str = "cpu"
):
    """
    Convert MobileCLIP2 model to TorchScript format.
    
    Args:
        pretrained_path: Path to the pretrained .pt file
        output_path: Path to save the TorchScript .ts file
        device: Device to use for conversion ('cpu' or 'cuda')
    """
    
    print(f"Loading model from {pretrained_path}...")
    
    # Check if pretrained model exists
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(
            f"Pretrained model not found at {pretrained_path}. "
            f"Please download mobileclip2_b.pt first."
        )
    
    # Get model name from file
    pretrained_name = os.path.basename(pretrained_path)
    model_name = {"mobileclip2_b.pt": "MobileCLIP2-B"}[pretrained_name]
    
    # Set model kwargs
    model_kwargs = {}
    if not (model_name.endswith("S3") or model_name.endswith("S4") or model_name.endswith("L-14")):
        model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}
    
    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained_path, **model_kwargs
    )
    
    # Prepare model for inference
    model.eval()
    model = reparameterize_model(model)
    model = model.to(device)
    
    print(f"Model loaded on {device}")
    
    # Create dummy input for tracing text encoder
    # MobileCLIP2 uses standard CLIP tokenizer with max length 77
    dummy_text_input = torch.randint(0, 49408, (1, 77)).to(device)
    
    print("Tracing text encoder...")
    
    # Wrap the model's encode_text method
    text_encoder_wrapper = TextEncoderWrapper(model).to(device)
    text_encoder_wrapper.eval()
    
    # Trace the text encoder wrapper
    with torch.no_grad():
        traced_text_encoder = torch.jit.trace(
            text_encoder_wrapper,
            dummy_text_input,
            check_trace=True
        )
    
    # Save the traced model
    print(f"Saving TorchScript model to {output_path}...")
    torch.jit.save(traced_text_encoder, output_path)
    
    print("✓ Conversion successful!")
    
    # Test the saved model
    print("\nTesting saved TorchScript model...")
    loaded_model = torch.jit.load(output_path, map_location=device)
    
    with torch.no_grad():
        original_output = model.encode_text(dummy_text_input)
        traced_output = loaded_model(dummy_text_input)
        
        # Check if outputs match
        max_diff = torch.max(torch.abs(original_output - traced_output)).item()
        print(f"Maximum difference between original and traced: {max_diff:.6e}")
        
        if max_diff < 1e-5:
            print("✓ Outputs match! Model converted successfully.")
        else:
            print(f"⚠ Warning: Outputs differ by {max_diff:.6e}")
    
    # Print model info
    print(f"\nModel info:")
    print(f"  Input file: {pretrained_path}")
    print(f"  Output file: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    return output_path


def test_with_openclip_wrapper(pt_path: str = "mobileclip2_b.pt", ts_path: str = "mobileclip2_b.ts", device: str = "cpu"):
    """
    Test both .pt and .ts models using OpenCLIP wrapper from text_model.py
    
    Args:
        pt_path: Path to .pt file
        ts_path: Path to .ts file  
        device: Device to run on
    """
    import sys
    sys.path.insert(0, "./ultralytics/nn")
    from ultralytics.nn.text_model import OpenCLIP, MobileCLIPTS
    
    print("\n" + "="*60)
    print("Testing with OpenCLIP wrapper from text_model.py")
    print("="*60)
    
    # Load PT model
    print(f"\n1. Loading .pt model with OpenCLIP wrapper...")
    pt_model = OpenCLIP(device=torch.device(device), pretrained_path=pt_path)
    
    # Create test text inputs
    test_texts = [
        "a photo of a cat",
        "a photo of a dog", 
        "a person riding a bicycle"
    ]
    
    print(f"2. Test texts: {test_texts}")
    
    # Tokenize
    tokens = pt_model.tokenize(test_texts)
    print(f"3. Tokenized shape: {tokens.shape}")
    
    # Encode with PT model
    print(f"\n4. Encoding with .pt model...")
    with torch.no_grad():
        pt_features = pt_model.encode_text(tokens)
    print(f"   Output shape: {pt_features.shape}")
    print(f"   Output dtype: {pt_features.dtype}")
    print(f"   First feature vector norm: {pt_features[0].norm().item():.6f}")
    
    # Load TS model directly
    print(f"\n5. Loading .ts TorchScript model...")
    ts_model = MobileCLIPTS(clip_weight_name=ts_path, device=torch.device(device))
    ts_model.eval()
    
    # Encode with TS model
    print(f"6. Encoding with .ts model...")
    with torch.no_grad():
        ts_features = ts_model.encode_text(tokens)
        # Normalize like OpenCLIP does
        ts_features = ts_features / ts_features.norm(dim=-1, keepdim=True)
    print(f"   Output shape: {ts_features.shape}")
    print(f"   Output dtype: {ts_features.dtype}")
    print(f"   First feature vector norm: {ts_features[0].norm().item():.6f}")
    
    # Compare outputs
    print(f"\n7. Comparing outputs...")
    max_diff = torch.max(torch.abs(pt_features - ts_features)).item()
    mean_diff = torch.mean(torch.abs(pt_features - ts_features)).item()
    
    print(f"   Maximum difference: {max_diff:.6e}")
    print(f"   Mean difference: {mean_diff:.6e}")
    
    # Check similarity per text
    print(f"\n8. Per-text comparison:")
    for i, text in enumerate(test_texts):
        diff = torch.abs(pt_features[i] - ts_features[i]).max().item()
        cosine_sim = torch.cosine_similarity(pt_features[i:i+1], ts_features[i:i+1]).item()
        print(f"   [{i}] '{text}'")
        print(f"       Max diff: {diff:.6e}, Cosine similarity: {cosine_sim:.8f}")
    
    # Final verdict
    print(f"\n{'='*60}")
    if max_diff < 1e-5:
        print("✓ SUCCESS: .pt and .ts models produce identical outputs!")
    elif max_diff < 1e-3:
        print("✓ PASS: .pt and .ts models produce very similar outputs.")
        print(f"  (Small numerical differences expected: {max_diff:.6e})")
    else:
        print(f"⚠ WARNING: Outputs differ by {max_diff:.6e}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert MobileCLIP2 to TorchScript")
    parser.add_argument(
        "--input",
        type=str,
        default="mobileclip2_b.pt",
        help="Path to input .pt file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mobileclip2_b.ts",
        help="Path to output .ts file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for conversion"
    )
    
    args = parser.parse_args()
    
    try:
        output_path = convert_to_torchscript(
            pretrained_path=args.input,
            output_path=args.output,
            device=args.device
        )
        
        # Test with OpenCLIP wrapper
        output_path=args.output
        test_with_openclip_wrapper(
            pt_path=args.input,
            ts_path=output_path,
            device=args.device
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
