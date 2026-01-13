import ultralytics, os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)

import torch
import torch.nn as nn
import open_clip
import copy

def reparameterize_model(model):
    """
    Reparameterize the model for inference.
    """
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    return model

def patch_text_global_pool():
    """
    Patch the text_global_pool function to use device-agnostic operations
    This must be done BEFORE creating the model
    """
    import open_clip.transformer as transformer_module
    
    # Save original function
    original_text_global_pool = transformer_module.text_global_pool
    
    def text_global_pool_patched(x, tokens, pool_type="argmax", eos_token_id=None):
        """
        Device-agnostic version using torch.gather to avoid device mismatch
        This approach works better with TorchScript tracing
        """
        if pool_type == "argmax":
            # Use gather instead of advanced indexing to avoid device issues
            token_indices = tokens.argmax(dim=-1)  # [batch_size]
            token_indices = token_indices.unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1]
            token_indices = token_indices.expand(-1, -1, x.shape[-1])  # [batch_size, 1, hidden_dim]
            x = torch.gather(x, 1, token_indices).squeeze(1)  # [batch_size, hidden_dim]
        elif pool_type == "eos":
            if eos_token_id is None:
                eos_token_id = tokens.argmax(dim=-1)
            eos_token_id = eos_token_id.unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1]
            eos_token_id = eos_token_id.expand(-1, -1, x.shape[-1])  # [batch_size, 1, hidden_dim]
            x = torch.gather(x, 1, eos_token_id).squeeze(1)  # [batch_size, hidden_dim]
        else:
            x = x.mean(dim=1)
        return x
    
    # Replace the function
    transformer_module.text_global_pool = text_global_pool_patched
    print("✓ Patched text_global_pool for device compatibility")
    
    return original_text_global_pool

class TextModelWithNormalization(nn.Module):
    """Wrapper that includes L2 normalization in the TorchScript model"""
    def __init__(self, text_model):
        super().__init__()
        self.text_model = text_model
    
    def forward(self, tokens):
        # Get text features
        features = self.text_model(tokens)
        # L2 normalize
        features = features / features.norm(dim=-1, keepdim=True)
        return features

def convert_mobileclip_to_torchscript(pt_path, output_path, device="cuda:0"):
    """
    Convert MobileCLIP .pt model to TorchScript .ts format
    with proper device handling for cross-device compatibility
    
    Args:
        pt_path: Path to the .pt weights file
        output_path: Path to save the .ts file
        device: Device to use for conversion
    """
    print(f"Loading model from {pt_path}...")
    
    # Patch text_global_pool BEFORE loading the model
    original_func = patch_text_global_pool()
    
    try:
        # Load model
        model_name = "MobileCLIP2-B"
        model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}
        
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pt_path, **model_kwargs
        )
        
        # Set to eval mode and reparameterize
        model.eval()
        model = reparameterize_model(model)
        model.to(device)
        
        print("Converting to TorchScript...")
        
        # Get text model and wrap with normalization
        text_model = model.text
        text_model_with_norm = TextModelWithNormalization(text_model)
        text_model_with_norm.eval()
        
        # Create example input for tracing
        tokenizer = open_clip.get_tokenizer(model_name)
        example_texts = ["a photo of a cat", "a dog", "person"]
        example_tokens = tokenizer(example_texts).to(device)
        
        print("Tracing model on original device...")
        # Trace the model
        with torch.no_grad():
            traced_model = torch.jit.trace(text_model_with_norm, example_tokens)
            
            # Optimize for inference
            traced_model = torch.jit.optimize_for_inference(traced_model)
        
        # Save
        print(f"Saving to {output_path}...")
        torch.jit.save(traced_model, output_path)
        
        # Verify on original device
        print("\nVerifying conversion on original device...")
        _verify_conversion(traced_model, example_tokens, text_model_with_norm, device)
        
        # Verify on CPU
        print("\nVerifying conversion on CPU...")
        loaded_model_cpu = torch.jit.load(output_path, map_location="cpu")
        example_tokens_cpu = example_tokens.to("cpu")
        text_model_cpu = TextModelWithNormalization(text_model.to("cpu"))
        _verify_conversion(loaded_model_cpu, example_tokens_cpu, text_model_cpu, "cpu")
        
        return traced_model
    
    finally:
        # Restore original function
        import open_clip.transformer as transformer_module
        transformer_module.text_global_pool = original_func

def _verify_conversion(traced_model, example_tokens, reference_model, device):
    """Helper function to verify conversion on a specific device"""
    with torch.no_grad():
        # Get reference output
        original_output = reference_model(example_tokens)
        
        # Get traced output
        traced_output = traced_model(example_tokens)
        
        diff_trace = (original_output - traced_output).abs().max().item()
        
        print(f"Max difference (reference vs traced): {diff_trace:.2e}")
        
        # Check if output is normalized
        traced_norm = traced_output.norm(dim=-1)
        print(f"Output norm (should be ~1.0): min={traced_norm.min():.6f}, max={traced_norm.max():.6f}")
        
        if diff_trace < 1e-5 and (traced_norm - 1.0).abs().max() < 1e-5:
            print(f"✓ Conversion successful on {device}! Model includes normalization.")
        elif diff_trace < 1e-3:
            print(f"⚠ Conversion successful on {device} with small numerical differences.")
        else:
            print(f"✗ Warning: Significant differences detected on {device}!")

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    pt_path = "mobileclip2_b.pt"
    output_path = "mobileclip2_b_new.ts"
    
    if not os.path.exists(pt_path):
        print(f"Error: {pt_path} not found!")
        exit(1)
    
    convert_mobileclip_to_torchscript(pt_path, output_path, device)
    
    print("\n" + "="*60)
    print("Testing cross-device compatibility...")
    print("="*60)
    
    # Compare with original if exists
    if os.path.exists("mobileclip2_b.ts"):
        print("\nLoading models...")
        original_ts_cuda = torch.jit.load("mobileclip2_b.ts", map_location=device)
        original_ts_cpu = torch.jit.load("mobileclip2_b.ts", map_location="cpu")
        
        new_ts_cuda = torch.jit.load(output_path, map_location=device)
        new_ts_cpu = torch.jit.load(output_path, map_location="cpu")
        
        # Test with same inputs
        tokenizer = open_clip.get_tokenizer("MobileCLIP2-B")
        test_texts = ["a photo of a cat", "a photo of a dog", "car", "person walking"]
        
        print("\n--- Testing on CUDA ---")
        tokens_cuda = tokenizer(test_texts).to(device)
        with torch.no_grad():
            try:
                output_original_cuda = original_ts_cuda(tokens_cuda)
                output_new_cuda = new_ts_cuda(tokens_cuda)
                diff_cuda = (output_original_cuda - output_new_cuda).abs()
                print(f"Max difference (old vs new): {diff_cuda.max().item():.2e}")
                print(f"Mean difference (old vs new): {diff_cuda.mean().item():.2e}")
                if diff_cuda.max().item() < 1e-3:
                    print("✓ CUDA outputs match")
            except Exception as e:
                print(f"Error on CUDA: {type(e).__name__}: {str(e)[:100]}")
        
        print("\n--- Testing on CPU ---")
        tokens_cpu = tokenizer(test_texts).to("cpu")
        with torch.no_grad():
            try:
                output_original_cpu = original_ts_cpu(tokens_cpu)
                print(f"✓ Original .ts works on CPU")
            except Exception as e:
                print(f"✗ Original .ts fails on CPU: {type(e).__name__}")
            
            try:
                output_new_cpu = new_ts_cpu(tokens_cpu)
                print(f"✓ New .ts works on CPU")
                
                if 'output_original_cpu' in locals():
                    diff_cpu = (output_original_cpu - output_new_cpu).abs()
                    print(f"Max difference (old vs new): {diff_cpu.max().item():.2e}")
                    print(f"Mean difference (old vs new): {diff_cpu.mean().item():.2e}")
                    if diff_cpu.max().item() < 1e-3:
                        print("✓ CPU outputs match")
            except Exception as e:
                print(f"✗ New .ts fails on CPU: {type(e).__name__}: {str(e)[:100]}")
    
    print(f"\nConversion complete! New file saved as: {output_path}")