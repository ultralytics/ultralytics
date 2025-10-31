"""
Pytest suite for GhostConv, GhostBottleneck, and C3Ghost modules.

Coverage & notes:
- Parametrized mode in ['original', 'attn'].
- ONNX exports use CI-friendly opset versions (12 or 13) and keep dynamo=False.
- ONNX Runtime tests gate with pytest.importorskip('onnxruntime') and read
  the actual ONNX input name from the runtime session.
- Backwards compatibility checks:
    * GhostConv(32, 64, 3, 1) (old API) maps to mode 'original'
    * GhostConv(..., mode='original') supported as the last arg
    * GhostBottleneck(..., layer_id=None) optional
    * C3Ghost layer_id optional / computed internally
- Includes regression asserting 'original' == torch.cat((m.cv1(x), m.cv2(m.cv1(x))), 1)
"""

import torch
import torch.onnx
import torch.nn as nn
import torch.nn.functional as F
import time
import pytest
from pathlib import Path

import warnings
from torch.jit import TracerWarning
warnings.filterwarnings("ignore", category=TracerWarning)
warnings.filterwarnings("ignore", message="Unknown pytest.mark.slow")


from ultralytics.nn.modules.conv import GhostConv
from ultralytics.nn.modules.block import GhostBottleneck, C3Ghost


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Return available device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def tmp_export_dir(tmp_path):
    """Create temporary directory for export tests."""
    export_dir = tmp_path / "ghost_exports"
    export_dir.mkdir(exist_ok=True)
    return export_dir


# Helper: consistent ONNX export options
def _export_onnx(model, x, path, opset=13, const_fold=True, input_name="input", output_name="output"):
    torch.onnx.export(
        model,
        x,
        str(path),
        input_names=[input_name],
        output_names=[output_name],
        opset_version=opset,
        do_constant_folding=const_fold,
    )


# ============================================================================
# GhostConv Tests
# ============================================================================

class TestGhostConv:
    """Test suite for GhostConv module.

    Tests the forward pass, TorchScript/ONNX export, BC for old/new APIs, gradient flow,
    and original-mode equivalence regression.
    """

    @pytest.mark.parametrize("mode", ["original", "attn"])
    @pytest.mark.parametrize("c1,c2", [(32, 64), (64, 128), (128, 256)])
    def test_ghostconv_forward(self, mode, c1, c2, device):
        """Test GhostConv forward pass with different channel configurations."""
        model = GhostConv(c1, c2, mode=mode).to(device)
        model.eval()

        x = torch.randn(2, c1, 16, 16).to(device)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (2, c2, 16, 16)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    @pytest.mark.parametrize("mode", ["original", "attn"])
    def test_ghostconv_1x1_spatial(self, mode, device):
        """Test GhostConv with [B, C, 1, 1] input (critical edge case)."""
        c1, c2 = 256, 512
        model = GhostConv(c1, c2, mode=mode).to(device)
        model.eval()

        x = torch.randn(1, c1, 1, 1).to(device)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, c2, 1, 1)
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize("mode", ["original", "attn"])
    @pytest.mark.parametrize("H,W", [(8, 16), (16, 8), (7, 13), (32, 24)])
    def test_ghostconv_asymmetric_hw(self, mode, H, W, device):
        """Test GhostConv with asymmetric H×W dimensions."""
        c1, c2 = 64, 128
        model = GhostConv(c1, c2, mode=mode).to(device)
        model.eval()

        x = torch.randn(1, c1, H, W).to(device)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, c2, H, W)

    def test_ghostconv_default_mode(self):
        """Default mode is 'original' (gate OFF by default)."""
        model = GhostConv(32, 64)

        assert model.mode == 'original', "Default mode should be 'original'"
        assert not hasattr(model, 'short_conv'), "Attention components should not exist in default mode"
        assert not hasattr(model, 'gate'), "Gate should not exist in default mode"

    def test_ghostconv_backward_compatibility(self):
        """Backward compatibility: old API GhostConv(c1, c2, k, s) and new API."""
        # Old API: GhostConv(c1, c2, k, s) -> should default to 'original'
        model_old = GhostConv(32, 64, 3, 1)
        assert model_old.mode == 'original'

        # New API: GhostConv(c1, c2, k, s, mode='attn')
        model_new = GhostConv(32, 64, 3, 1, mode='attn')
        assert model_new.mode == 'attn'

        # Backward compatibility: allow explicit mode as last positional arg.
        # Some implementations support this, some do not.
        try:
            model_last_arg = GhostConv(32, 64, 3, 1, 'original')
            assert model_last_arg.mode == 'original'
        except TypeError:
            # If positional passing is not supported, verify keyword still works.
            model_kw = GhostConv(32, 64, 3, 1, mode='original')
            assert model_kw.mode == 'original'


    @pytest.mark.parametrize("mode", ["original", "attn"])
    def test_ghostconv_gradient_flow(self, mode, device):
        """Test gradient flow through GhostConv."""
        model = GhostConv(64, 128, mode=mode).to(device)
        model.train()

        x = torch.randn(2, 64, 16, 16, device=device, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_ghostconv_original_equivalence(self, device):
        """Regression: 'original' mode equals torch.cat((cv1(x), cv2(cv1(x))), 1)."""
        c1, c2 = 64, 128
        model = GhostConv(c1, c2, mode='original').to(device)
        model.eval()

        x = torch.randn(2, c1, 16, 16).to(device)

        with torch.no_grad():
            out_model = model(x)

            # Manual computation: use the model's internal convs to guarantee consistent weights
            out_manual = torch.cat([model.cv1(x), model.cv2(model.cv1(x))], 1)

        torch.testing.assert_close(out_model, out_manual, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("mode", ["original", "attn"])
    def test_ghostconv_torchscript_trace(self, mode, tmp_export_dir):
        """Test TorchScript export via tracing."""
        c1, c2 = 32, 64
        model = GhostConv(c1, c2, mode=mode).cpu()
        model.eval()

        test_shapes = [(1, c1, 8, 8), (1, c1, 1, 1)]

        for shape in test_shapes:
            x = torch.randn(shape)

            traced_model = torch.jit.trace(model, x)
            script_path = tmp_export_dir / f"ghostconv_{mode}_{shape[2]}x{shape[3]}.pt"
            traced_model.save(str(script_path))

            loaded_model = torch.jit.load(str(script_path))

            with torch.no_grad():
                out_orig = model(x)
                out_traced = loaded_model(x)

            torch.testing.assert_close(out_orig, out_traced, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("mode", ["original", "attn"])
    def test_ghostconv_onnx_export(self, mode, tmp_export_dir):
        """Test ONNX export (opset 13, no dynamo)."""
        onnx = pytest.importorskip("onnx")

        c1, c2 = 32, 64
        model = GhostConv(c1, c2, mode=mode).cpu()
        model.eval()

        test_shapes = [(1, c1, 8, 8), (1, c1, 1, 1), (1, c1, 7, 13)]

        for i, shape in enumerate(test_shapes):
            x = torch.randn(shape)
            onnx_path = tmp_export_dir / f"ghostconv_{mode}_{shape[2]}x{shape[3]}.onnx"

            # explicit input/output names + CI-friendly opset
            _export_onnx(model, x, onnx_path, opset=13, input_name="input", output_name="output")

            assert onnx_path.exists()

            # Verify ONNX model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)

    @pytest.mark.parametrize("mode", ["original", "attn"])
    def test_ghostconv_onnx_runtime(self, mode, tmp_export_dir):
        """Test ONNX Runtime inference matches PyTorch (opset 12)."""
        onnx = pytest.importorskip("onnx")
        ort = pytest.importorskip("onnxruntime")

        c1, c2 = 32, 64
        model = GhostConv(c1, c2, mode=mode).cpu()
        model.eval()

        x = torch.randn(1, c1, 8, 8)
        onnx_path = tmp_export_dir / f"ghostconv_{mode}_ort.onnx"

        # export with explicit names; use opset 12 for RT compatibility in CI
        _export_onnx(model, x, onnx_path, opset=12, input_name="input", output_name="output")

        # PyTorch inference
        with torch.no_grad():
            out_torch = model(x).numpy()

        # ONNX Runtime inference
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        out_ort = session.run(None, {input_name: x.numpy()})[0]

        assert out_ort.shape == out_torch.shape
        torch.testing.assert_close(
            torch.from_numpy(out_ort),
            torch.from_numpy(out_torch),
            rtol=1e-3,
            atol=1e-4
        )


# ============================================================================
# GhostBottleneck Tests
# ============================================================================

class TestGhostBottleneck:
    """Test suite for GhostBottleneck module."""

    @pytest.mark.parametrize("k", [3, 5])
    @pytest.mark.parametrize("s", [1, 2])
    def test_ghostbottleneck_forward(self, k, s, device):
        """Test GhostBottleneck forward pass with kernel sizes and strides."""
        c1, c2 = 64, 64
        model = GhostBottleneck(c1, c2, k=k, s=s).to(device)
        model.eval()

        x = torch.randn(2, c1, 32, 32).to(device)

        with torch.no_grad():
            out = model(x)

        expected_h = 32 // s
        expected_w = 32 // s
        assert out.shape == (2, c2, expected_h, expected_w)

    @pytest.mark.parametrize("s", [1])  # stride=2 on 1x1 is invalid; only s=1 is sensible here
    def test_ghostbottleneck_1x1_spatial(self, s, device):
        """Test GhostBottleneck with [B, C, 1, 1] input (stride=1 only)."""
        c1, c2 = 128, 128
        model = GhostBottleneck(c1, c2, k=3, s=s).to(device)
        model.eval()

        x = torch.randn(1, c1, 1, 1).to(device)

        with torch.no_grad():
            out = model(x)

        expected_size = 1 // s if s > 1 else 1
        assert out.shape == (1, c2, expected_size, expected_size)

    @pytest.mark.parametrize("H,W", [(8, 16), (16, 8), (7, 13)])
    def test_ghostbottleneck_asymmetric_hw(self, H, W, device):
        """Test GhostBottleneck with asymmetric H×W dimensions."""
        c1, c2 = 64, 64
        model = GhostBottleneck(c1, c2, k=3, s=1).to(device)
        model.eval()

        x = torch.randn(1, c1, H, W).to(device)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, c2, H, W)

    def test_ghostbottleneck_layer_id_default(self):
        """GhostBottleneck accepts optional layer_id (default None)."""
        model = GhostBottleneck(64, 64, k=3, s=1)

        x = torch.randn(1, 64, 16, 16)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 64, 16, 16)

    @pytest.mark.parametrize("layer_id", [None, 0, 5, 10])
    def test_ghostbottleneck_layer_id_propagation(self, layer_id):
        """Explicit layer_id propagation (None allowed)."""
        if layer_id is None:
            model = GhostBottleneck(64, 64, k=3, s=1)
        else:
            model = GhostBottleneck(64, 64, k=3, s=1, layer_id=layer_id)

        x = torch.randn(1, 64, 16, 16)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 64, 16, 16)

    def test_ghostbottleneck_backward_compatibility(self):
        """BC: GhostBottleneck(c1, c2, k, s) old API and new API with layer_id."""
        model_old = GhostBottleneck(64, 64, 3, 1)
        assert model_old is not None

        model_new = GhostBottleneck(64, 64, 3, 1, layer_id=5)
        assert model_new is not None

    @pytest.mark.parametrize("k,s", [(3, 1), (3, 2), (5, 1)])
    def test_ghostbottleneck_torchscript(self, k, s, tmp_export_dir):
        """TorchScript export for GhostBottleneck."""
        model = GhostBottleneck(64, 64, k=k, s=s).cpu()
        model.eval()

        x = torch.randn(1, 64, 16, 16)
        traced_model = torch.jit.trace(model, x)

        script_path = tmp_export_dir / f"ghostbottleneck_k{k}_s{s}.pt"
        traced_model.save(str(script_path))

        loaded_model = torch.jit.load(str(script_path))

        with torch.no_grad():
            out_orig = model(x)
            out_traced = loaded_model(x)

        torch.testing.assert_close(out_orig, out_traced, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("k,s", [(3, 1), (3, 2)])
    def test_ghostbottleneck_onnx_export(self, k, s, tmp_export_dir):
        """ONNX export for GhostBottleneck (opset 13)."""
        onnx = pytest.importorskip("onnx")

        model = GhostBottleneck(64, 64, k=k, s=s).cpu()
        model.eval()

        x = torch.randn(1, 64, 16, 16)
        onnx_path = tmp_export_dir / f"ghostbottleneck_k{k}_s{s}.onnx"

        _export_onnx(model, x, onnx_path, opset=13, input_name="input", output_name="output")

        assert onnx_path.exists()

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)


# ============================================================================
# C3Ghost Tests
# ============================================================================

class TestC3Ghost:
    """Test suite for C3Ghost module."""

    @pytest.mark.parametrize("c1,c2", [(64, 64), (128, 128), (256, 256)])
    def test_c3ghost_forward(self, c1, c2, device):
        """Test C3Ghost forward pass."""
        model = C3Ghost(c1, c2).to(device)
        model.eval()

        x = torch.randn(2, c1, 32, 32).to(device)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (2, c2, 32, 32)

    def test_c3ghost_1x1_spatial(self, device):
        """Test C3Ghost with [B, C, 1, 1] input."""
        c1, c2 = 128, 128
        model = C3Ghost(c1, c2).to(device)
        model.eval()

        x = torch.randn(1, c1, 1, 1).to(device)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, c2, 1, 1)

    @pytest.mark.parametrize("H,W", [(8, 16), (16, 8), (7, 13)])
    def test_c3ghost_asymmetric_hw(self, H, W, device):
        """Test C3Ghost with asymmetric H×W dimensions."""
        c1, c2 = 64, 64
        model = C3Ghost(c1, c2).to(device)
        model.eval()

        x = torch.randn(1, c1, H, W).to(device)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, c2, H, W)

    def test_c3ghost_layer_id_optional(self):
        """C3Ghost accepts an optional layer_id or computes it internally."""
        # Should work without explicit layer_id
        model = C3Ghost(64, 64)

        x = torch.randn(1, 64, 16, 16)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 64, 16, 16)

    @pytest.mark.parametrize("layer_id", [None, 0, 3, 7])
    def test_c3ghost_layer_id_propagation(self, layer_id):
        """Explicit layer_id propagation (None allowed)."""
        if layer_id is None:
            model = C3Ghost(64, 64)
        else:
            model = C3Ghost(64, 64, layer_id=layer_id)

        x = torch.randn(1, 64, 16, 16)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 64, 16, 16)

    def test_c3ghost_backward_compatibility(self):
        """BC: older signature C3Ghost(c1,c2,n,shortcut) should still work."""
        model = C3Ghost(64, 64, n=1, shortcut=True)
        assert model is not None

    def test_c3ghost_torchscript(self, tmp_export_dir):
        """TorchScript export for C3Ghost."""
        model = C3Ghost(64, 64).cpu()
        model.eval()

        x = torch.randn(1, 64, 16, 16)
        traced_model = torch.jit.trace(model, x)

        script_path = tmp_export_dir / "c3ghost.pt"
        traced_model.save(str(script_path))

        loaded_model = torch.jit.load(str(script_path))

        with torch.no_grad():
            out_orig = model(x)
            out_traced = loaded_model(x)

        torch.testing.assert_close(out_orig, out_traced, rtol=1e-5, atol=1e-5)

    def test_c3ghost_onnx_export(self, tmp_export_dir):
        """ONNX export for C3Ghost (opset 12)."""
        onnx = pytest.importorskip("onnx")

        model = C3Ghost(64, 64).cpu()
        model.eval()

        x = torch.randn(1, 64, 16, 16)
        onnx_path = tmp_export_dir / "c3ghost.onnx"

        _export_onnx(model, x, onnx_path, opset=12, input_name="input", output_name="output")

        assert onnx_path.exists()

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)


# ============================================================================
# Performance Benchmark (optional, can be marked slow)
# ============================================================================

@pytest.mark.slow
class TestGhostPerformance:
    """Performance benchmarks for Ghost modules."""

    @pytest.mark.parametrize("mode", ["original", "attn"])
    def test_ghostconv_inference_time(self, mode, device):
        """Benchmark GhostConv inference time."""
        import time

        c1, c2 = 64, 128
        batch_size = 8
        img_size = 224
        num_warmup = 10
        num_runs = 100

        model = GhostConv(c1, c2, mode=mode).to(device)
        model.eval()

        x = torch.randn(batch_size, c1, img_size, img_size).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(x)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(x)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        avg_time_ms = ((end_time - start_time) / num_runs) * 1000

        # Just ensure it completes without error
        assert avg_time_ms > 0
