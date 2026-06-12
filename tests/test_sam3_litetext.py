# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Unit tests for SAM3-LiteText (MobileCLIP lightweight text encoder) integration.

Run with::

    pytest tests/test_sam3_litetext.py -v

Local checkpoints are required for the predictor test. Set the environment variable
``SAM3_LITETEXT_CKPT`` to the path of any ``efficient_sam3_text_*`` or ``sam3-litetext-*``
checkpoint, or place one at the default path used below.  All other tests run without weights.
"""

from __future__ import annotations

import os

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_CKPT = os.environ.get("SAM3_LITETEXT_CKPT", "")
_CKPT_AVAILABLE = bool(_DEFAULT_CKPT) and os.path.isfile(_DEFAULT_CKPT)

_TEST_IMAGE = os.path.join(os.path.dirname(__file__), "..", "..", "dog_person.jpeg")
# Fallback to any ultralytics asset if the bespoke image isn't there
if not os.path.isfile(_TEST_IMAGE):
    from ultralytics.utils import ASSETS

    _TEST_IMAGE = str(ASSETS / "bus.jpg")

requires_ckpt = pytest.mark.skipif(not _CKPT_AVAILABLE, reason="SAM3-LiteText checkpoint not found")


# ---------------------------------------------------------------------------
# 1. Detection helpers
# ---------------------------------------------------------------------------


class TestDetectionHelpers:
    """Tests for the filename-based auto-detection helpers in build_sam3."""

    def setup_method(self):
        """Load detection helper functions from build_sam3."""
        from ultralytics.models.sam.build_sam3 import _detect_litetext_backbone, _detect_litetext_context_length

        self.detect_backbone = _detect_litetext_backbone
        self.detect_ctx = _detect_litetext_context_length

    # --- backbone detection ---

    @pytest.mark.parametrize(
        "name,expected",
        [
            # Ultralytics naming convention
            ("sam3-litetext-s0.pt", "S0"),
            ("sam3-litetext-s1.pt", "S1"),
            ("sam3-litetext-l.pt", "L"),
            # Original EfficientSAM3 naming convention
            ("efficient_sam3_text_s0_ctx16_fixed.pt", "S0"),
            ("efficient_sam3_text_s1_ctx32_fixed.pt", "S1"),
            ("efficient_sam3_text_l_ctx16_fixed.pt", "L"),
            # HuggingFace Ultralytics-native naming (sam3_litetext_mobileclip_*)
            ("sam3_litetext_mobileclip_s0_ctx16.pt", "S0"),
            ("sam3_litetext_mobileclip_s0_ctx32.pt", "S0"),
            ("sam3_litetext_mobileclip_s1_ctx16.pt", "S1"),
            ("sam3_litetext_mobileclip_s1_ctx32.pt", "S1"),
            ("sam3_litetext_mobileclip2_l_ctx16.pt", "L"),
            ("sam3_litetext_mobileclip2_l_ctx32.pt", "L"),
            # Image-encoder-only checkpoints must NOT be detected as LiteText
            ("efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt", None),
            ("efficient_sam3_image_encoder_mobileclip_s1_ctx16.pt", None),
            ("efficient_sam3_image_encoder_mobileclip_2_l_ctx16.pt", None),
            # Standard SAM3 - must return None
            ("sam3.pt", None),
            ("sam2_b.pt", None),
            ("yolo11n.pt", None),
        ],
    )
    def test_detect_backbone(self, name, expected):
        """Verify backbone detection from the checkpoint filename."""
        assert self.detect_backbone(name) == expected

    def test_detect_backbone_ignores_directory_name(self):
        """Full paths containing 'litetext' in a *directory* should not be mistaken for a LiteText file."""
        full_path = f"/some/sam3_litetext/checkpoints/{__file__}/sam3.pt"
        assert self.detect_backbone(full_path) is None

    # --- context length detection ---

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("efficient_sam3_text_s0_ctx16_fixed.pt", 16),
            ("efficient_sam3_text_s0_ctx32_fixed.pt", 32),
            ("sam3-litetext-s0.pt", 16),  # no ctx tag → default 16
            ("sam3-litetext-l.pt", 16),
        ],
    )
    def test_detect_context_length(self, name, expected):
        """Verify context length detection from the checkpoint filename."""
        assert self.detect_ctx(name) == expected

    def test_detect_context_length_custom_default(self):
        """Verify the custom default is returned when context cannot be detected."""
        assert self.detect_ctx("sam3.pt", default=32) == 32


# ---------------------------------------------------------------------------
# 2. TextStudentEncoder - unit tests (no weights needed)
# ---------------------------------------------------------------------------


class TestTextStudentEncoder:
    """Tests for TextStudentEncoder instantiation and forward pass."""

    _S0_CFG = {
        "dim": 512,
        "model_name": "mct",
        "n_transformer_layers": 4,
        "n_heads_per_layer": 8,
        "ffn_multiplier_per_layer": 4.0,
        "norm_layer": "layer_norm_fp32",
        "context_length": 77,
        "vocab_size": 49408,
        "causal_masking": False,
    }

    def _make_encoder(self, context_length=16, output_dim=256):
        from ultralytics.models.sam.sam3.text_encoder_student import TextStudentEncoder

        return TextStudentEncoder(cfg=self._S0_CFG, context_length=context_length, output_dim=output_dim)

    def test_instantiation(self):
        """Verify TextStudentEncoder is instantiated with the correct context length."""
        enc = self._make_encoder()
        assert enc.context_length == 16

    def test_positional_embedding_shape(self):
        """Encoder is built at cfg['context_length'] (77); truncation happens via set_context_length()."""
        enc = self._make_encoder(context_length=16)
        pos_embed = enc.encoder.positional_embedding.pos_embed.pos_embed
        assert pos_embed.shape[2] == 77, f"Expected 77 (cfg size before truncation), got {pos_embed.shape[2]}"
        # After set_context_length the embedding is truncated to the operational length.
        enc.set_context_length(16)
        pos_embed = enc.encoder.positional_embedding.pos_embed.pos_embed
        assert pos_embed.shape[2] == 16, f"Expected 16 after set_context_length, got {pos_embed.shape[2]}"

    def test_set_context_length(self):
        """Verify set_context_length updates context_length and positional embedding."""
        enc = self._make_encoder(context_length=32)
        enc.set_context_length(16)
        assert enc.context_length == 16
        pos_embed = enc.encoder.positional_embedding.pos_embed.pos_embed
        assert pos_embed.shape[2] == 16

    def test_forward_output_shapes(self):
        """Verify forward pass returns correctly-shaped (mask, memory, embeds) tensors."""
        enc = self._make_encoder(context_length=16, output_dim=256)
        enc.eval()
        texts = ["a cat", "a dog"]
        with torch.no_grad():
            mask, memory, embeds = enc(texts, device=torch.device("cpu"))
        B, ctx = 2, 16
        assert mask.shape == (B, ctx), f"mask shape {mask.shape}"
        assert memory.shape == (ctx, B, 256), f"memory shape {memory.shape}"
        assert embeds.shape[1] == B, f"embeds dim-1 should be B={B}"

    def test_forward_mask_is_bool(self):
        """Verify the attention mask returned by forward() is a boolean tensor."""
        enc = self._make_encoder()
        enc.eval()
        with torch.no_grad():
            mask, _, _ = enc(["hello"], device=torch.device("cpu"))
        assert mask.dtype == torch.bool

    def test_forward_device_inference(self):
        """forward() should work without an explicit device argument."""
        enc = self._make_encoder(context_length=16)
        enc.eval()
        with torch.no_grad():
            mask, memory, _ = enc(["test prompt"])
        assert memory.shape[0] == 16


# ---------------------------------------------------------------------------
# 3. Build helpers - all three variants (no weights loaded)
# ---------------------------------------------------------------------------


class TestLitetextConfigs:
    """Test that _LITETEXT_CONFIGS covers all three variants with correct dims."""

    def test_configs_exist(self):
        """Verify _LITETEXT_CONFIGS contains all three backbone variants."""
        from ultralytics.models.sam.build_sam3 import _LITETEXT_CONFIGS

        assert set(_LITETEXT_CONFIGS) == {"S0", "S1", "L"}

    @pytest.mark.parametrize(
        "variant,expected_dim,expected_layers",
        [
            ("S0", 512, 4),
            ("S1", 512, 12),
            ("L", 768, 12),
        ],
    )
    def test_config_values(self, variant, expected_dim, expected_layers):
        """Verify embed dim and layer count for each backbone variant."""
        from ultralytics.models.sam.build_sam3 import _LITETEXT_CONFIGS

        cfg = _LITETEXT_CONFIGS[variant]
        assert cfg["dim"] == expected_dim
        assert cfg["n_transformer_layers"] == expected_layers


# ---------------------------------------------------------------------------
# 4. reparameterize() - no weights needed
# ---------------------------------------------------------------------------


class TestReparameterize:
    """Tests for reparameterize() on RepMixerBlock, MobileCLIPTextTransformer, and TextStudentEncoder."""

    _MCT_CFG = {
        "dim": 512,
        "model_name": "mct",
        "n_transformer_layers": 4,
        "n_heads_per_layer": 8,
        "ffn_multiplier_per_layer": 4.0,
        "norm_layer": "layer_norm_fp32",
        "context_length": 16,
        "vocab_size": 49408,
        "causal_masking": False,
    }

    def test_repmixer_blocks_are_fused_after_reparameterize(self):
        """RepMixerBlock.token_mixer must have reparam_conv after reparameterize()."""
        from ultralytics.models.sam.sam3.mobile_clip import MobileCLIPTextTransformer, RepMixerBlock

        enc = MobileCLIPTextTransformer(cfg=self._MCT_CFG, projection_dim=512)
        enc.eval()
        enc.reparameterize()

        repmixer_layers = [ly for ly in enc.transformer if isinstance(ly, RepMixerBlock)]
        assert len(repmixer_layers) == 2, "MCT should have 2 RepMixerBlock bookend layers"
        for layer in repmixer_layers:
            assert hasattr(
                layer.token_mixer, "reparam_conv"
            ), "RepMixer.token_mixer should have reparam_conv after reparameterize()"

    def test_reparameterize_preserves_output(self):
        """Encoder output must be identical before and after reparameterize() (within fp32 tolerance)."""
        from ultralytics.models.sam.sam3.text_encoder_student import TextStudentEncoder

        enc = TextStudentEncoder(cfg=self._MCT_CFG, context_length=16, output_dim=256)
        enc.eval()

        tokens = torch.randint(1, 49408, (1, 16))
        with torch.no_grad():
            out_before = enc.encoder(tokens, return_all_tokens=False)

        enc.reparameterize()

        with torch.no_grad():
            out_after = enc.encoder(tokens, return_all_tokens=False)

        assert torch.allclose(
            out_before, out_after, atol=1e-4
        ), f"reparameterize() changed model output; max diff={( out_before - out_after).abs().max().item():.2e}"

    def test_student_encoder_reparameterize_no_error(self):
        """TextStudentEncoder.reparameterize() must not raise on any variant."""
        from ultralytics.models.sam.sam3.text_encoder_student import TextStudentEncoder

        enc = TextStudentEncoder(cfg=self._MCT_CFG, context_length=16, output_dim=256)
        enc.eval()
        enc.reparameterize()  # should not raise

    def test_base_encoder_reparameterize_is_noop(self):
        """reparameterize() on a base (S1/L) encoder should be a silent no-op."""
        from ultralytics.models.sam.sam3.text_encoder_student import TextStudentEncoder

        s1_cfg = {**self._MCT_CFG, "model_name": "base", "n_transformer_layers": 12}
        enc = TextStudentEncoder(cfg=s1_cfg, context_length=16, output_dim=256)
        enc.eval()
        enc.reparameterize()  # no RepMixer blocks → no-op


# ---------------------------------------------------------------------------
# 5. _peek_litetext_pos_embed_length helper
# ---------------------------------------------------------------------------


class TestPeekPosEmbedLength:
    """Tests for _peek_litetext_pos_embed_length() checkpoint inspector."""

    def test_returns_none_for_missing_file(self):
        """Verify _peek returns None for a non-existent checkpoint file."""
        from ultralytics.models.sam.build_sam3 import _peek_litetext_pos_embed_length

        assert _peek_litetext_pos_embed_length("/nonexistent/path/fake.pt") is None

    def test_returns_correct_length(self, tmp_path):
        """Verify the correct pos-embed length is read from a saved checkpoint."""
        from ultralytics.models.sam.build_sam3 import _peek_litetext_pos_embed_length

        ckpt_path = tmp_path / "fake_s0_ctx32.pt"
        state = {
            "model.backbone.language_backbone.encoder.positional_embedding.pos_embed.pos_embed": torch.zeros(
                1, 1, 32, 512
            )
        }
        torch.save({"model": state}, str(ckpt_path))
        assert _peek_litetext_pos_embed_length(str(ckpt_path)) == 32

    def test_returns_correct_length_ctx16(self, tmp_path):
        """Verify pos-embed length 16 is read from a ctx16 checkpoint."""
        from ultralytics.models.sam.build_sam3 import _peek_litetext_pos_embed_length

        ckpt_path = tmp_path / "fake_s0_ctx16.pt"
        state = {
            "model.backbone.language_backbone.encoder.positional_embedding.pos_embed.pos_embed": torch.zeros(
                1, 1, 16, 512
            )
        }
        torch.save({"model": state}, str(ckpt_path))
        assert _peek_litetext_pos_embed_length(str(ckpt_path)) == 16

    def test_returns_none_when_key_absent(self, tmp_path):
        """Verify _peek returns None when the expected key is absent."""
        from ultralytics.models.sam.build_sam3 import _peek_litetext_pos_embed_length

        ckpt_path = tmp_path / "no_litetext_key.pt"
        torch.save({"model": {"some.other.key": torch.zeros(1, 1, 16, 512)}}, str(ckpt_path))
        assert _peek_litetext_pos_embed_length(str(ckpt_path)) is None


# ---------------------------------------------------------------------------
# 6. build_sam3_image_model - architecture check (no weights loaded)
# ---------------------------------------------------------------------------


class TestBuildSam3ImageModel:
    """Test that build_sam3_image_model wires the correct encoder type."""

    # We patch _load_checkpoint so the test runs without a real .pt file.
    @pytest.fixture(autouse=True)
    def patch_load_checkpoint(self, monkeypatch):
        """Patch _load_checkpoint so tests run without real checkpoint files."""
        import ultralytics.models.sam.build_sam3 as m

        monkeypatch.setattr(m, "_load_checkpoint", lambda model, path: model)

    @pytest.mark.parametrize(
        "fake_name",
        [
            "sam3-litetext-s0.pt",
            "sam3-litetext-s1.pt",
            "sam3-litetext-l.pt",
        ],
    )
    def test_litetext_encoder_is_used(self, fake_name):
        """Verify LiteText filenames route to TextStudentEncoder."""
        from ultralytics.models.sam.build_sam3 import build_sam3_image_model
        from ultralytics.models.sam.sam3.text_encoder_student import TextStudentEncoder

        model = build_sam3_image_model(fake_name)
        assert isinstance(
            model.backbone.language_backbone, TextStudentEncoder
        ), f"Expected TextStudentEncoder for {fake_name}"

    def test_standard_sam3_uses_ve_encoder(self):
        """Verify standard sam3.pt routes to VETextEncoder."""
        from ultralytics.models.sam.build_sam3 import build_sam3_image_model
        from ultralytics.models.sam.sam3.text_encoder_ve import VETextEncoder

        model = build_sam3_image_model("sam3.pt")
        assert isinstance(model.backbone.language_backbone, VETextEncoder)

    @pytest.mark.parametrize(
        "fake_name,expected_ctx",
        [
            ("efficient_sam3_text_s0_ctx16_fixed.pt", 16),
            ("efficient_sam3_text_s0_ctx32_fixed.pt", 32),
        ],
    )
    def test_context_length_auto_detected(self, fake_name, expected_ctx):
        """Verify context length is auto-detected from the checkpoint name."""
        from ultralytics.models.sam.build_sam3 import build_sam3_image_model

        model = build_sam3_image_model(fake_name)
        enc = model.backbone.language_backbone
        assert enc.context_length == expected_ctx


# ---------------------------------------------------------------------------
# 7. Full predictor inference (requires local checkpoint)
# ---------------------------------------------------------------------------


@requires_ckpt
class TestSAM3LiteTextPredictor:
    """End-to-end predictor test using a local SAM3-LiteText checkpoint."""

    @pytest.fixture(scope="class")
    def predictor(self):
        """Create SAM3SemanticPredictor with the local checkpoint for end-to-end testing."""
        from ultralytics.models.sam.predict import SAM3SemanticPredictor

        overrides = dict(
            conf=0.1,
            task="segment",
            mode="predict",
            model=_DEFAULT_CKPT,
            half=False,
            save=False,
            verbose=False,
        )
        return SAM3SemanticPredictor(overrides=overrides)

    def test_encoder_type_after_load(self, predictor):
        """Verify encoder type is TextStudentEncoder after loading a LiteText checkpoint."""
        from ultralytics.models.sam.sam3.text_encoder_student import TextStudentEncoder

        # Run a predict call to trigger model loading
        predictor(source=_TEST_IMAGE, text=["dog"])
        assert isinstance(predictor.model.backbone.language_backbone, TextStudentEncoder)

    def test_predict_dog(self, predictor):
        """Verify predictor returns at least one mask for the 'dog' text prompt."""
        results = predictor(source=_TEST_IMAGE, text=["dog"])
        assert len(results) == 1
        assert results[0].masks is not None, "Expected at least one mask for 'dog' prompt"
        assert len(results[0].masks) >= 1
