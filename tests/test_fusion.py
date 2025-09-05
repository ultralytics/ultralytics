# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Test fusion utilities for text and visual prompt embeddings.
"""

import torch

from ultralytics.utils.fusion import (
    PromptEmbeddingFusion,
    CrossAttentionFusion, 
    fuse_prompt_embeddings
)


class TestPromptEmbeddingFusion:
    """Test the PromptEmbeddingFusion class."""

    def test_concat_fusion(self):
        """Test concatenation fusion method."""
        print("Testing concat fusion...")
        fusion = PromptEmbeddingFusion('concat')
        
        tpe = torch.randn(2, 10, 512)
        vpe = torch.randn(2, 5, 512)
        
        result = fusion(tpe, vpe)
        assert result.shape == (2, 15, 512)
        
        # Test with only TPE
        result_tpe = fusion(tpe, None)
        assert torch.equal(result_tpe, tpe)
        
        # Test with only VPE
        result_vpe = fusion(None, vpe)
        assert torch.equal(result_vpe, vpe)
        print("✓ Concat fusion tests passed")

    def test_sum_fusion(self):
        """Test sum fusion method."""
        print("Testing sum fusion...")
        fusion = PromptEmbeddingFusion('sum')
        
        tpe = torch.randn(2, 10, 512)
        vpe = torch.randn(2, 10, 512)  # Same sequence length
        
        result = fusion(tpe, vpe)
        assert result.shape == (2, 10, 512)
        assert torch.allclose(result, tpe + vpe)
        
        # Test error case with different sequence lengths
        vpe_diff = torch.randn(2, 5, 512)
        try:
            fusion(tpe, vpe_diff)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "same sequence length" in str(e)
        print("✓ Sum fusion tests passed")

    def test_attention_fusion(self):
        """Test attention fusion method."""
        print("Testing attention fusion...")
        fusion = PromptEmbeddingFusion('attention', embed_dim=512)
        
        tpe = torch.randn(2, 10, 512)
        vpe = torch.randn(2, 5, 512)
        
        result = fusion(tpe, vpe)
        assert result.shape == (2, 10, 512)  # Same as TPE (queries)
        
        # Should be different from input due to attention
        assert not torch.allclose(result, tpe)
        print("✓ Attention fusion tests passed")

    def test_invalid_method(self):
        """Test error handling for invalid fusion method."""
        print("Testing invalid method...")
        try:
            PromptEmbeddingFusion('invalid')
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unsupported fusion method" in str(e)
        print("✓ Invalid method test passed")

    def test_no_embeddings(self):
        """Test error handling when no embeddings provided."""
        print("Testing no embeddings...")
        fusion = PromptEmbeddingFusion('concat')
        try:
            fusion(None, None)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "At least one of TPE or VPE" in str(e)
        print("✓ No embeddings test passed")

    def test_shape_validation(self):
        """Test input shape validation."""
        print("Testing shape validation...")
        fusion = PromptEmbeddingFusion('concat')
        
        # Test wrong number of dimensions
        tpe_2d = torch.randn(10, 512)  # Missing batch dimension
        vpe = torch.randn(2, 5, 512)
        try:
            fusion(tpe_2d, vpe)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "3D tensors" in str(e)
        
        # Test mismatched batch sizes
        tpe = torch.randn(2, 10, 512)
        vpe_diff_batch = torch.randn(3, 5, 512)
        try:
            fusion(tpe, vpe_diff_batch)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "same batch size" in str(e)
        
        # Test mismatched embedding dimensions
        tpe = torch.randn(2, 10, 512)
        vpe_diff_embed = torch.randn(2, 5, 256)
        try:
            fusion(tpe, vpe_diff_embed)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "same embedding dimension" in str(e)
        print("✓ Shape validation tests passed")


class TestCrossAttentionFusion:
    """Test the CrossAttentionFusion module."""

    def test_forward(self):
        """Test forward pass."""
        print("Testing cross attention forward...")
        attention_fusion = CrossAttentionFusion(embed_dim=512, num_heads=8)
        
        tpe = torch.randn(2, 10, 512)
        vpe = torch.randn(2, 5, 512)
        
        result = attention_fusion(tpe, vpe)
        assert result.shape == (2, 10, 512)
        
        # Should be different from input due to attention
        assert not torch.allclose(result, tpe)
        print("✓ Cross attention tests passed")

    def test_invalid_embed_dim(self):
        """Test error handling for invalid embedding dimension."""
        print("Testing invalid embed dim...")
        try:
            CrossAttentionFusion(embed_dim=511, num_heads=8)  # Not divisible by num_heads
            assert False, "Should have raised AssertionError"
        except AssertionError:
            pass
        print("✓ Invalid embed dim test passed")


def test_fuse_prompt_embeddings_function():
    """Test the convenience function."""
    print("Testing convenience function...")
    tpe = torch.randn(2, 10, 512)
    vpe = torch.randn(2, 5, 512)
    
    # Test concat method
    result_concat = fuse_prompt_embeddings(tpe, vpe, method='concat')
    assert result_concat.shape == (2, 15, 512)
    
    # Test sum method (need same sequence length)
    vpe_same_len = torch.randn(2, 10, 512)
    result_sum = fuse_prompt_embeddings(tpe, vpe_same_len, method='sum')
    assert result_sum.shape == (2, 10, 512)
    assert torch.allclose(result_sum, tpe + vpe_same_len)
    
    # Test attention method
    result_attn = fuse_prompt_embeddings(tpe, vpe, method='attention', embed_dim=512)
    assert result_attn.shape == (2, 10, 512)
    print("✓ Convenience function tests passed")


def main():
    """Run all tests."""
    print("Running prompt embedding fusion tests...\n")
    
    test_suite = TestPromptEmbeddingFusion()
    test_suite.test_concat_fusion()
    test_suite.test_sum_fusion()
    test_suite.test_attention_fusion()
    test_suite.test_invalid_method()
    test_suite.test_no_embeddings()
    test_suite.test_shape_validation()
    
    test_cross_attn = TestCrossAttentionFusion()
    test_cross_attn.test_forward()
    test_cross_attn.test_invalid_embed_dim()
    
    test_fuse_prompt_embeddings_function()
    
    print("\n🎉 All tests passed!")


if __name__ == "__main__":
    main()