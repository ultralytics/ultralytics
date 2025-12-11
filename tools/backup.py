class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """
        Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """Fuse the batch normalization layer in the BNContrastiveHead module."""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    def forward_fuse(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Passes input out unchanged."""
        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias

    def forward_open_end(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Forward function with open-end text prompt refinement.

        Args:
            x (torch.Tensor): Image features with shape (B, C, H, W).
            w (torch.Tensor): Text prompt features with shape (B, K, C).

        Returns:
            (torch.Tensor): Similarity scores with shape (B, K, H, W).
        """
        # Normalize spatial features
        x = self.norm(x)  # (B, C, H, W)
        B, C, H, W = x.shape
        
        # Step 1: Compute similarity between image features and open-end text embeddings
        # x: (B, C, H, W), open_end_te: (N, C) -> similarity: (B, N, H, W)
        open_end_te_norm = F.normalize(self.open_end_te, dim=-1, p=2)  # (N, C)
        open_end_sim = torch.einsum("bchw,nc->bnhw", x, open_end_te_norm)  # (B, N, H, W)
        open_end_sim = open_end_sim * self.logit_scale.exp() + self.bias  # (B, N, H, W)
        
        # Step 2: Apply softmax over open-end tags dimension to get attention weights
        tau = 10.0
        open_end_weights = F.softmax(open_end_sim * tau, dim=1)  # (B, N, H, W), softmax over N
        
        # Step 3: Weighted combination of open-end tag embeddings for each spatial location
        # Reshape for matrix multiplication: (B, N, H*W) @ (N, C) -> (B, C, H*W)
        open_end_weights_flat = open_end_weights.view(B, self.open_end_te.shape[0], H * W)  # (B, N, H*W)
        refined_features = torch.einsum("bnh,nc->bch", open_end_weights_flat, self.open_end_te)  # (B, C, H*W)
        refined_features = refined_features.view(B, C, H, W)  # (B, C, H, W)
        
        # Step 4: Compute final similarity scores with target text prompts
        # refined_features: (B, C, H, W), w: (B, K, C) -> score: (B, K, H, W)
        w_norm = F.normalize(w, dim=-1, p=2)  # (B, K, C)
        score = torch.einsum("bchw,bkc->bkhw", refined_features, w_norm)  # (B, K, H, W)
        # score = score * self.logit_scale.exp() + self.bias  # (B, K, H, W)
        
        return score

    def fuse_open_end_tp(self, open_end_te: torch.Tensor) -> None:
        """
        Fuse open-end text embeddings for feature refinement.

        Args:
            open_end_te (torch.Tensor): Open-end text embeddings with shape (N, C),
                where N is the number of open-end tags and C is the embedding dimension.
        """
        assert len(open_end_te.shape) == 2, f"Expected 2D tensor for open_end_te, got shape {open_end_te.shape}"
        assert open_end_te.shape[1] == self.norm.num_features, (
            f"Embedding dimension mismatch: expected {self.norm.num_features}, got {open_end_te.shape[1]}"
        )
        
        # Normalize and register the open-end text embeddings
        self.register_buffer("open_end_te", F.normalize(open_end_te, dim=-1, p=2))
        
        # Switch to open-end forward mode
        self.forward = self.forward_open_end
