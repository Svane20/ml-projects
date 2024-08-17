import torch
import torch.nn as nn


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) module.

    Args:
        img_size (int): Size of the input image (assumed square).
        in_channels (int): Number of input channels (default is 3 for RGB).
        patch_size (int): Size of patches to extract from input images.
        num_transformer_layers (int): Number of transformer layers in the encoder.
        embedding_dim (int): Dimension of the token embeddings.
        mlp_size (int): Dimension of the feedforward network.
        num_heads (int): Number of attention heads.
        attn_dropout (float): Dropout rate for the attention mechanism.
        embedding_dropout (float): Dropout rate after patch and positional embeddings.
        num_classes (int): Number of output classes.
    """

    def __init__(self,
                 img_size: int = 224,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 num_transformer_layers: int = 12,
                 embedding_dim: int = 768,
                 mlp_size: int = 3072,
                 num_heads: int = 12,
                 attn_dropout: float = 0,
                 embedding_dropout: float = 0.1,
                 num_classes: int = 1000) -> None:
        super().__init__()

        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."

        num_patches = (img_size * img_size) // patch_size ** 2

        # Create patch embeddings
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)

        # Add positional embeddings to the patches
        self.positional_embedding = PositionalEmbedding(num_patches=num_patches,
                                                        embedding_dim=embedding_dim)

        # Transformer Encoder
        self.transformer_encoder = TransformerEncoder(embedding_dim=embedding_dim,
                                                      num_heads=num_heads,
                                                      mlp_size=mlp_size,
                                                      num_layers=num_transformer_layers,
                                                      attn_dropout=attn_dropout)

        # Layer normalization before the classification head
        self.norm = nn.LayerNorm(embedding_dim)

        # Classifier head
        self.classification_head = nn.Linear(embedding_dim, num_classes)

        # Dropout layer after embedding
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Generate patch embeddings
        x = self.patch_embedding(x)

        # Add positional embeddings
        x = self.positional_embedding(x)

        # Apply dropout after embeddings
        x = self.embedding_dropout(x)

        # Pass through the transformer encoder
        x = self.transformer_encoder(x)

        # Normalize the output from the class token
        x = self.norm(x[:, 0])

        # Pass the class token to the classification head
        x = self.classification_head(x)

        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder module.

    Args:
        embedding_dim (int): Dimension of the input embeddings.
        num_heads (int): Number of attention heads.
        mlp_size (int): Dimension of the feedforward network.
        num_layers (int): Number of transformer layers.
        attn_dropout (float): Dropout rate for the attention mechanism.
    """

    def __init__(self, embedding_dim: int, num_heads: int, mlp_size: int, num_layers: int, attn_dropout: float) -> None:
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim,
                                       nhead=num_heads,
                                       dim_feedforward=mlp_size,
                                       dropout=attn_dropout),
            num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class PositionalEmbedding(nn.Module):
    """Adds positional embeddings to the tokenized patches.

    Args:
        num_patches (int): Number of patches in the input image.
        embedding_dim (int): Size of the embedding vector. Default is 768.
    """

    def __init__(self, num_patches: int, embedding_dim: int = 768) -> None:
        super().__init__()

        self.class_token = nn.Parameter(torch.randn(size=(1, 1, embedding_dim)))
        self.position_embedding = nn.Parameter(torch.randn(size=(1, num_patches + 1, embedding_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand class_token to match the batch size and concatenate it to the input
        # Final shape [batch_size, N+1, embedding_dim], where N is the number of patches
        return torch.cat([self.class_token.expand(size=(x.size(0), -1, -1)), x], dim=1) + self.position_embedding


class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence of learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Default is 3.
        patch_size (int): Size of patches to convert input image into. Default is 16.
        embedding_dim (int): Size of embedding to turn image into. Default is 768.
    """

    def __init__(self, in_channels: int = 3, patch_size: int = 16, embedding_dim: int = 768) -> None:
        super().__init__()

        self.projection = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size,
                                    stride=patch_size)
        self.flatten = nn.Flatten(start_dim=2,  # Only flatten the feature map dimensions into a single vector
                                  end_dim=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]
        return self.flatten(self.projection(x)).permute(0, 2, 1)
