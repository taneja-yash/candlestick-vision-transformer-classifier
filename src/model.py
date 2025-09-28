import torch
from torch import nn
from colorama import Fore
from einops.layers.torch import Rearrange
from torchinfo import summary
from matplotlib import pyplot as plt


class MLPBlock(nn.Sequential):
    def __init__(self, embed_dim, mlp_dim):
        super(MLPBlock, self).__init__(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(0.25),
        )


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, batch_first=True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=batch_first, dropout=0.25
        )
        self.dropout = nn.Dropout(0.25)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim, mlp_dim)

    def forward(self, x):
        z1 = self.ln_1(x)
        z1 = self.self_attention(z1, z1, z1, need_weights=False)[0] + x
        z1 = self.dropout(z1)
        z2 = self.ln_2(z1)
        z2 = self.mlp(z2) + z1
        return z2


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, num_layers):
        super().__init__()
        self.layers = nn.Sequential()
        for x in range(num_layers):
            self.layers.add_module(
                f"encoder_layer_{x}", EncoderBlock(embed_dim, num_heads, mlp_dim)
            )

    def forward(self, x):
        return self.layers(x)


def pos_encoding(seq_length, dim_size):
    p = torch.zeros((seq_length, dim_size))
    for k in range(seq_length):
        for i in range(int(dim_size / 2)):
            p[k, 2 * i] = torch.sin(torch.tensor(k / (10000 ** (2 * i / dim_size))))
            p[k, 2 * i + 1] = torch.cos(torch.tensor(k / (10000 ** (2 * i / dim_size))))
    return p


class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = Rearrange(
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=8, p2=8
        )
        self.class_token = nn.Parameter(torch.randn(1, 1, 1024))
        self.embedding = nn.Linear(192, 1024)
        self.register_buffer("positional_embedding", pos_encoding(118, 1024))
        self.emb_dropout = nn.Dropout(0.25)
        self.norm1 = nn.LayerNorm(192)
        self.norm2 = nn.LayerNorm(1024)
        self.encoder = Encoder(1024, 8, 2056, 3)
        self.mlp1 = nn.Linear(1024, 5)

    def forward(self, x):
        x = self.patch(x)
        x = self.norm1(x)
        x = self.embedding(x)
        x = self.norm2(x)

        batch_size = x.shape[0]
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        patch_embedding = torch.cat([class_tokens, x], dim=1)

        final_embedding = patch_embedding + self.positional_embedding
        final_embedding_dropout = self.emb_dropout(final_embedding)

        encoded_embedding = self.encoder(final_embedding_dropout)

        attended_class_token = encoded_embedding[:, 0, :]
        z1 = self.mlp1(attended_class_token)
        return z1


if __name__ == "__main__":
    stack_components = [
        "encoder",
        "encoder_block",
        "mlp_block",
        "pos_embedding",
        "vit",
    ]
    component = stack_components[0]

    if component == "encoder":
        encoder = Encoder(1024, 8, 2056, 3)
        summary(encoder, (1, 118, 1024))
    elif component == "encoder_block":
        encoder_block = EncoderBlock(1024, 8, 2056)
        summary(encoder_block, (1, 118, 1024))
    elif component == "mlp_block":
        mlp_head = MLPBlock(768, 1024)
        summary(mlp_head, (1, 118, 768))
    elif component == "pos_embedding":
        pos = pos_encoding(118, 1024)
        cax = plt.matshow(pos)
        plt.gcf().colorbar(cax)
        plt.show()
    elif component == "vit":
        model = ViT()
        summary(model, (1, 3, 104, 72))
