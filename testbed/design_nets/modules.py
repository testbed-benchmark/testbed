import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, dims, 
                 activation=nn.ReLU, 
                 norm=None,
                 dropout=0.0,
                 activate_last=False):
        """
        Implements a basic MLP. 

        dims: list of layer dimensions, e.g., [3, 256, 256, 16]
        activation: activation class (nn.ReLU, nn.GELU, ...)
        norm: normalization class (nn.LayerNorm, nn.BatchNorm1d) or None
        dropout: dropout probability
        activate_last: whether to apply activation after the final layer
        """
        super().__init__()

        layers = []
        n_layers = len(dims) - 1

        for i in range(n_layers):
            in_dim, out_dim = dims[i], dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))

            is_last = i == n_layers - 1
            if not is_last or activate_last:

                if norm is not None:
                    layers.append(norm(out_dim))

                if activation is not None:
                    layers.append(activation())

                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    """
    Implements a standard transformer-style encoder block with pre-norms.
    x = x + MHA(LN(x))
    x = x + MLP(LN(x))

    Heuristics:
    - dim / heads = 32
    """

    def __init__(self, dim, num_heads, batch_first=True, expand=4, dropout=0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=batch_first, dropout=dropout)
        self.mlp = MLP([dim, expand*dim, dim], activation=nn.GELU)

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
        # x: [b, T, dim]
        # out: [b, T, dim]

        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + self.dropout(attn_out)

        h = self.ln2(x)
        x = x + self.dropout(self.mlp(h))

        return x


class AttentionPool(nn.Module):
    """
    Implements a learnable, attention-based pooling mechanism.

    """
    def __init__(self, dim, num_heads, query_init_scale = 0.02):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * query_init_scale)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [b, T, dim]
        q = self.query.expand(x.size(0), -1, -1)  # [b, 1, dim]
        out, _ = self.attn(q, self.ln(x), self.ln(x))
        return out.squeeze(1)  # [b, dim]