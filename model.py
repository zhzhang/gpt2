import math

import torch
from torch import nn


class AttentionHead(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        # Reason for the above assert is detailed in the forward pass.
        self.n_heads = n_heads
        self.d_model = d_model
        # TODO: see what happens if we don't project in multi head attention.

        # Because in self attention Q, K, V are all just equal to the input x, we can project them all at once.
        self.input_projection = nn.Linear(d_model, 3 * d_model)

        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # X shape is (batch_size, sequence_length, d_model)
        batch_size, sequence_length, d_model = x.size()
        # (batch_size, sequence_length, 3 * d_model)
        x = self.input_projection(x)
        # Each of q, k, v is (batch_size, sequence_length, n_dim)
        q, k, v = x.split(d_model, dim=2)
        # Allocate slices of q, k, v for each head.
        # A linear map from d_input -> d_head is the same as a linear map from
        # d_input -> d_input because that linear map is just a stack of matrices
        # of size d_input x (d_input / n_heads) so long as there is divisibility.
        # Each of q, k, v is now (batch_size, sequence_length, n_heads, d_head)
        # where d_head = d_model / n_heads
        q = q.view(batch_size, sequence_length, self.n_heads, d_model // self.n_heads)
        k = k.view(batch_size, sequence_length, self.n_heads, d_model // self.n_heads)
        v = v.view(batch_size, sequence_length, self.n_heads, d_model // self.n_heads)
        # Swap the sequence length and head dimensions, as we are trying to end up with a sequence_length x sequence_length matrix.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # (batch_size, n_heads, sequence_length, sequence_length)
        attn = q @ k.transpose(-2, -1)
        # Scale the attention scores by 1 / sqrt(d_model)
        # TODO: what happens if we put the scale after the mask?
        attn = attn / math.sqrt(d_model)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        # TODO: see if this softmax is in the right dimension.
        attn = attn.softmax(dim=-2)
        # (batch_size, n_heads, sequence_length, d_head)
        attn = attn @ v
        # Swap back the sequence length and head dimensions.
        output = attn.transpose(1, 2)
        output = output.reshape(batch_size, sequence_length, d_model)
        output = self.output_projection(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.input = nn.Linear(d_model, 4 * d_model)
        self.output = nn.Linear(4 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(torch.relu(self.input(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attention = AttentionHead(d_model, n_heads)
        self.feed_forward = FeedForward(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x += self.attention(x, mask)
        x += self.feed_forward(x)
        return x


class GPT(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask)
        return self.output(x)
