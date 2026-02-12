import math
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F


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

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(64, 64)).view(1, 1, 64, 64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        attn = attn.masked_fill(
            self.bias[:, :, :sequence_length, :sequence_length] == 0, float("-inf")
        )
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
        self.input = nn.Linear(d_model, 4 * d_model)
        self.output = nn.Linear(4 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = F.gelu(x)
        x = self.output(x)
        return x


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attention = AttentionHead(d_model, n_heads)
        self.feed_forward = FeedForward(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x) + x
        x = self.feed_forward(x) + x
        return x


class GPT(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.token_embedding = nn.Embedding(50257, d_model)
        self.position_embedding = nn.Embedding(64, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, d_model)

        # TODO: understand this
        self.lm_head = nn.Linear(d_model, 50257, bias=False)
        self.lm_head.LLMC_SKIP_INIT = 1  # don't init this one, we will tie weights
        self.token_embedding.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        token_embeds = self.token_embedding(idx)
        position_embeds = self.position_embedding(torch.arange(idx.size(1)))
        x = token_embeds + position_embeds

        for block in self.blocks:
            x = block(x)
        x = self.layer_norm(x)
        x = self.output(x)

        loss = None
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return x, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, zero_stage):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        print("using regular AdamW")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= 64 else idx[:, -64:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
