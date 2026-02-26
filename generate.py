import time
import copy

import tiktoken
import torch
from torch.nn import functional as F
from typing import Optional

from model import GPT


def _configure_cache(model: GPT, cached: bool) -> None:
    for block in model.blocks:
        block.cached = cached
        block.attention.cache = None


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    cached: bool = True,
    validate_cache_impl: bool = False,
) -> torch.Tensor:
    _configure_cache(model, cached=cached)
    assert not validate_cache_impl or cached, (
        "validate_cache_impl requires cached=True."
    )

    validation_model = None
    if validate_cache_impl:
        validation_model = copy.deepcopy(model)
        validation_model.eval()
        _configure_cache(validation_model, cached=False)

    first_token = True
    prefill_time = None
    generation_time = 0
    for _ in range(max_new_tokens):
        start_time = time.time()
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = (
            idx
            if idx.size(1) <= model.context_length
            else idx[:, -model.context_length :]
        )
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        if validation_model is not None:
            uncached_logits, _ = validation_model(idx_cond)
            torch.testing.assert_close(
                logits,
                uncached_logits,
                rtol=1e-5,
                atol=1e-5,
                msg="Cached and uncached logits diverged.",
            )
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
        elapsed_time = time.time() - start_time
        if first_token:
            prefill_time = elapsed_time
            first_token = False
        else:
            generation_time += elapsed_time

    print(f"Prefill time: {1000 * prefill_time} ms")
    print(f"Generation time: {1000 * generation_time} ms")
    print(f"Time per token: {1000 * generation_time / (max_new_tokens - 1)} ms ")

    return idx


if __name__ == "__main__":
    model = GPT.from_pretrained()
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    input_text = "Once upon a time, there was a boy who"
    tokens = torch.tensor(enc.encode(input_text)).unsqueeze(0)
    output = generate(model, tokens, max_new_tokens=256)
    print(enc.decode(output[0].tolist()))
