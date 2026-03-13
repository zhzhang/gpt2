import argparse
import torch

from model import Model, ModelConfig, PositionEmbeddingType
from train_gpt2 import GPT


def load_mapped_model_from_train_gpt2() -> tuple[Model, GPT]:
    """Load smallest GPT-2 from train_gpt2.py and map weights into model.py Model."""
    reference_model = GPT.from_pretrained("gpt2")
    reference_model.eval()

    model = Model(
        ModelConfig(
            n_layers=12,
            n_q_heads=12,
            n_kv_heads=12,  # Must match GPT-2 MHA for exact parity.
            d_model=768,
            max_sequence_length=1024,
            vocab_size=50257,
            position_embedding_type=PositionEmbeddingType.LEARNED,
        )
    )
    model.eval()

    target_sd = model.state_dict()
    source_sd = reference_model.state_dict()

    target_keys = [k for k in target_sd.keys() if not k.endswith(".attention.mask")]
    source_keys = [k for k in source_sd.keys() if not k.endswith(".attn.masked_bias")]
    source_keys = [k for k in source_keys if not k.endswith(".attn.bias")]

    parameter_name_mapping = {}
    for key in target_keys:
        target_key = None
        first, rest = key.split(".", 1)
        if first == "token_embedding":
            target_key = "transformer.wte.weight"
        elif first == "position_embedding":
            target_key = "transformer.wpe.weight"
        elif first == "blocks":
            idx, rest = rest.split(".", 1)
            if rest == "layer_norm_1.weight":
                target_key = f"transformer.h.{idx}.ln_1.weight"
            elif rest == "layer_norm_1.bias":
                target_key = f"transformer.h.{idx}.ln_1.bias"
            elif rest == "layer_norm_2.weight":
                target_key = f"transformer.h.{idx}.ln_2.weight"
            elif rest == "layer_norm_2.bias":
                target_key = f"transformer.h.{idx}.ln_2.bias"
            elif rest == "attention.input_projection.weight":
                target_key = f"transformer.h.{idx}.attn.c_attn.weight"
            elif rest == "attention.input_projection.bias":
                target_key = f"transformer.h.{idx}.attn.c_attn.bias"
            elif rest == "attention.output_projection.weight":
                target_key = f"transformer.h.{idx}.attn.c_proj.weight"
            elif rest == "attention.output_projection.bias":
                target_key = f"transformer.h.{idx}.attn.c_proj.bias"
            elif rest == "feed_forward.c_fc.weight":
                target_key = f"transformer.h.{idx}.mlp.c_fc.weight"
            elif rest == "feed_forward.c_fc.bias":
                target_key = f"transformer.h.{idx}.mlp.c_fc.bias"
            elif rest == "feed_forward.c_proj.weight":
                target_key = f"transformer.h.{idx}.mlp.c_proj.weight"
            elif rest == "feed_forward.c_proj.bias":
                target_key = f"transformer.h.{idx}.mlp.c_proj.bias"
        elif first == "layer_norm":
            if rest == "weight":
                target_key = "transformer.ln_f.weight"
            elif rest == "bias":
                target_key = "transformer.ln_f.bias"
        elif first == "lm_head":
            target_key = "lm_head.weight"
        if target_key is not None:
            parameter_name_mapping[key] = target_key

    assert len(parameter_name_mapping) == len(target_keys), (
        f"incomplete mapping: {len(parameter_name_mapping)} vs {len(target_keys)}"
    )
    assert len(source_keys) == len(target_keys), (
        f"mismatched keys: {len(source_keys)} != {len(target_keys)}"
    )

    with torch.no_grad():
        for key in target_keys:
            source_key = parameter_name_mapping[key]
            assert source_sd[source_key].shape == target_sd[key].shape
            target_sd[key].copy_(source_sd[source_key])

    return model, reference_model


@torch.no_grad()
def compare_logits(device: str, seq_len: int, seed: int) -> None:
    model, reference_model = load_mapped_model_from_train_gpt2()
    model = model.to(device)
    reference_model = reference_model.to(device)

    g = torch.Generator(device=device if "cuda" in device else "cpu")
    g.manual_seed(seed)
    idx = torch.randint(low=0, high=50257, size=(1, seq_len), generator=g, device=device)
    targets = idx.clone()

    ref_logits, _ = reference_model(idx, targets=targets, return_logits=True)
    logits, _ = model(idx, targets=targets)

    diff = (logits - ref_logits).abs()
    print(f"device={device}")
    print(f"sequence_length={seq_len}")
    print(f"max_abs_diff={diff.max().item():.8f}")
    print(f"mean_abs_diff={diff.mean().item():.8f}")
    print(f"allclose(atol=1e-5, rtol=1e-5)={torch.allclose(logits, ref_logits, atol=1e-5, rtol=1e-5)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sequence_length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    compare_logits(device=args.device, seq_len=args.sequence_length, seed=args.seed)


if __name__ == "__main__":
    main()
