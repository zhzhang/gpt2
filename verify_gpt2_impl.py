from transformers import GPT2LMHeadModel
from model import Model, ModelConfig, PositionEmbeddingType

print("mine: loading weights from pretrained gpt2 base")

model = Model(
    ModelConfig(
        n_layers=12,
        n_q_heads=12,
        n_kv_heads=3,
        d_model=768,
        max_sequence_length=1024,
        vocab_size=50257,
        position_embedding_type=PositionEmbeddingType.LEARNED,
    )
)
sd = model.state_dict()
sd_keys = sd.keys()
sd_keys = [
    k for k in sd_keys if not k.endswith(".attention.mask")
]  # discard this mask / buffer, not a param

# init a huggingface/transformers model
pretrained_model = GPT2LMHeadModel.from_pretrained("gpt2")
pretrained_sd = pretrained_model.state_dict()

# copy while ensuring all of the parameters are aligned and match in names and shapes
pretrained_sd_keys = pretrained_sd.keys()
# ignore mask buffers
pretrained_sd_keys = [
    k for k in pretrained_sd_keys if not k.endswith(".attn.masked_bias")
]
pretrained_sd_keys = [k for k in pretrained_sd_keys if not k.endswith(".attn.bias")]
transposed = [
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
]
parameter_name_mapping = {}
for k in sd_keys:
    target_key = None
    first, rest = k.split(".", 1)
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
        elif rest == "feed_forward.input.weight":
            target_key = f"transformer.h.{idx}.mlp.c_fc.weight"
        elif rest == "feed_forward.input.bias":
            target_key = f"transformer.h.{idx}.mlp.c_fc.bias"
        elif rest == "feed_forward.output.weight":
            target_key = f"transformer.h.{idx}.mlp.c_proj.weight"
        elif rest == "feed_forward.output.bias":
            target_key = f"transformer.h.{idx}.mlp.c_proj.bias"
    elif first == "layer_norm":
        if rest == "weight":
            target_key = "transformer.ln_f.weight"
        elif rest == "bias":
            target_key = "transformer.ln_f.bias"
    elif first == "lm_head":
        target_key = "lm_head.weight"
    if target_key is not None:
        parameter_name_mapping[k] = target_key
# basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
# this means that we have to transpose these weights when we import them
assert len(pretrained_sd_keys) == len(sd_keys) and len(pretrained_sd_keys) == len(
    parameter_name_mapping
), f"mismatched keys: {len(pretrained_sd_keys)} != {len(sd_keys)}"
transposed = [
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
]
for k in sd_keys:
    target_key = parameter_name_mapping[k]
    if any(target_key.endswith(w) for w in transposed):
        # special treatment for the Conv1D weights we need to transpose
        assert pretrained_sd[target_key].shape[::-1] == sd[k].shape
        with torch.no_grad():
            sd[k].copy_(pretrained_sd[target_key].t())
    else:
        # vanilla copy over the other parameters
        assert pretrained_sd[target_key].shape == sd[k].shape
        with torch.no_grad():
            sd[k].copy_(pretrained_sd[target_key])
