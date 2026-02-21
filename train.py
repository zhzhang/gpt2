import time
import glob
import math

import numpy as np
import tiktoken
import torch
from model import GPT
from hellaswag import evaluate

torch.autograd.set_detect_anomaly(True)


def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print(
            "---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README"
        )
        print(
            "---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try"
        )
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, (
            f"did not find any files that match the pattern {filename_pattern}"
        )

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print(
            f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files"
        )

        # kick things off
        self.current_shard = None
        self.reset()

    def reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds advance the shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x, y


# args error checking and convenience variables
B, T = 4, 64
assert 1 <= T <= 1024


# rng / reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

enc = tiktoken.get_encoding("gpt2")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

# init the model, either from scratch or from OpenAI pretrained checkpoint

model = GPT(d_model=768, n_heads=12, n_layers=12, context_length=1024)
model.to(device)
# model = GPT.from_pretrained()
model.train()


# load tokens
train_loader = DistributedDataLoader("fineweb10B/fineweb_train_*.bin", B, T, 0, 1)
val_loader = DistributedDataLoader("fineweb10B/fineweb_val_*.bin", B, T, 0, 1)

LEARNING_RATE = 1e-4
LEARNING_RATE_DECAY_FRAC = 1.0
WEIGHT_DECAY = 0.0
WARMUP_ITERS = 0
NUM_ITERATIONS = 20000
VAL_LOSS_EVERY = 100
VAL_MAX_STEPS = 20
EVAL_EVERY = 500
GRAD_ACCUM_STEPS = 1
GRAD_CLIP = 1.0

# init the optimizer
optimizer = model.configure_optimizers(
    weight_decay=WEIGHT_DECAY,
    learning_rate=LEARNING_RATE,
    betas=(0.9, 0.95),
    zero_stage=0,
)


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    min_lr = LEARNING_RATE * LEARNING_RATE_DECAY_FRAC
    # 1) linear warmup for warmup_iters steps
    if it < WARMUP_ITERS:
        return LEARNING_RATE * (it + 1) / WARMUP_ITERS
    # 2) if it > lr_decay_iters, return min learning rate
    if it > NUM_ITERATIONS:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - WARMUP_ITERS) / (NUM_ITERATIONS - WARMUP_ITERS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (LEARNING_RATE - min_lr)


timings = []
norm = -1.0  # dummy value to print in inference-only mode
for step in range(NUM_ITERATIONS + 1):
    t0 = time.time()
    last_step = step == NUM_ITERATIONS

    # once in a while evaluate the validation dataset
    if (step % VAL_LOSS_EVERY == 0 or last_step) and (val_loader is not None):
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss = 0.0
            for _ in range(VAL_MAX_STEPS):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                val_loss += loss.item()
            val_loss /= VAL_MAX_STEPS
        # log to console and to file
        print(f"val loss {val_loss}")

    if step % EVAL_EVERY == 0:
        acc, acc_norm = evaluate(model, device)
        print(f"acc: {acc:.4f} acc_norm: {acc_norm:.4f}")

    # bit confusing: we want to make sure to eval and sample on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # micro-batch loop where we do gradient accumulation to reach desired total batch size
    lossf = (
        0.0  # for getting the mean loss (as simple float) over the accumulation steps
    )
    for micro_step in range(1):
        # fetch a batch
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # forward pass
        _, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN, so we scale the loss here
        loss = loss / GRAD_ACCUM_STEPS
        lossf += loss.detach()  # keep track of the mean loss
        # backward pass
        loss.backward()
    lossf = lossf.item()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    # step the optimizer
    optimizer.step()
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    # time and print
    t1 = time.time()
    # the 0th iteration is often an outlier (much slower) => skip logging it
    print(
        f"step {step + 1:4d}/{NUM_ITERATIONS} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1 - t0) * 1000:.2f} ms)"
    )

    # keep track of smooth timings, last 20 iterations
    if step > 0 and step > NUM_ITERATIONS - 20:
        timings.append(t1 - t0)
