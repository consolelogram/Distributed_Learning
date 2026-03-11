"""
worker0.py — Pipeline Parallel Training, Stage 0

Owns:
  - Character embedding table       [vocab_size, d_model]
  - Positional embedding table      [seq_len,    d_model]
  - Transformer blocks 0–3

Communication (training):
  - SENDS    hidden state tensor    [batch, seq_len, d_model]  → worker1
  - RECEIVES gradient tensor        [batch, seq_len, d_model]  ← worker1

Communication (generation):
  - SENDS    {"cmd": "GENERATE", "context": [...], "seed_text": str}
  - Loop GEN_STEPS times:
      SENDS    hidden state          [1, T, d_model]            → worker1
      RECEIVES next token id (int)                              ← worker1
  - SENDS    {"cmd": "DONE"}

Run AFTER starting worker1.py (worker1 is the TCP server).
"""

import socket
import pickle
import struct
import math
import torch
import torch.nn as nn
import torch.optim as optim
import urllib.request

# ─────────────────────────────────────────────
# Hyperparameters  (must match worker1.py)
# ─────────────────────────────────────────────
D_MODEL    = 128
N_HEADS    = 4
D_FF       = D_MODEL * 4
SEQ_LEN    = 256
BATCH_SIZE = 16
LR         = 3e-4
NUM_STEPS  = 15000
SEED       = 42

# Generation settings (must match worker1.py)
GEN_STEPS = 500

# Seed prompt — feel free to change this
GEN_SEED_TEXT = "ROMEO:\nWhat light through yonder window breaks? It is"

WORKER1_HOST = "127.0.0.1"
WORKER1_PORT = 29500

# ─────────────────────────────────────────────
# 1. Dataset & Tokenizer
# ─────────────────────────────────────────────
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def load_dataset():
    try:
        with urllib.request.urlopen(DATA_URL) as r:
            text = r.read().decode("utf-8")
    except Exception:
        text = ("To be or not to be that is the question.\n" * 500)
    return text

text = load_dataset()

chars      = sorted(set(text))
vocab_size = len(chars)
stoi       = {ch: i for i, ch in enumerate(chars)}
itos       = {i: ch for ch, i in stoi.items()}

def encode(s):
    return [stoi[c] for c in s if c in stoi]

def decode(ids):
    return "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)

print(f"[worker0] Vocabulary size : {vocab_size}")
print(f"[worker0] Corpus length   : {len(data)} tokens")

# ─────────────────────────────────────────────
# 2. Batch Sampler — identical seed as worker1
# ─────────────────────────────────────────────
rng = torch.Generator()
rng.manual_seed(SEED)

def get_batch():
    ix = torch.randint(len(data) - SEQ_LEN - 1, (BATCH_SIZE,), generator=rng)
    x  = torch.stack([data[i : i + SEQ_LEN]     for i in ix])
    y  = torch.stack([data[i + 1 : i + SEQ_LEN + 1] for i in ix])
    return x, y

# ─────────────────────────────────────────────
# 3. Model Components
# ─────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model,     bias=False)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)

        def reshape(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)
        scale   = math.sqrt(self.d_head)
        scores  = torch.matmul(q, k.transpose(-2, -1)) / scale

        mask   = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out  = torch.matmul(attn, v)
        out  = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = FeedForward(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class Stage0Model(nn.Module):
    """
    Front half of the transformer.
      Input  : token ids  [B, T]
      Output : hidden     [B, T, D_MODEL]
    """

    def __init__(self, vocab_size, d_model, n_heads, d_ff, seq_len, n_blocks=4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(seq_len,    d_model)
        self.blocks    = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_blocks)]
        )

    def forward(self, idx):
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)  # [B, T, D]
        for block in self.blocks:
            x = block(x)
        return x   # [B, T, D_MODEL]


# ─────────────────────────────────────────────
# 4. Socket Helpers
# ─────────────────────────────────────────────

def send_msg(sock, payload):
    raw    = pickle.dumps(payload)
    length = struct.pack("<Q", len(raw))
    sock.sendall(length + raw)


def recv_msg(sock):
    def recv_exact(n):
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Socket closed prematurely")
            buf += chunk
        return buf

    length_bytes = recv_exact(8)
    length       = struct.unpack("<Q", length_bytes)[0]
    raw          = recv_exact(length)
    return pickle.loads(raw)


# ─────────────────────────────────────────────
# 5. Build Model & Optimizer
# ─────────────────────────────────────────────
torch.manual_seed(SEED)
model     = Stage0Model(vocab_size, D_MODEL, N_HEADS, D_FF, SEQ_LEN, n_blocks=4)
optimizer = optim.Adam(model.parameters(), lr=LR)

n_params = sum(p.numel() for p in model.parameters())
print(f"[worker0] Stage-0 parameters: {n_params:,}")

# ─────────────────────────────────────────────
# 6. Connect to Worker1 (TCP client)
# ─────────────────────────────────────────────
print(f"[worker0] Connecting to worker1 at {WORKER1_HOST}:{WORKER1_PORT} ...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((WORKER1_HOST, WORKER1_PORT))
print("[worker0] Connected.")

# ─────────────────────────────────────────────
# 7. Training Loop
# ─────────────────────────────────────────────
for step in range(1, NUM_STEPS + 1):

    model.train()
    optimizer.zero_grad()

    x, _ = get_batch()
    # x: [BATCH_SIZE, SEQ_LEN]

    # Forward through stage-0
    hidden_out = model(x)
    # hidden_out: [BATCH_SIZE, SEQ_LEN, D_MODEL]

    # Send activation to worker1
    send_msg(sock, hidden_out.detach())

    # Receive gradient ∂Loss/∂hidden_out from worker1
    grad_from_w1 = recv_msg(sock)
    # grad_from_w1: [BATCH_SIZE, SEQ_LEN, D_MODEL]

    # Inject gradient to complete the backward pass through blocks 3 → 0
    torch.autograd.backward(
        tensors      = [hidden_out],
        grad_tensors = [grad_from_w1]
    )

    optimizer.step()

    if step % 100 == 0:
        print(f"[worker0] step {step}/{NUM_STEPS}")

print("[worker0] Training complete.")

# ─────────────────────────────────────────────
# 8. Autoregressive Text Generation
# ─────────────────────────────────────────────
#
# worker0 drives the generation loop:
#   - maintains the growing token sequence (context)
#   - feeds the context through its own layers each step
#   - sends the hidden state to worker1
#   - receives the next token id back
#   - appends it and repeats

print("\n" + "─" * 60)
print("[worker0] Starting generation ...")
print("─" * 60)

model.eval()

# Encode the seed prompt into token ids
seed_ids = encode(GEN_SEED_TEXT)
if not seed_ids:
    # Fallback if seed chars aren't in vocab
    seed_ids = [0]

context = seed_ids[:]   # mutable list we'll grow one token at a time

# Tell worker1 we're entering generation mode and share the seed
send_msg(sock, {
    "cmd"      : "GENERATE",
    "context"  : context,
    "seed_text": GEN_SEED_TEXT,
})

print(f"[worker0] Seed: {repr(GEN_SEED_TEXT)}")
print(f"[worker0] Generating {GEN_STEPS} characters ...")

with torch.no_grad():
    for gen_step in range(GEN_STEPS):

        # Crop context to the last SEQ_LEN tokens (positional embedding limit)
        ctx_tensor = torch.tensor(
            context[-SEQ_LEN:], dtype=torch.long
        ).unsqueeze(0)
        # ctx_tensor: [1, T]  where T = min(len(context), SEQ_LEN)

        # Forward through stage-0
        hidden = model(ctx_tensor)
        # hidden: [1, T, D_MODEL]

        # Send to worker1 for the second half of the forward pass + sampling
        send_msg(sock, hidden)

        # Receive the sampled next token
        next_id = recv_msg(sock)   # int

        context.append(next_id)

# Signal generation is complete
send_msg(sock, {"cmd": "DONE", "seed_text": GEN_SEED_TEXT})

print("[worker0] Generation complete. See worker1 terminal for output.")
sock.close()
