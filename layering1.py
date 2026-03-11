"""
worker1.py — Pipeline Parallel Training, Stage 1

Owns:
  - Transformer blocks 4–7
  - Final LayerNorm
  - Linear language-model head  → logits [batch, seq_len, vocab_size]

Communication (training):
  - RECEIVES hidden state tensor   [batch, seq_len, d_model]  ← worker0
  - SENDS    gradient tensor       [batch, seq_len, d_model]  → worker0

Communication (generation):
  - RECEIVES hidden state          [1, T, d_model]            ← worker0
  - SENDS    next token id (int)                              → worker0
  ... repeated GEN_STEPS times, then receives {"cmd": "DONE"}

Run this script FIRST — it is the TCP server.
"""

import socket
import pickle
import struct
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import urllib.request

# ─────────────────────────────────────────────
# Hyperparameters  (must match worker0.py)
# ─────────────────────────────────────────────
D_MODEL    = 128
N_HEADS    = 4
D_FF       = D_MODEL * 4
SEQ_LEN    = 256
BATCH_SIZE = 16
LR         = 3e-4
NUM_STEPS  = 15000
SEED       = 42

# Generation settings
GEN_STEPS       = 500   # characters to generate after training
GEN_TEMPERATURE = 0.8   # < 1.0 = sharper, > 1.0 = more random
GEN_TOP_K       = 40    # sample only from the top-k most likely chars

HOST = "0.0.0.0"
PORT = 29500

# ─────────────────────────────────────────────
# 1. Dataset & Tokenizer  (identical to worker0)
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

print(f"[worker1] Vocabulary size : {vocab_size}")
print(f"[worker1] Corpus length   : {len(data)} tokens")

# ─────────────────────────────────────────────
# 2. Batch Sampler — same seed as worker0
# ─────────────────────────────────────────────
rng = torch.Generator()
rng.manual_seed(SEED)

def get_batch():
    ix = torch.randint(len(data) - SEQ_LEN - 1, (BATCH_SIZE,), generator=rng)
    x  = torch.stack([data[i : i + SEQ_LEN]         for i in ix])
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


class Stage1Model(nn.Module):
    """
    Back half of the transformer.
      Training input  : hidden [B, T, D_MODEL]   ← received from worker0
      Generation input: hidden [1, T, D_MODEL]   ← one sequence at a time
      Output          : logits [B, T, vocab_size]
    """

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_blocks=4):
        super().__init__()
        self.blocks   = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_blocks)]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x      = self.ln_final(x)
        logits = self.lm_head(x)
        return logits


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
# 5. Generation Sampler
# ─────────────────────────────────────────────

def sample_next_token(logits_last_pos, temperature, top_k):
    """
    logits_last_pos : [vocab_size]  — raw scores for the next character

    Steps:
      1. Divide by temperature  →  sharpen or flatten the distribution
      2. Keep only the top-k scores, set the rest to -inf
      3. Softmax  →  valid probability distribution
      4. Multinomial sample  →  one token index
    """
    logits = logits_last_pos / temperature

    # Top-k filter
    if top_k > 0:
        top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        threshold   = top_vals[-1]                          # smallest kept value
        logits      = logits.masked_fill(logits < threshold, float("-inf"))

    probs    = F.softmax(logits, dim=-1)                    # [vocab_size]
    next_tok = torch.multinomial(probs, num_samples=1)      # [1]
    return next_tok.item()


# ─────────────────────────────────────────────
# 6. Build Model & Optimizer
# ─────────────────────────────────────────────
torch.manual_seed(SEED)
model     = Stage1Model(vocab_size, D_MODEL, N_HEADS, D_FF, n_blocks=4)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn   = nn.CrossEntropyLoss()

n_params = sum(p.numel() for p in model.parameters())
print(f"[worker1] Stage-1 parameters: {n_params:,}")

# ─────────────────────────────────────────────
# 7. Start TCP Server, wait for worker0
# ─────────────────────────────────────────────
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)
print(f"[worker1] Listening on {HOST}:{PORT} — waiting for worker0 ...")
conn, addr = server.accept()
print(f"[worker1] worker0 connected from {addr}")

# ─────────────────────────────────────────────
# 8. Training Loop
# ─────────────────────────────────────────────
loss_accum = 0.0

for step in range(1, NUM_STEPS + 1):

    model.train()
    optimizer.zero_grad()

    _, y = get_batch()
    # y: [BATCH_SIZE, SEQ_LEN]

    # Receive activation from worker0
    hidden_recv = recv_msg(conn)
    # hidden_recv: [BATCH_SIZE, SEQ_LEN, D_MODEL]

    # Graph anchoring — register as leaf so .grad is populated after backward
    boundary = hidden_recv.detach().requires_grad_(True)

    logits = model(boundary)
    # logits: [BATCH_SIZE, SEQ_LEN, vocab_size]

    B, T, V = logits.shape
    loss = loss_fn(
        logits.view(B * T, V),
        y.view(B * T)
    )

    loss_accum += loss.item()
    loss.backward()

    assert boundary.grad is not None, "boundary.grad is None — autograd graph broken"
    send_msg(conn, boundary.grad)
    # Transmitted gradient shape: [BATCH_SIZE, SEQ_LEN, D_MODEL]

    optimizer.step()

    if step % 100 == 0:
        avg_loss = loss_accum / 100
        print(f"[worker1] step {step}/{NUM_STEPS}  |  avg loss: {avg_loss:.4f}")
        loss_accum = 0.0

print("[worker1] Training complete.")

# ─────────────────────────────────────────────
# 9. Autoregressive Text Generation
# ─────────────────────────────────────────────
#
# Protocol (one character at a time):
#
#   worker0  ──► {"cmd": "GENERATE", "context": [tok, tok, ...], "seed_text": str}
#   loop GEN_STEPS times:
#     worker0  ──► hidden tensor  [1, T, D_MODEL]
#     worker1  ──► next_token_id  (int)
#   worker0  ──► {"cmd": "DONE"}
#
# worker1 owns the vocab (itos) and the lm_head, so it both samples
# the token AND can decode the final output for display.

print("\n" + "─" * 60)
print("[worker1] Entering generation mode ...")
print("─" * 60)

model.eval()

# Receive seed context from worker0
msg = recv_msg(conn)
assert msg["cmd"] == "GENERATE", f"Expected GENERATE, got {msg['cmd']}"

seed_text     = msg["seed_text"]
generated_ids = list(msg["context"])   # list of ints, starts as the seed

print(f"[worker1] Seed      : {repr(seed_text)}")
print(f"[worker1] Generating {GEN_STEPS} characters ...\n")

with torch.no_grad():
    for _ in range(GEN_STEPS):

        # worker0 sends the current context window through its half of the model
        hidden = recv_msg(conn)
        # hidden: [1, T, D_MODEL]  where T = min(len(context), SEQ_LEN)

        # Run through stage-1 layers
        logits = model(hidden)
        # logits: [1, T, vocab_size]

        # Only the last position predicts the NEXT character
        next_logits = logits[0, -1, :]         # [vocab_size]

        # Sample under temperature + top-k
        next_id = sample_next_token(next_logits, GEN_TEMPERATURE, GEN_TOP_K)

        # Return the chosen token to worker0 so it can update the context
        send_msg(conn, next_id)

        generated_ids.append(next_id)

# Receive termination signal
msg = recv_msg(conn)
assert msg["cmd"] == "DONE"

# ─── Print the result ────────────────────────────────────────────
full_text  = decode(generated_ids)
seed_chars = len(seed_text)

print("═" * 60)
print("GENERATED TEXT")
print("═" * 60)
print(full_text)
print("═" * 60)
print(f"\n[{seed_chars} seed chars] + [{GEN_STEPS} generated chars]")
print(f"[temperature={GEN_TEMPERATURE}, top_k={GEN_TOP_K}]")

conn.close()
server.close()
