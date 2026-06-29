"""
Custom Decision Transformer implementation in pure PyTorch.

Architecture mirrors the original DT paper (Chen et al. 2021):
  - GPT-2 style causal transformer encoder
  - Tokens are interleaved as [r_0, s_0, a_0, r_1, s_1, a_1, ...]
  - Action predictions come from the state token positions

jaxtyping annotations are used throughout to document & check tensor shapes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int, Bool
from torch import Tensor


# ---------------------------------------------------------------------------
# Causal Self-Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (autoregressive) self-attention.

    Parameters
    ----------
    hidden_size : int  – model dimension d
    n_head      : int  – number of attention heads (d must be divisible by n_head)
    attn_pdrop  : float – dropout on attention weights
    """

    def __init__(self, hidden_size: int, n_head: int, attn_pdrop: float = 0.1):
        super().__init__()
        assert hidden_size % n_head == 0, "hidden_size must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = hidden_size // n_head

        # Fused QKV projection
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_drop = nn.Dropout(attn_pdrop)

    def forward(
        self,
        x: Float[Tensor, "batch seq d"],
        attn_bias: Float[Tensor, "1 1 seq seq"] | None = None,
    ) -> Float[Tensor, "batch seq d"]:
        B, T, d = x.shape
        H, dh = self.n_head, self.head_dim

        # Project to Q, K, V  →  (B, T, 3d)
        qkv: Float[Tensor, "batch seq 3d"] = self.qkv(x)
        q, k, v = qkv.split(d, dim=-1)  # each (B, T, d)

        # Reshape to multi-head form  →  (B, H, T, dh)
        def to_heads(t: Float[Tensor, "batch seq d"]) -> Float[Tensor, "batch heads seq head_dim"]:
            return t.view(B, T, H, dh).transpose(1, 2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        # Scaled dot-product scores  →  (B, H, T, T)
        scale = math.sqrt(dh)
        scores: Float[Tensor, "batch heads seq seq"] = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Causal mask: upper-triangle positions get -inf so they softmax to 0
        causal: Bool[Tensor, "seq seq"] = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, float("-inf"))

        # Optional padding-aware bias (from attention_mask, added before softmax)
        if attn_bias is not None:
            scores = scores + attn_bias

        weights: Float[Tensor, "batch heads seq seq"] = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        # Aggregate values  →  (B, H, T, dh)  →  (B, T, d)
        out: Float[Tensor, "batch seq d"] = (
            torch.matmul(weights, v)
            .transpose(1, 2)
            .contiguous()
            .view(B, T, d)
        )
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm transformer block: LN → Attention → residual → LN → FFN → residual.

    Parameters
    ----------
    hidden_size : int   – model dimension
    n_head      : int   – number of attention heads
    n_inner     : int   – inner FFN dimension (typically 4 * hidden_size)
    resid_pdrop : float – dropout on residuals
    attn_pdrop  : float – dropout on attention weights
    """

    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        n_inner: int,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, n_head, attn_pdrop)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, n_inner),
            nn.ReLU(),
            nn.Linear(n_inner, hidden_size),
        )
        self.drop = nn.Dropout(resid_pdrop)

    def forward(
        self,
        x: Float[Tensor, "batch seq d"],
        attn_bias: Float[Tensor, "1 1 seq seq"] | None = None,
    ) -> Float[Tensor, "batch seq d"]:
        x = x + self.drop(self.attn(self.ln1(x), attn_bias))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


# ---------------------------------------------------------------------------
# Decision Transformer
# ---------------------------------------------------------------------------

class MuJoCoDecisionTransformer(nn.Module):
    """
    Decision Transformer for continuous-action MuJoCo environments.

    Sequence layout per timestep (interleaved):
        [RTG_t, state_t, action_t]  → flattened to length 3*T

    Actions are predicted from the *state* token positions (index 1 in each triple).

    Parameters
    ----------
    state_dim   : int   – observation dimension
    act_dim     : int   – action dimension
    hidden_size : int   – transformer model dimension (default 128)
    max_ep_len  : int   – maximum episode length for timestep embedding
    n_layer     : int   – number of transformer blocks (default 3)
    n_head      : int   – number of attention heads (default 1)
    n_inner     : int   – FFN inner dimension (default 4 * hidden_size)
    resid_pdrop : float – residual dropout rate
    attn_pdrop  : float – attention dropout rate
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_size: int = 128,
        max_ep_len: int = 1000,
        n_layer: int = 3,
        n_head: int = 1,
        n_inner: int | None = None,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size

        if n_inner is None:
            n_inner = 4 * hidden_size

        # --- Input embedders ---
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)
        self.embed_rtg = nn.Linear(1, hidden_size)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)

        # --- Transformer stack ---
        self.embed_drop = nn.Dropout(resid_pdrop)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, n_head, n_inner, resid_pdrop, attn_pdrop)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)

        # --- Action head (continuous, Tanh-bounded) ---
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, act_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        states: Float[Tensor, "batch seq state_dim"],
        actions: Float[Tensor, "batch seq act_dim"],
        returns_to_go: Float[Tensor, "batch seq 1"],
        timesteps: Int[Tensor, "batch seq"],
        attention_mask: Int[Tensor, "batch seq"] | None = None,
    ) -> Float[Tensor, "batch seq act_dim"]:
        """
        Parameters
        ----------
        states         : (B, T, state_dim)  – normalised observations
        actions        : (B, T, act_dim)    – previous actions (zero-padded at t=0)
        returns_to_go  : (B, T, 1)          – return-to-go targets
        timesteps      : (B, T)             – absolute timestep indices (int)
        attention_mask : (B, T) or None     – 1 for real tokens, 0 for padding

        Returns
        -------
        action_preds   : (B, T, act_dim)    – predicted continuous actions
        """
        B, T, _ = states.shape

        # --- Embed all token types  →  each (B, T, d) ---
        time_emb: Float[Tensor, "batch seq d"] = self.embed_timestep(timesteps)

        s_emb: Float[Tensor, "batch seq d"] = self.embed_state(states) + time_emb
        a_emb: Float[Tensor, "batch seq d"] = self.embed_action(actions) + time_emb
        r_emb: Float[Tensor, "batch seq d"] = self.embed_rtg(returns_to_go) + time_emb

        # --- Interleave [r_0, s_0, a_0, r_1, s_1, a_1, ...]  →  (B, 3T, d) ---
        # Stack along a new "token type" axis, then merge into sequence dim
        stacked: Float[Tensor, "batch seq 3 d"] = torch.stack(
            (r_emb, s_emb, a_emb), dim=2
        )  # (B, T, 3, d)
        x: Float[Tensor, "batch seq3 d"] = stacked.reshape(B, 3 * T, self.hidden_size)
        x = self.embed_drop(x)

        # --- Build attention bias from padding mask (optional) ---
        attn_bias: Float[Tensor, "1 1 seq3 seq3"] | None = None
        if attention_mask is not None:
            # Expand mask to cover all three tokens per timestep
            mask3: Int[Tensor, "batch seq3"] = attention_mask.repeat_interleave(3, dim=1)
            # Positions where key is padded  →  subtract large value from scores
            # Shape: (B, 1, 1, 3T) broadcast over (B, H, 3T, 3T)
            pad_bias: Float[Tensor, "batch 1 1 seq3"] = (
                (1 - mask3.float()).unsqueeze(1).unsqueeze(2) * -1e9
            )
            attn_bias = pad_bias  # type: ignore[assignment]

        # --- Transformer layers ---
        for block in self.blocks:
            x = block(x, attn_bias)

        x = self.ln_f(x)  # (B, 3T, d)

        # --- Extract state-token positions (index 1 in each [r,s,a] triple) ---
        # Reshape back to (B, T, 3, d), then pick token-type index 1
        x_typed: Float[Tensor, "batch seq 3 d"] = x.reshape(B, T, 3, self.hidden_size)
        state_tokens: Float[Tensor, "batch seq d"] = x_typed[:, :, 1, :]  # (B, T, d)

        action_preds: Float[Tensor, "batch seq act_dim"] = self.predict_action(state_tokens)
        return action_preds
