# pre_screening_layer.py
# Author: Furkan Nar — November 11, 2025
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Callable, Any, Union
import logging
import copy
import asyncio

# ----------------------------------------------------------------------
# 1. Imports from the rest of the project
# ----------------------------------------------------------------------
from main_memory import AsyncMemoryManager, ConversationTurn
from summarizer_agent import STYLE_TEMPLATES, summarize_tensor, SummarizerAgent
from settings import (
    ATTENTION_HEADS, EMBEDDING_DIM, DROPOUT_RATE,
    SEQUENCE_LENGTH, ATTENTION_CACHE_ENABLED
)

# ----------------------------------------------------------------------
# 2. ScalarGates – per-head gating 
# ----------------------------------------------------------------------
class ScalarGates(nn.Module):
    """K' = sigmoid(log @ W + b) * K | W: [d_k, 1]"""
    def __init__(self, d_k: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(d_k, 1))
        self.b = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.W, mean=0.0, std=0.02)

    def forward(self, K: torch.Tensor, log: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_k = K.shape
        log_flat = log.view(-1, d_k)                     # [N, d_k]
        gate_flat = torch.sigmoid(log_flat @ self.W + self.b)  # [N, 1]
        K_flat = K.view(-1, d_k)
        gated_flat = gate_flat * K_flat
        gated = gated_flat.view(batch_size, seq_len, d_k)
        gate = gate_flat.view(batch_size, seq_len, 1)
        return gated, gate


# ----------------------------------------------------------------------
# 3. Attention Cache + Entropy utilities
# ----------------------------------------------------------------------
class AttentionCache:
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
        self.logger = logging.getLogger(__name__)

    def get(self, key: str):
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None

    def put(self, key: str, value):
        if len(self.cache) >= self.max_size:
            lru_key = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_key]
            del self.access_count[lru_key]
        stored = value.clone().detach() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        self.cache[key] = stored
        self.access_count[key] = 1

    def clear(self):
        self.cache.clear()
        self.access_count.clear()


attention_cache = AttentionCache() if ATTENTION_CACHE_ENABLED else None


def entropy(attn_weights: torch.Tensor) -> float:
    if attn_weights is None or attn_weights.numel() == 0:
        return 0.0
    vec = attn_weights.view(-1)
    return -torch.sum(vec * torch.log(vec + 1e-8)).item()


def optimize_output(cache: AttentionCache, key: str,
                    output: torch.Tensor, attn_weights: torch.Tensor,
                    logger=None):
    cached = cache.get(key)
    if cached is not None:
        if logger:
            logger.info(f"Cache HIT: {key}")
        return cached

    mean_weights = attn_weights.mean(dim=1)          # [batch, seq, seq]
    score = entropy(mean_weights)
    if logger:
        logger.info(f"Attention Entropy: {score:.4f} | Key: {key}")

    if score < 2.5:
        cache.put(key, output)
        if logger:
            logger.info(f"Cache STORE: {key}")
    return output


# ----------------------------------------------------------------------
# 4. Scaled Dot-Product Attention 
# ----------------------------------------------------------------------
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask_bool = mask.bool() if mask.dtype != torch.bool else mask
            scores = scores.masked_fill(~mask_bool, float('-1e9'))

        scores = scores - scores.max(dim=-1, keepdim=True).values
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        output = torch.matmul(weights, value)
        return output, weights


# ----------------------------------------------------------------------
# 5. MemorialThalamus – the “reflexive memory” layer
# ----------------------------------------------------------------------
class MemorialThalamus:
    """
    Produces *reflex* summaries (emotional, code, intent, …) on top of a hidden
    representation.  It is fully async and can be called from any attention
    layer that has a SummarizerAgent available.
    Furkan has made the first version of the code, later on improved by AI
    """
    def __init__(self):
        # ConversationTurn is a callable that returns the current turn dict
        self.get_conversation = ConversationTurn

    # ------------------------------------------------------------------
    # Helper: one reflex
    # ------------------------------------------------------------------
    async def _process_reflex(self,
                              reflex_type: str,
                              summarizer: SummarizerAgent,
                              model_embedding_layer: Optional[object],
                              hidden: torch.Tensor,
                              user_id: str,
                              context_data: Optional[Dict] = None) -> torch.Tensor:
        style_map = {
            "emotional": "story",
            "code": "technical",
            "intent": "brief",
            "planner": "bullet",
            "deep_context": "deep"
        }
        style = style_map.get(reflex_type, "brief")
        return await summarize_tensor(
            summarizer=summarizer,
            model_embedding_layer=model_embedding_layer,
            hidden=hidden,
            user_id=user_id,
            style=style,
            context_data=context_data
        )

    # ------------------------------------------------------------------
    # Public API – multi-reflex summary
    # ------------------------------------------------------------------
    async def memory_summary(self,
                             hidden: torch.Tensor,
                             summarizer: SummarizerAgent,
                             model_embedding_layer: Optional[object],
                             user_id: str,
                             conversation_window: int = 10,
                             include_user_meta: bool = False,
                             filter_keywords: Optional[List[str]] = None,
                             summary_style: str = "bullet",
                             embedding_profile: Optional[str] = None,
                             include_reflexes: Union[bool, List[str]] = True,
                             reflex_sensitivity: Optional[float] = 0.75,
                             reflex_profile: Optional[str] = "default",
                             reflex_source: Optional[str] = "conversation",
                             reflex_timing: Optional[str] = "post_summary",
                             time_range: Optional[Tuple[str, str]] = None,
                             context_data: Optional[Dict] = None) -> Dict[str, torch.Tensor]:

        # Guard: zero window → just return the flag
        if conversation_window == 0:
            return {"reflexes": include_reflexes}

        # Resolve which reflexes to run
        reflexes: List[str] = []
        if isinstance(include_reflexes, list):
            reflexes.extend(include_reflexes)
        elif include_reflexes:
            reflexes.extend(["emotional", "code", "intent"])

        results: Dict[str, torch.Tensor] = {}
        for reflex in reflexes:
            tensor = await self._process_reflex(
                reflex_type=reflex,
                summarizer=summarizer,
                model_embedding_layer=model_embedding_layer,
                hidden=hidden,
                user_id=user_id,
                context_data=context_data
            )
            results[reflex] = tensor

        return results


# ----------------------------------------------------------------------
# 6. ReflectiveMultiHeadAttention now uses MemorialThalamus
# ----------------------------------------------------------------------
class ReflectiveMultiHeadAttention(nn.Module):
    """
    Final version:
      • Per-head ScalarGates
      • Entropy-based caching
      • Async MemorialThalamus summarisation (reflexes)
      • Safe fallback adapter
    """
    def __init__(self,
                 d_model: int = EMBEDDING_DIM,
                 num_heads: int = ATTENTION_HEADS,
                 dropout: float = DROPOUT_RATE,
                 cache: Optional[AttentionCache] = None,
                 summarizer: Optional[SummarizerAgent] = None,
                 hidden_adapter: Optional[nn.Module] = None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Standard MHA projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # Per-head gates
        self.scalar_gates = nn.ModuleList([ScalarGates(self.d_k) for _ in range(num_heads)])

        # Caching
        self.cache = cache or attention_cache

        # Summarizer (required for reflexes)
        self.summarizer = summarizer

        # Adapter for hidden states before summarisation
        self.hidden_adapter = hidden_adapter or nn.Identity()

        # MemorialThalamus instance (shared across all heads)
        self.thalamus = MemorialThalamus()

        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                cache_key: Optional[str] = None,
                logistics: Optional[torch.Tensor] = None,
                # ---- NEW ----
                user_id: str = "unknown",
                include_reflexes: Union[bool, List[str]] = True,
                reflex_context: Optional[Dict] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        batch_size, seq_len, _ = query.size()
        residual = query

        # 1. CACHE HIT
        if cache_key and self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached, None

        # 2. QKV projection
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 3. PER-HEAD GATING (logistics = optional log-tensor)
        if logistics is not None:
            if logistics.dim() == 3:
                logistics = logistics.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
            elif logistics.dim() != 4:
                raise ValueError("logistics must be [B, S, d_k] or [B, S, H, d_k]")

            gated_K_list = []
            for h in range(self.num_heads):
                K_h = K[:, h, :, :]
                log_h = logistics[:, :, h, :]
                gated_K_h, _ = self.scalar_gates[h](K_h, log_h)
                gated_K_list.append(gated_K_h)
            K = torch.stack(gated_K_list, dim=1)

        # 4. MASK & ATTENTION
        if mask is not None:
            mask = mask.unsqueeze(1)               # broadcast over heads
        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # 5. Output projection + residual
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(attn_output)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        # 6. ENTROPY-BASED CACHING
        if cache_key and self.cache:
            output = optimize_output(self.cache, cache_key, output, attn_weights, self.logger)

        # ------------------------------------------------------------------
        # 7.  ASYNC REFLEX SUMMARISATION via MemorialThalamus
        # ------------------------------------------------------------------
        if self.summarizer and include_reflexes:
            # Fire-and-forget – we do **not** block the forward pass.
            # The caller can await `self.last_reflex_task` if needed.
            hidden = self.hidden_adapter(output)

            async def _run_reflexes():
                return await self.thalamus.memory_summary(
                    hidden=hidden,
                    summarizer=self.summarizer,
                    model_embedding_layer=None,
                    user_id=user_id,
                    include_reflexes=include_reflexes,
                    context_data=reflex_context
                )

            self.last_reflex_task = asyncio.create_task(_run_reflexes())
        else:
            self.last_reflex_task = None

        return output, attn_weights

    # ------------------------------------------------------------------
    # Helper to await the latest reflex dict 
    # ------------------------------------------------------------------
    async def get_latest_reflexes(self) -> Optional[Dict[str, torch.Tensor]]:
        if hasattr(self, "last_reflex_task") and self.last_reflex_task:
            return await self.last_reflex_task
        return None


# ----------------------------------------------------------------------
# 8. PositionalEncoding 
# ----------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int = EMBEDDING_DIM,
                 max_len: int = SEQUENCE_LENGTH,
                 encoding_type: str = "fixed"):
        super().__init__()
        self.encoding_type = encoding_type
        if encoding_type == "fixed":
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                                 (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(1)
            self.register_buffer('pe', pe)
        else:
            self.pe = nn.Parameter(torch.randn(max_len, d_model))
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoding_type == "fixed":
            x = x + self.pe[:x.size(0), :]
        else:
            x = x + self.pe[:x.size(0), :].unsqueeze(1)
        return self.dropout(x)


# ----------------------------------------------------------------------
# 9. AttentionVisualizer 
# ----------------------------------------------------------------------
class AttentionVisualizer:
    def __init__(self, attention_tensor: torch.Tensor, tokens: List[str]):
        self.attention_tensor = attention_tensor
        self.tokens = tokens

    def analyze_layer(self, layer_idx: int, query_idx: int):
        layer = self.attention_tensor[layer_idx]
        avg_attn = layer.mean(dim=0)[query_idx].detach().cpu()
        topk = torch.topk(avg_attn, k=min(5, avg_attn.numel()))
        top_tokens = [
            {"token": self.tokens[i], "attention": float(w)}
            for i, w in zip(topk.indices, topk.values)
        ]
        entropies = [entropy(layer[h][query_idx].detach().cpu())
                     for h in range(len(layer))]
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
        return {"avg_entropy": avg_entropy, "top_tokens_averaged": top_tokens}


def quick_visualize(attention_tensor, tokens, query_token,
                    layer_idx=0, summary_style=None):
    if query_token not in tokens:
        print(f"Token '{query_token}' not found")
        return
    viz = AttentionVisualizer(attention_tensor, tokens)
    idx = tokens.index(query_token)
    analysis = viz.analyze_layer(layer_idx, idx)
    print(f"\nQuery: '{query_token}' | Avg Entropy: {analysis['avg_entropy']:.4f}")
    print("Top attended:")
    for t in analysis['top_tokens_averaged']:
        print(f" '{t['token']}': {t['attention']:.4f}")
