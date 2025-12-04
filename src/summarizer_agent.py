#src/summarizer_agent.py
import os
import hashlib
import importlib
import pkgutil
from typing import Literal, Dict, Optional
import torch
from diskcache import Cache

from settings import CONFIDENCE_THRESHOLD, SUMMARY_TIMEOUT, OPENAI_MODEL, MEMORY_JSON


async def summarize_tensor(
        summarizer: "SummarizerAgent",
        model_embedding_layer: object,
        hidden: torch.Tensor,
        user_id: str,
        style: str = "brief",
        context_data: object = None
) -> torch.Tensor:
    """
    Convert hidden tensor → textual summary → re-embed → shape-preserving tensor.

    hidden: [batch, seq_len, d_model]
    returns: [batch, seq_len, d_model]
    :rtype: torch.Tensor
    """
    batch, seq_len, d_model = hidden.shape

    # Step 1 — Extract semantic message from tensor
    pooled = hidden.mean(dim=1)  # [batch, d_model]
    # detokenization not required — summarizer uses text prompt only

    # Step 2 — Format a text message
    message = f"Tensor semantic state: {pooled.tolist()}"

    # Step 3 — Generate textual summary using SummarizerAgent
    result = await summarizer.generate_summary(
        user_id=user_id,
        message=message,
        style=style,
        context_data=context_data,
        force_local=True
    )

    summary_text = result["summary"]

    # Step 4 — Convert summary text to tokens → embeddings
    tokens = summarizer.tokenizer.encode(summary_text, add_special_tokens=True)
    token_tensor = torch.tensor(tokens).unsqueeze(0).to(hidden.device)

    # embedding: [1, len, d_model]
    embedded = model_embedding_layer(token_tensor)

    # Step 5 — Pool to a single semantic vector
    summary_vec = embedded.mean(dim=1)  # [1, d_model]

    # Step 6 — Broadcast to full sequence
    summary_expanded = summary_vec.unsqueeze(1).expand(batch, seq_len, d_model)

    # Step 7 — Merge with original hidden state
    # Gentle correction — avoid destabilizing attention
    gate = torch.sigmoid((hidden.mean(dim=-1, keepdim=True)))  # [batch, seq_len, 1]

    return hidden + gate * summary_expanded

STYLE_TEMPLATES: Dict[str, str] = {
    "brief": (
        "Aşağıdaki bağlam ve intent'e göre kullanıcı mesajını "
        "kısa ve öz bir şekilde özetle."
    ),
    "deep": (
        "Aşağıdaki bağlam ve intent'e göre kullanıcı mesajını "
        "detaylı ve kapsamlı bir şekilde özetle."
    ),
    "bullet": (
        "Aşağıdaki bağlam ve intent'e göre kullanıcı mesajını "
        "madde işaretleriyle özetle."
    ),
    "technical": (
        "Aşağıdaki bağlam ve intent'e göre kullanıcı mesajını "
        "teknik ve ayrıntılı bir şekilde özetle."
    ),
    "story": (
        "Aşağıdaki bağlam ve intent'e göre kullanıcı mesajını "
        "hikaye formatında özetle."
    )
}


def _load_style_plugins():
    pkg_name = __name__.rsplit('.', 1)[0] + ".summary_styles"
    try:
        package = importlib.import_module(pkg_name)
    except ModuleNotFoundError:
        return
    for finder, name, is_pkg in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{pkg_name}.{name}")
        tpl = getattr(module, "STYLE_TEMPLATE", None)
        if isinstance(tpl, dict):
            STYLE_TEMPLATES.update(tpl)


_load_style_plugins()

# Cache setup
CACHE_DIR = os.getenv("SUMMARY_CACHE_DIR", ".prescreening_cache")
if Cache:
    cache = Cache(CACHE_DIR)
else:
    cache: Dict[str, tuple[str, float]] = {}


def _make_cache_key(user_id: str, message: str, style: str, context_data: Optional[Dict] = None) -> str:
    content = f"{user_id}|{style}|{message}"
    if context_data:
        content += "|" + context_data.get("summary", "") + "|" + "|".join(context_data.get("example_messages", []))
    return hashlib.sha256(content.encode()).hexdigest()

class SummarizerAgent:
    def __init__(
            self,
            memory_path: str = MEMORY_JSON,
            model_name: str = OPENAI_MODEL,
            timeout: int = SUMMARY_TIMEOUT,
            confidence_threshold: float = CONFIDENCE_THRESHOLD  # Moved to config
    ):
        return






