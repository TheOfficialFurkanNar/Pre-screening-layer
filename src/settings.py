# config/settings.py
#NOTE: I used a OpenAI model for this one to censor the original PioneerAI model (My own Proto-AGI system)
#You can modify the model with your own version.

import os
from pathlib import Path
from dotenv import load_dotenv

# -------------------------
# Helper type conversion functions
# -------------------------
def _b(val, default=False):
    """Boolean parse"""
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}

def _i(val, default=0):
    """Int parse"""
    try:
        return int(str(val).strip())
    except Exception:
        return default

def _f(val, default=0.0):
    """Float parse"""
    try:
        return float(str(val).strip())
    except Exception:
        return default

def _stripq(s):
    """Çift/tek tırnakları temizle"""
    if s is None:
        return ""
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s

# OpenAI Settings #Used OpenAI to censor the original PioneerAI model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_MODEL_STREAM = os.getenv("OPENAI_MODEL_STREAM", OPENAI_MODEL)
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
HTTP_TIMEOUT = _i(os.getenv("HTTP_TIMEOUT_SEC", os.getenv("HTTP_TIMEOUT", 30)), 30)

# Model parameters
MAX_TOKENS = _i(os.getenv("MAX_TOKENS", 500), 500)
TEMPERATURE = _f(os.getenv("TEMPERATURE", 0.7), 0.7)
TOP_P = _f(os.getenv("TOP_P", 1.0), 1.0)
FREQUENCY_PENALTY = _f(os.getenv("FREQUENCY_PENALTY", 0.0), 0.0)
PRESENCE_PENALTY = _f(os.getenv("PRESENCE_PENALTY", 0.0), 0.0)

# Session timeout
SESSION_TIMEOUT = _i(os.getenv("SESSION_TIMEOUT_SEC", os.getenv("SESSION_TIMEOUT", 300)), 300)
# You can change this
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
MEMORY_JSON = os.getenv("MEMORY_JSON", str(DATA_DIR / "memory.json"))
MEMORY_TXT = os.getenv("MEMORY_TXT", str(DATA_DIR / "conversation_memory.txt"))

# Summary and context limits
SUMMARY_TIMEOUT = _i(os.getenv("SUMMARY_TIMEOUT_SEC", os.getenv("SUMMARY_TIMEOUT", 15)), 15)
MAX_CONTEXT_TOKENS = _i(os.getenv("MAX_CONTEXT_TOKENS", 2000), 2000)

# Text encoding options.
ENCODING = os.getenv("ENCODING", "utf-8")
ENSURE_ASCII = _b(os.getenv("ENSURE_ASCII", "false"), False)

# Attention Mechanism Configuration
ATTENTION_HEADS = _i(os.getenv("ATTENTION_HEADS", 8), 8)
EMBEDDING_DIM = _i(os.getenv("EMBEDDING_DIM", 512), 512)
SEQUENCE_LENGTH = _i(os.getenv("SEQUENCE_LENGTH", 1024), 1024)
DROPOUT_RATE = _f(os.getenv("DROPOUT_RATE", 0.1), 0.1)
LEARNING_RATE = _f(os.getenv("LEARNING_RATE", 0.0001), 0.0001)
NUM_ENCODER_LAYERS = _i(os.getenv("NUM_ENCODER_LAYERS", 6), 6)
NUM_DECODER_LAYERS = _i(os.getenv("NUM_DECODER_LAYERS", 6), 6)

POSITIONAL_ENCODING_TYPE = os.getenv("POSITIONAL_ENCODING_TYPE", "fixed")
HYBRID_MODE_THRESHOLD = _f(os.getenv("HYBRID_MODE_THRESHOLD", 0.8), 0.8)
CACHE_SIZE = _i(os.getenv("CACHE_SIZE", 1000), 1000)
