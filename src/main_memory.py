# src/memory_manager.py

from __future__ import annotations

import asyncio
import json
import time
import html
import shutil
import tempfile
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from collections import deque
import aiofiles

from settings import MEMORY_JSON, MEMORY_TXT, ENCODING

# ----------------------------
# Data Models
# ----------------------------
SCHEMA_VERSION = "1.0"

def _sanitize(s: Any) -> str:
    if s is None:
        return ""
    if isinstance(s, (str, bytes)):
        return html.escape(str(s))
    if asyncio.iscoroutine(s):
        raise ValueError(f"Coroutine passed to _sanitize; must be awaited: {s}")
    return html.escape(str(s))

@dataclass
class ConversationTurn:
    user: str
    bot: str
    timestamp: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    model_used: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    def __post_init__(self) -> None:
        self.user = _sanitize(self.user)
        self.bot = _sanitize(self.bot)
        if self.user_id is not None:
            self.user_id = _sanitize(self.user_id)
        if self.session_id is not None:
            self.session_id = _sanitize(self.session_id)

    def to_dict(self) -> Dict[str, Any]:
        """Plain JSON-serializable dict (no html escaping added here)."""
        return {
            "user": self.user,
            "bot": self.bot,
            "timestamp": self.timestamp,
            "intent": self.intent,
            "confidence": self.confidence,
            "model_used": self.model_used,
            "user_id": self.user_id,
            "session_id": self.session_id,
        }

@dataclass
class CacheEntry:
    value: Any
    expires_at: Optional[float] = None
    access_count: int = 0
    created_at: float = field(default_factory=lambda: time.time())

# ----------------------------
# Memory Manager
# ----------------------------
class AsyncMemoryManager:
    def __init__(
        self,
        max_history: int = 20,
        max_memory_mb: int = 100,
        max_cache_size: int = 1000,
        write_buffer_size: int = 5,
        backup_enabled: bool = True,
        flush_interval_sec: float = 3.0,
        snapshot_max_mb: Optional[int] = None
    ) -> None:
        # Config
        self.max_history = max_history
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.max_cache_size = int(max_cache_size)
        self.write_buffer_size = int(write_buffer_size)
        self.backup_enabled = bool(backup_enabled)
        self.flush_interval_sec = float(flush_interval_sec)
        self.snapshot_max_bytes = None if snapshot_max_mb is None else int(snapshot_max_mb * 1024 * 1024)

        # State
        self.conversation_history: deque[ConversationTurn] = deque(maxlen=max_history)
        self.pending_writes: List[ConversationTurn] = []
        self._flush_task: Optional[asyncio.Task] = None
        self._flush_event = asyncio.Event()

        # Locks
        self._write_lock = asyncio.Lock()
        self._cache_lock = asyncio.Lock()

        # Cache
        self._cache: Dict[str, CacheEntry] = {}

        # Telemetry
        self._flush_count = 0
        self._load_count = 0
        self._error_count = 0

        # Logging
        self.logger = logging.getLogger(__name__)

        # Paths (harden)
        self.json_path = self._safe_path(MEMORY_JSON)
        self.text_path = self._safe_path(MEMORY_TXT)

        self._ensure_directories()
        self.logger.info(
            "AsyncMemoryManager init "
            f"(max_history={max_history}, buffer={write_buffer_size}, interval={flush_interval_sec}s)"
        )

    # ---------- Lifecycle ----------
    async def __aenter__(self) -> "AsyncMemoryManager":
        await self.load_memory()
        self._start_background_flusher()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.cleanup()

    # ---------- Directory/Path ----------
    def _safe_path(self, path_str: str) -> Path:
        """Normalize path, forbid directory traversal outside cwd/project."""
        p = Path(path_str).expanduser().resolve()
        return p

    def _ensure_directories(self) -> None:
        try:
            self.json_path.parent.mkdir(parents=True, exist_ok=True)
            self.text_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Error creating memory directories: {e}")
            raise

    # ---------- Memory Usage ----------
    def _approx_bytes_turn(self, t: ConversationTurn) -> int:
        """Approximate byte size of a turn, with robust type checking."""
        def b(s: Optional[Any], field_name: str) -> int:
            try:
                if s is None:
                    return 0
                if asyncio.iscoroutine(s):
                    self.logger.error(f"Coroutine found in field {field_name}: {s}")
                    raise ValueError(f"Coroutine passed to _approx_bytes_turn in {field_name}")
                if isinstance(s, (str, bytes)):
                    return len(s.encode(ENCODING or "utf-8", errors="ignore") if isinstance(s, str) else s)
                self.logger.warning(f"Unexpected type {type(s)} in field {field_name}; converting to str")
                return len(str(s).encode(ENCODING or "utf-8", errors="ignore"))
            except Exception as e:
                self.logger.error(f"Error processing field {field_name}: {e}")
                return 0

        try:
            return (
                b(t.user, "user") +
                b(t.bot, "bot") +
                b(t.timestamp, "timestamp") +
                b(t.intent, "intent") +
                16  # rough overhead for float/bools/fields
            )
        except Exception as e:
            self.logger.error(f"_approx_bytes_turn failed for turn {t}: {e}")
            return 0

    async def _check_memory_usage(self) -> None:
        """Check and trim memory usage if exceeding limits."""
        try:
            conv_bytes = sum(self._approx_bytes_turn(t) for t in self.conversation_history)
            cache_bytes = 0
            for k, v in self._cache.items():
                cache_bytes += len(k.encode(ENCODING or "utf-8", errors="ignore"))
                if isinstance(v.value, (str, bytes)):
                    cache_bytes += len(v.value if isinstance(v.value, bytes) else v.value.encode(ENCODING or "utf-8", errors="ignore"))
                else:
                    cache_bytes += 64  # rough estimate for non-string values

            total = conv_bytes + cache_bytes
            if total > self.max_memory_bytes:
                target = max(1, len(self.conversation_history) // 2)
                original = len(self.conversation_history)
                while len(self.conversation_history) > target:
                    self.conversation_history.popleft()
                self.logger.warning(
                    f"Memory usage {total/1024/1024:.2f} MB exceeded limit; "
                    f"trimmed history from {original} to {len(self.conversation_history)} turns."
                )
        except Exception as e:
            self.logger.error(f"Memory usage check failed: {e}")

    # ---------- Persistence ----------
    def _create_backup(self, file_path: Path) -> Optional[Path]:
        if not self.backup_enabled or not file_path.exists():
            return None
        backup = file_path.with_suffix(file_path.suffix + f".backup.{int(time.time())}")
        try:
            shutil.copy2(file_path, backup)
            return backup
        except Exception as e:
            self.logger.warning(f"Backup failed for {file_path}: {e}")
            return None

    def _validate_payload(self, data: Dict[str, Any]) -> bool:
        return (
            isinstance(data, dict)
            and data.get("schema", {}).get("version") == SCHEMA_VERSION
            and isinstance(data.get("conversation"), list)
        )

    async def load_memory(self) -> bool:
        """Load conversation from JSON snapshot; tolerant of corruption."""
        self._load_count += 1
        try:
            if not self.json_path.exists():
                self.logger.info("No memory snapshot found; starting fresh.")
                return True

            async with aiofiles.open(self.json_path, "r", encoding=ENCODING) as f:
                raw = await f.read()

            if not raw.strip():
                self.logger.info("Empty memory snapshot; starting fresh.")
                return True

            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                self.logger.error(f"Corrupt JSON snapshot ({self.json_path}): {e}; starting fresh.")
                return True

            if not self._validate_payload(data):
                if isinstance(data, dict) and isinstance(data.get("conversation"), list):
                    self.logger.warning("Old memory schema detected; importing without schema meta.")
                else:
                    self.logger.error("Invalid memory schema; starting fresh.")
                    return True

            loaded = 0
            for turn in data.get("conversation", []):
                try:
                    ct = ConversationTurn(**turn)
                    self.conversation_history.append(ct)
                    loaded += 1
                except Exception as e:
                    self.logger.warning(f"Skipping invalid turn: {e}")

            self.logger.info(f"Loaded {loaded} turns from snapshot.")
            await self._check_memory_usage()
            return True
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"load_memory failed: {e}")
            return True  # don't block app startup

    async def _write_snapshot_atomic(self, payload: Dict[str, Any]) -> None:
        """Atomic write of full JSON snapshot."""
        temp_file: Optional[Path] = None
        try:
            if self.snapshot_max_bytes is not None:
                estimate = len(json.dumps(payload, ensure_ascii=False).encode(ENCODING or "utf-8"))
                if estimate > self.snapshot_max_bytes:
                    self.logger.warning(
                        f"Snapshot size {estimate/1024/1024:.2f} MB exceeds cap; "
                        "pruning oldest turns."
                    )
                    while estimate > self.snapshot_max_bytes and len(payload["conversation"]) > 1:
                        payload["conversation"].pop(0)
                        estimate = len(json.dumps(payload, ensure_ascii=False).encode(ENCODING or "utf-8"))

            self.json_path.parent.mkdir(parents=True, exist_ok=True)
            self._create_backup(self.json_path)

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding=ENCODING,
                dir=str(self.json_path.parent),
                delete=False,
            ) as tf:
                json.dump(payload, tf, ensure_ascii=False, indent=2)
                temp_file = Path(tf.name)

            shutil.move(str(temp_file), str(self.json_path))
        except Exception as e:
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            raise e

    async def flush_to_disk(self) -> bool:
        if not self.pending_writes:
            return True

        async with self._write_lock:
            if not self.pending_writes:
                return True

            try:
                payload = {
                    "schema": {"version": SCHEMA_VERSION, "generated_at": datetime.utcnow().isoformat()},
                    "conversation": [t.to_dict() for t in self.conversation_history],
                }

                await self._write_snapshot_atomic(payload)

                try:
                    async with aiofiles.open(self.text_path, "a", encoding=ENCODING) as f:
                        for t in self.pending_writes:
                            await f.write(f"[{t.timestamp}] USER: {t.user}\n[{t.timestamp}] BOT : {t.bot}\n\n")
                except Exception as e:
                    self.logger.warning(f"Failed writing text log {self.text_path}: {e}")

                flushed = len(self.pending_writes)
                self.pending_writes.clear()
                self._flush_count += 1
                self.logger.info(f"Flushed {flushed} turn(s) to disk (snapshot #{self._flush_count}).")
                return True
            except Exception as e:
                self._error_count += 1
                self.logger.error(f"flush_to_disk failed: {e}")
                return False

    # ---------- Background Flusher ----------
    def _start_background_flusher(self) -> None:
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush_loop(), name="memory-flusher")

    async def _flush_loop(self) -> None:
        """Debounced flushing: trigger on buffer size or interval."""
        try:
            while True:
                try:
                    await asyncio.wait_for(self._flush_event.wait(), timeout=self.flush_interval_sec)
                    self._flush_event.clear()
                    if self.pending_writes:
                        await self.flush_to_disk()
                except asyncio.TimeoutError:
                    pass  # periodic tick
                except asyncio.CancelledError:
                    if self.pending_writes:
                        await self.flush_to_disk()
                    raise
                except Exception as e:
                    self._error_count += 1
                    self.logger.error(f"Flusher loop error: {e}")
        finally:
            try:
                if self.pending_writes:
                    await self.flush_to_disk()
            except Exception:
                pass

    # ---------- Conversation Ops ----------
    async def add_turn(
        self,
        user_input: Union[str, asyncio.Future],
        bot_response: Union[str, asyncio.Future],
        intent: Optional[Union[str, asyncio.Future]] = None,
        confidence: Optional[Union[float, asyncio.Future]] = None,
        model_used: Optional[Union[str, asyncio.Future]] = None,
        user_id: Optional[Union[str, asyncio.Future]] = None,
        session_id: Optional[Union[str, asyncio.Future]] = None,
    ) -> None:
        """Add one turn, awaiting any coroutine inputs."""
        try:
            # Await any coroutines
            user_input = await user_input if asyncio.iscoroutine(user_input) else user_input
            bot_response = await bot_response if asyncio.iscoroutine(bot_response) else bot_response
            intent = await intent if asyncio.iscoroutine(intent) else intent
            confidence = await confidence if asyncio.iscoroutine(confidence) else confidence
            model_used = await model_used if asyncio.iscoroutine(model_used) else model_used
            user_id = await user_id if asyncio.iscoroutine(user_id) else user_id
            session_id = await session_id if asyncio.iscoroutine(session_id) else session_id

            # Validate types
            if not isinstance(user_input, str) or not isinstance(bot_response, str):
                self.logger.warning(f"Invalid turn: user_input={type(user_input)}, bot_response={type(bot_response)}")
                return
            if intent is not None and not isinstance(intent, str):
                self.logger.warning(f"Invalid intent type: {type(intent)}")
                intent = None
            if confidence is not None and not isinstance(confidence, (int, float)):
                self.logger.warning(f"Invalid confidence type: {type(confidence)}")
                confidence = None
            if model_used is not None and not isinstance(model_used, str):
                self.logger.warning(f"Invalid model_used type: {type(model_used)}")
                model_used = None
            if user_id is not None and not isinstance(user_id, str):
                self.logger.warning(f"Invalid user_id type: {type(user_id)}")
                user_id = None
            if session_id is not None and not isinstance(session_id, str):
                self.logger.warning(f"Invalid session_id type: {type(session_id)}")
                session_id = None

            if not user_input or not bot_response:
                self.logger.warning("Refusing to add turn with empty input/response.")
                return

            if len(user_input) > 20_000 or len(bot_response) > 100_000:
                self.logger.warning("Turn exceeds maximum allowed size; dropping.")
                return

            turn = ConversationTurn(
                user=user_input,
                bot=bot_response,
                timestamp=datetime.utcnow().isoformat(),
                intent=intent,
                confidence=confidence,
                model_used=model_used,
                user_id=user_id,
                session_id=session_id,
            )

            self.conversation_history.append(turn)
            self.pending_writes.append(turn)
            self.logger.debug(f"Added turn: user={user_input[:50]}..., bot={bot_response[:50]}...")

            await self._check_memory_usage()

            if len(self.pending_writes) >= self.write_buffer_size:
                self._flush_event.set()
        except Exception as e:
            self.logger.error(f"add_turn failed: {e}")

    def get_recent_messages(self, limit: int = 10) -> List[Dict[str, str]]:
        try:
            limit = max(1, min(limit, len(self.conversation_history)))
            recent = list(self.conversation_history)[-limit:]
            return [{"user": t.user, "bot": t.bot} for t in recent]
        except Exception as e:
            self.logger.error(f"get_recent_messages failed: {e}")
            return []

    def get_user_conversation(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 20,
        format: str = "pairs",
    ) -> List[Dict[str, str]]:
        try:
            if format not in {"pairs", "messages"}:
                format = "pairs"
            limit = max(1, min(limit, 1000))

            items: List[ConversationTurn] = list(self.conversation_history)
            if user_id is not None:
                items = [t for t in items if t.user_id == user_id]
            if session_id is not None:
                items = [t for t in items if t.session_id == session_id]

            items = items[-limit:] if items else []

            if format == "messages":
                msgs: List[Dict[str, str]] = []
                for t in items:
                    msgs.append({"role": "user", "content": t.user})
                    msgs.append({"role": "assistant", "content": t.bot})
                return msgs
            else:
                return [{"user": t.user, "bot": t.bot} for t in items]
        except Exception as e:
            self.logger.error(f"get_user_conversation failed: {e}")
            return []

    async def set_user_conversation(
        self,
        turns: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        overwrite: bool = True,
    ) -> None:
        try:
            if not isinstance(turns, list):
                self.logger.error("Turns must be a list.")
                return
            if len(turns) > 1000:
                self.logger.warning(f"Limiting turns from {len(turns)} to 1000.")
                turns = turns[-1000:]

            def _valid(t: Dict[str, Any]) -> bool:
                return isinstance(t, dict) and ("user" in t) and ("bot" in t)

            filtered = [t for t in turns if _valid(t)]
            if not filtered:
                self.logger.warning("No valid turns provided to set_user_conversation.")
                return

            if overwrite:
                if user_id is None and session_id is None:
                    self.conversation_history.clear()
                else:
                    def keep(t: ConversationTurn) -> bool:
                        mu = True if user_id is None else (t.user_id != user_id)
                        ms = True if session_id is None else (t.session_id != session_id)
                        return mu or ms
                    self.conversation_history = deque(
                        [t for t in self.conversation_history if keep(t)],
                        maxlen=self.conversation_history.maxlen,
                    )

            now = datetime.utcnow().isoformat()
            for t in filtered:
                ct = ConversationTurn(
                    user=str(t.get("user", "")),
                    bot=str(t.get("bot", "")),
                    timestamp=str(t.get("timestamp", now)),
                    intent=t.get("intent"),
                    confidence=t.get("confidence"),
                    model_used=t.get("model_used"),
                    user_id=user_id if user_id is not None else t.get("user_id"),
                    session_id=session_id if session_id is not None else t.get("session_id"),
                )
                self.conversation_history.append(ct)
                self.pending_writes.append(ct)

            if len(self.pending_writes) >= self.write_buffer_size:
                self._flush_event.set()
        except Exception as e:
            self.logger.error(f"set_user_conversation failed: {e}")

    def clear_memory(self) -> None:
        try:
            self.conversation_history.clear()
            self.pending_writes.clear()
            self._flush_event.clear()
            self.logger.info("In-memory conversation cleared.")
        except Exception as e:
            self.logger.error(f"clear_memory failed: {e}")

    # ---------- Pruning & Utilities ----------
    def prune_older_than(self, days: int) -> int:
        """Remove turns older than N days. Returns number of removed turns."""
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            before = len(self.conversation_history)
            kept: List[ConversationTurn] = []
            for t in self.conversation_history:
                try:
                    ts = datetime.fromisoformat(t.timestamp.replace("Z", "+00:00"))
                except Exception:
                    continue
                if ts >= cutoff:
                    kept.append(t)
            self.conversation_history = deque(kept, maxlen=self.conversation_history.maxlen)
            removed = before - len(self.conversation_history)
            if removed:
                self._flush_event.set()
            return removed
        except Exception as e:
            self.logger.error(f"prune_older_than failed: {e}")
            return 0

    async def export_json(self, path: Union[str, Path]) -> bool:
        """Write full snapshot to a custom path (for backups/exports)."""
        try:
            target = self._safe_path(str(path))
            payload = {
                "schema": {"version": SCHEMA_VERSION, "exported_at": datetime.utcnow().isoformat()},
                "conversation": [t.to_dict() for t in self.conversation_history],
            }
            target.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(target, "w", encoding=ENCODING) as f:
                await f.write(json.dumps(payload, ensure_ascii=False, indent=2))
            return True
        except Exception as e:
            self.logger.error(f"export_json failed: {e}")
            return False

    async def import_json(self, path: Union[str, Path], append: bool = True) -> bool:
        """Load a snapshot from a path (merge or replace)."""
        try:
            src = self._safe_path(str(path))
            if not src.exists():
                self.logger.error(f"import_json: file not found {src}")
                return False

            async with aiofiles.open(src, "r", encoding=ENCODING) as f:
                raw = await f.read()

            data = json.loads(raw)
            if not isinstance(data, dict) or "conversation" not in data:
                self.logger.error("import_json: invalid structure")
                return False

            turns = []
            for t in data["conversation"]:
                try:
                    turns.append(ConversationTurn(**t))
                except Exception:
                    continue

            if not append:
                self.conversation_history.clear()
            for ct in turns:
                self.conversation_history.append(ct)
            self._flush_event.set()
            return True
        except Exception as e:
            self.logger.error(f"import_json failed: {e}")
            return False

    # ---------- Cache ----------
    def _make_cache_key(self, key: str, namespace: Optional[str]) -> str:
        return f"{namespace}::{key}" if namespace else key

    def _evict_expired_locked(self) -> None:
        now = time.time()
        to_del = [k for k, v in self._cache.items() if v.expires_at is not None and v.expires_at <= now]
        for k in to_del:
            self._cache.pop(k, None)

        if len(self._cache) > self.max_cache_size:
            drop_count = len(self._cache) - self.max_cache_size
            for k in sorted(self._cache, key=lambda x: self._cache[x].access_count)[:drop_count]:
                self._cache.pop(k, None)

    async def set_cache(
        self,
        key: str,
        value: Any,
        ttl: Optional[Union[int, float]] = None,
        namespace: Optional[str] = None,
    ) -> None:
        try:
            full = self._make_cache_key(key, namespace)
            exp = None if ttl is None else (time.time() + float(ttl))
            async with self._cache_lock:
                self._evict_expired_locked()
                if len(self._cache) >= self.max_cache_size:
                    lk = min(self._cache, key=lambda k: self._cache[k].access_count)
                    self._cache.pop(lk, None)
                self._cache[full] = CacheEntry(value=value, expires_at=exp, access_count=1)
        except Exception as e:
            self.logger.error(f"set_cache failed: {e}")

    async def get_cache(self, key: str, default: Any = None, namespace: Optional[str] = None) -> Any:
        try:
            full = self._make_cache_key(key, namespace)
            async with self._cache_lock:
                self._evict_expired_locked()
                entry = self._cache.get(full)
                if entry is None:
                    return default
                if entry.expires_at is not None and entry.expires_at <= time.time():
                    self._cache.pop(full, None)
                    return default
                entry.access_count += 1
                return entry.value
        except Exception as e:
            self.logger.error(f"get_cache failed: {e}")
            return default

    async def delete_cache(self, key: str, namespace: Optional[str] = None) -> bool:
        try:
            full = self._make_cache_key(key, namespace)
            async with self._cache_lock:
                return self._cache.pop(full, None) is not None
        except Exception as e:
            self.logger.error(f"delete_cache failed: {e}")
            return False

    async def clear_cache(self, namespace: Optional[str] = None) -> None:
        try:
            async with self._cache_lock:
                if namespace is None:
                    self._cache.clear()
                else:
                    pref = f"{namespace}::"
                    for k in [k for k in self._cache.keys() if k.startswith(pref)]:
                        self._cache.pop(k, None)
        except Exception as e:
            self.logger.error(f"clear_cache failed: {e}")

    # ---------- Stats & Cleanup ----------
    def get_stats(self) -> Dict[str, Any]:
        try:
            conv_bytes = sum(self._approx_bytes_turn(t) for t in self.conversation_history)
            return {
                "schema_version": SCHEMA_VERSION,
                "total_conversations": len(self.conversation_history),
                "memory_usage_mb": conv_bytes / 1024 / 1024,
                "cache_size": len(self._cache),
                "pending_writes": len(self.pending_writes),
                "flush_count": self._flush_count,
                "load_count": self._load_count,
                "error_count": self._error_count,
            }
        except Exception as e:
            self.logger.error(f"get_stats failed: {e}")
            return {}

    async def cleanup(self) -> None:
        """Flush buffers, stop flusher, and compact cache."""
        try:
            if self._flush_task and not self._flush_task.done():
                self._flush_task.cancel()
                try:
                    await self._flush_task
                except asyncio.CancelledError:
                    pass

            if self.pending_writes:
                await self.flush_to_disk()

            async with self._cache_lock:
                self._evict_expired_locked()

            self.logger.info("MemoryManager cleanup completed.")
        except Exception as e:
            self.logger.error(f"cleanup failed: {e}")
