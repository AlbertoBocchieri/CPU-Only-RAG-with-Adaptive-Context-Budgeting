from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time

_LLAMA_CACHE: dict[str, object] = {}
DEFAULT_DIRECT_TEMPLATE = (
    "Context:\n"
    "{context}\n\n"
    "Based on the context above, answer this question with just the answer, no explanation:\n\n"
    "Q: {question}\n"
    "A:"
)
_ANSWER_PREFIX_RE = re.compile(
    r"^\s*(answer\s*:|the answer is\s*:?)\s*",
    flags=re.IGNORECASE,
)
_CONSERVATIVE_LEAD_RE = re.compile(
    r"^\s*(according to|based on|from)\s+the\s+(context|passage|text)[,:]?\s*",
    flags=re.IGNORECASE,
)
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>\s*", flags=re.IGNORECASE | re.DOTALL)
_THINK_TAG_RE = re.compile(r"</?think\b[^>]*>", flags=re.IGNORECASE)
_TRAILING_PAREN_RE = re.compile(r"\s*\(.*?\)\s*$")
_TRAILING_DOT_RE = re.compile(r"\.\s*$")
_SPACES_RE = re.compile(r"\s+")

FEW_SHOT_EXTRACTIVE_SYSTEM = (
    "Answer each question using ONLY the provided context. "
    "Give ONLY the answer string, with no explanation.\n\n"
    "Example 1\n"
    "Context:\n"
    "[1] The Eiffel Tower was built in 1889 for the World's Fair in Paris.\n"
    "[2] Gustave Eiffel's company designed and built the tower.\n"
    "Q: Who designed the Eiffel Tower?\n"
    "A: Gustave Eiffel\n\n"
    "Example 2\n"
    "Context:\n"
    "[1] The Amazon River flows through Brazil and Peru.\n"
    "[2] Recent studies suggest the Amazon may be 6,992 km long.\n"
    "Q: How long is the Amazon River?\n"
    "A: 6,992 km"
)


@dataclass(slots=True)
class GenerationResult:
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    t_prompt_build_ms: float = 0.0
    t_llm_total_ms: float = 0.0
    ttft_ms: float = 0.0
    t_prefill_ms: float = 0.0
    t_decode_total_ms: float = 0.0
    tokens_per_second_decode: float = 0.0
    tokens_per_second_prefill: float = 0.0


class LlamaCppGenerator:
    def __init__(
        self,
        gguf_path: str,
        n_ctx: int,
        n_threads: int,
        n_batch: int,
        n_threads_batch: int | None = None,
        n_ubatch: int | None = None,
        prefix_cache_enabled: bool = False,
        prefix_cache_backend: str = "ram",
        prefix_cache_capacity_mb: int = 256,
        prefix_cache_dir: str = "cache/llama_prompt_cache",
    ):
        from llama_cpp import Llama

        key = f"{gguf_path}|{n_ctx}|{n_threads}|{n_batch}|{n_threads_batch}|{n_ubatch}"
        if key in _LLAMA_CACHE:
            self.llm = _LLAMA_CACHE[key]
        else:
            kwargs = {
                "model_path": gguf_path,
                "n_ctx": n_ctx,
                "n_threads": n_threads,
                "n_batch": n_batch,
                "n_gpu_layers": 0,
                "chat_format": "chatml",
                "verbose": False,
            }
            if n_threads_batch is not None:
                kwargs["n_threads_batch"] = int(n_threads_batch)
            if n_ubatch is not None:
                kwargs["n_ubatch"] = int(n_ubatch)
            self.llm = Llama(
                **kwargs,
            )
            _LLAMA_CACHE[key] = self.llm

        self.prefix_cache_enabled = False
        self.prefix_cache_backend = str(prefix_cache_backend or "ram").lower()
        self.prefix_cache_capacity_mb = int(prefix_cache_capacity_mb)
        self._configure_prefix_cache(
            enabled=bool(prefix_cache_enabled),
            backend=self.prefix_cache_backend,
            capacity_mb=self.prefix_cache_capacity_mb,
            cache_dir=str(prefix_cache_dir or "cache/llama_prompt_cache"),
        )

    def _configure_prefix_cache(
        self,
        enabled: bool,
        backend: str,
        capacity_mb: int,
        cache_dir: str,
    ) -> None:
        if not enabled:
            return
        try:
            from llama_cpp import LlamaDiskCache, LlamaRAMCache
        except Exception:
            return

        cap_bytes = max(64, int(capacity_mb)) * 1024 * 1024
        backend_norm = str(backend or "ram").strip().lower()
        try:
            if backend_norm == "disk":
                cache_path = Path(cache_dir)
                cache_path.mkdir(parents=True, exist_ok=True)
                cache_obj = LlamaDiskCache(
                    capacity_bytes=cap_bytes,
                    cache_dir=str(cache_path),
                )
            else:
                cache_obj = LlamaRAMCache(capacity_bytes=cap_bytes)
            self.llm.set_cache(cache_obj)
            self.prefix_cache_enabled = True
        except Exception:
            self.prefix_cache_enabled = False

    @staticmethod
    def _build_context_block(contexts: list[str]) -> str:
        rows = []
        for i, c in enumerate(contexts, start=1):
            rows.append(f"[{i}] {c.strip()}")
        return "\n\n".join(rows)

    @staticmethod
    def _postprocess_answer(answer: str, mode: str = "basic") -> str:
        text = str(answer or "").strip()
        if not text:
            return ""
        # Qwen3 may emit explicit thinking blocks even in no-think mode; strip them
        # before extracting the final short answer.
        text = _THINK_BLOCK_RE.sub("", text)
        text = _THINK_TAG_RE.sub("", text).strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        first_line = lines[0] if lines else text.strip()
        cleaned = _ANSWER_PREFIX_RE.sub("", first_line).strip()
        mode_norm = str(mode or "basic").strip().lower()
        if mode_norm != "conservative":
            return cleaned
        cleaned = _CONSERVATIVE_LEAD_RE.sub("", cleaned).strip()
        cleaned = _TRAILING_PAREN_RE.sub("", cleaned).strip()
        cleaned = cleaned.strip(" \t\"'")
        cleaned = _TRAILING_DOT_RE.sub("", cleaned).strip()
        cleaned = _SPACES_RE.sub(" ", cleaned).strip()
        return cleaned

    def generate(
        self,
        question: str,
        contexts: list[str],
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        repeat_penalty: float,
        prompt_mode: str = "rag_strict",
        direct_template: str = "",
        answer_postprocess_mode: str = "basic",
        enable_stream_timing: bool = True,
    ) -> GenerationResult:
        t_prompt0 = time.perf_counter()
        context_block = self._build_context_block(contexts)
        if prompt_mode == "direct":
            template = direct_template.strip() or DEFAULT_DIRECT_TEMPLATE
            user_prompt = template.format(context=context_block, question=question)
            messages = [{"role": "user", "content": user_prompt}]
        elif prompt_mode == "answer_only":
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Answer with only the final answer string. "
                        "No explanation, no reasoning, no extra words."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context_block}\n\n"
                        f"Question: {question}\n"
                        "Output only the answer."
                    ),
                },
            ]
        elif prompt_mode == "few_shot_extractive":
            messages = [
                {
                    "role": "system",
                    "content": FEW_SHOT_EXTRACTIVE_SYSTEM,
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context_block}\n\n"
                        f"Q: {question}\n"
                        "A:"
                    ),
                },
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a rigorous QA assistant for RAG evaluation. "
                        "Use only the provided context. If the answer is not in context, reply exactly: Non lo so. "
                        "Keep the answer concise and factual."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context_block}\n\n"
                        f"Question: {question}\n"
                        "Answer in the same language as the question."
                    ),
                },
            ]
        t_prompt_build_ms = (time.perf_counter() - t_prompt0) * 1000.0

        answer = ""
        usage: dict[str, int] = {}
        t_llm0 = time.perf_counter()
        ttft_ms = 0.0
        used_streaming = False

        if enable_stream_timing:
            try:
                used_streaming = True
                pieces: list[str] = []
                first_token_ts: float | None = None
                stream = self.llm.create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                    repeat_penalty=repeat_penalty,
                    stream=True,
                )
                for chunk in stream:
                    if "usage" in chunk and isinstance(chunk["usage"], dict):
                        usage = chunk["usage"]
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    choice0 = choices[0]
                    delta = choice0.get("delta", {})
                    text_piece = ""
                    if isinstance(delta, dict):
                        text_piece = str(delta.get("content") or "")
                    if not text_piece:
                        text_piece = str(choice0.get("text") or "")
                    if text_piece:
                        if first_token_ts is None:
                            first_token_ts = time.perf_counter()
                        pieces.append(text_piece)
                answer = "".join(pieces).strip()
                if first_token_ts is not None:
                    ttft_ms = (first_token_ts - t_llm0) * 1000.0
            except Exception:
                used_streaming = False

        if not used_streaming or not answer:
            out = self.llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                repeat_penalty=repeat_penalty,
            )
            answer = out["choices"][0]["message"]["content"].strip()
            usage = out.get("usage", {})

        t_llm_total_ms = (time.perf_counter() - t_llm0) * 1000.0
        if ttft_ms <= 0.0:
            ttft_ms = t_llm_total_ms
        t_prefill_ms = ttft_ms
        t_decode_total_ms = max(0.0, t_llm_total_ms - ttft_ms)

        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        if prompt_tokens <= 0:
            try:
                prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                prompt_tokens = int(len(self.llm.tokenize(prompt_text.encode("utf-8"))))
            except Exception:
                prompt_tokens = int(max(0, len(context_block.split()) + len(question.split())))
        if completion_tokens <= 0 and answer:
            try:
                completion_tokens = int(len(self.llm.tokenize(answer.encode("utf-8"))))
            except Exception:
                completion_tokens = int(max(1, len(answer.split())))
        if prompt_mode in {"direct", "answer_only", "few_shot_extractive"}:
            answer = self._postprocess_answer(answer, mode=answer_postprocess_mode)
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))

        tokens_per_second_decode = (
            float(completion_tokens / max(1e-9, (t_decode_total_ms / 1000.0)))
            if completion_tokens > 0 and t_decode_total_ms > 0.0
            else 0.0
        )
        tokens_per_second_prefill = (
            float(prompt_tokens / max(1e-9, (t_prefill_ms / 1000.0)))
            if prompt_tokens > 0 and t_prefill_ms > 0.0
            else 0.0
        )

        return GenerationResult(
            answer=answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            t_prompt_build_ms=float(t_prompt_build_ms),
            t_llm_total_ms=float(t_llm_total_ms),
            ttft_ms=float(ttft_ms),
            t_prefill_ms=float(t_prefill_ms),
            t_decode_total_ms=float(t_decode_total_ms),
            tokens_per_second_decode=float(tokens_per_second_decode),
            tokens_per_second_prefill=float(tokens_per_second_prefill),
        )
