from __future__ import annotations

import os
from typing import List, Tuple

try:
    import openai
except ImportError:  # pragma: no cover - optional dependency
    openai = None
try:
    from transformers import pipeline
except ImportError:  # pragma: no cover - optional dependency
    pipeline = None


def format_context(chunks: List[Tuple[float, str, str, str]]) -> str:
    lines = []
    for score, pdf_id, section_id, text in chunks:
        lines.append(f"[{pdf_id}:{section_id}] (score={score:.3f}) {text}")
    return "\n".join(lines)


def _truncate_chunks(
    chunks: List[Tuple[float, str, str, str]], max_chunks: int = 4, per_chunk_chars: int = 320, total_chars: int = 1600
) -> List[Tuple[float, str, str, str]]:
    truncated: List[Tuple[float, str, str, str]] = []
    used = 0
    for score, pdf_id, section_id, text in chunks[:max_chunks]:
        snippet = text[:per_chunk_chars]
        if used + len(snippet) > total_chars:
            break
        truncated.append((score, pdf_id, section_id, snippet))
        used += len(snippet)
    return truncated


def synthesize_answer(
    query: str,
    chunks: List[Tuple[float, str, str, str]],
    provider: str,
    model: str,
    max_tokens: int = 512,
) -> str:
    # Limit context for small models (e.g., flan-t5-small) to avoid overlength inputs.
    truncated_chunks = _truncate_chunks(chunks, max_chunks=4, per_chunk_chars=320, total_chars=1600)

    if provider == "openai" and openai is not None and os.environ.get("OPENAI_API_KEY"):
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        context_block = format_context(truncated_chunks)
        messages = [
            {
                "role": "system",
                "content": "You are a study assistant for math proofs. Use only the provided evidence. Cite as [pdf:section]. State if key steps are missing.",
            },
            {"role": "user", "content": f"Query: {query}\n\nEvidence:\n{context_block}"},
        ]
        resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=0.2)
        return resp.choices[0].message.content
    if provider == "hf" and pipeline is not None:
        context_block = format_context(truncated_chunks)
        prompt = (
            "You are a study assistant for math proofs. Use only the provided evidence. "
            "Answer concisely and include the formal definition if asked. "
            "Cite each statement as [pdf:section]. If evidence is missing, say so before answering.\n"
            f"Query: {query}\nEvidence:\n{context_block}\nAnswer:"
        )
        generator = pipeline("text2text-generation", model=model)
        out = generator(prompt, max_new_tokens=max_tokens, do_sample=False, truncation=True)
        return out[0]["generated_text"]
    # Fallback: deterministic summary
    bullet_lines = [f"- [{pdf}:{section}] {text[:200]}..." for _, pdf, section, text in truncated_chunks]
    return "Evidence summary:\n" + "\n".join(bullet_lines)

