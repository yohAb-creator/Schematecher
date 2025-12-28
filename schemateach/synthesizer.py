from __future__ import annotations

import os
import re
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
    for _, pdf_id, section_id, text in chunks:
        lines.append(f"[{pdf_id}:{section_id}] {text}")
    return "\n".join(lines)


def _select_chunks_for_query(query: str, chunks: List[Tuple[float, str, str, str]]) -> List[Tuple[float, str, str, str]]:
    q = query.lower()
    needs_def = any(phrase in q for phrase in ("what is", "define", "definition", "what's"))
    if not needs_def:
        return chunks
    preferred = []
    for item in chunks:
        text = item[3].lower()
        if "definition" in text or "is called" in text or "we call" in text:
            preferred.append(item)
    return preferred or chunks


def _focus_chunks_by_keywords(query: str, chunks: List[Tuple[float, str, str, str]]) -> List[Tuple[float, str, str, str]]:
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "for",
        "with",
        "why",
        "explain",
        "what",
        "is",
        "are",
        "does",
        "do",
        "definition",
        "define",
        "equivalence",
    }
    tokens = [t for t in re.findall(r"[a-zA-Z0-9\\-]+", query.lower()) if t not in stop]
    if not tokens:
        return chunks
    scored = []
    for idx, item in enumerate(chunks):
        text = item[3].lower()
        hits = sum(1 for t in tokens if t in text)
        scored.append((hits, idx, item))
    max_hits = max(hit for hit, _, _ in scored) if scored else 0
    if max_hits == 0:
        return chunks
    filtered = [(hit, idx, item) for hit, idx, item in scored if hit > 0]
    # Sort by hit count, keep stable order for ties.
    filtered.sort(key=lambda row: (-row[0], row[1]))
    return [item for _, _, item in filtered]


def _extract_definition(chunks: List[Tuple[float, str, str, str]], max_chars: int = 900) -> str | None:
    stop_re = re.compile(
        r"\\b(For example|Another example|Example\\s+\\d|Lemma\\s+\\d|Theorem\\s+\\d|Proposition\\s+\\d|Corollary\\s+\\d|Remark\\s+\\d|Exercise\\s+\\d|Proof)\\b",
        re.I,
    )
    for _, pdf_id, section_id, text in chunks:
        match = re.search(r"definition\\s+\\d+(?:\\.\\d+)*", text, re.I)
        if not match:
            continue
        snippet = text[match.start() :]
        stop = stop_re.search(snippet)
        if stop:
            snippet = snippet[: stop.start()]
        snippet = " ".join(snippet.split())
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rsplit(" ", 1)[0] + "..."
        return f"{snippet} [{pdf_id}:{section_id}]"
    return None


def _extract_definition_sentence(query: str, chunks: List[Tuple[float, str, str, str]]) -> str | None:
    tokens = [t for t in re.findall(r"[a-zA-Z0-9\\-]+", query.lower()) if t not in {"what", "is", "a", "an", "the", "define", "definition", "of"}]
    if not tokens:
        return None
    for _, pdf_id, section_id, text in chunks:
        sentences = re.split(r"(?<=[\\.\\!\\?])\\s+", text)
        for sentence in sentences:
            s = sentence.strip()
            if not s:
                continue
            if not re.search(r"\\b(is|are)\\b", s, re.I):
                continue
            if not re.search(r"\\b(A|An|The)\\b", s, re.I):
                continue
            s_lower = s.lower()
            if any(tok in s_lower for tok in tokens):
                return f"{s} [{pdf_id}:{section_id}]"
    return None


def _normalize_hf_model(model: str) -> str:
    if model.startswith("flan-t5-"):
        return f"google/{model}"
    return model


def _sentence_truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    sentences = re.split(r"(?<=[\\.\\!\\?])\\s+", text.strip())
    if len(sentences) <= 1:
        return text[:max_chars].rsplit(" ", 1)[0].rstrip() + "..."
    out = []
    used = 0
    for sentence in sentences:
        if not sentence:
            continue
        add_len = len(sentence) + (1 if out else 0)
        if used + add_len > max_chars:
            break
        out.append(sentence)
        used += add_len
    if not out:
        return text[:max_chars].rsplit(" ", 1)[0].rstrip() + "..."
    return " ".join(out).rstrip() + "..."


def _truncate_chunks(
    chunks: List[Tuple[float, str, str, str]], max_chunks: int, per_chunk_chars: int, total_chars: int
) -> List[Tuple[float, str, str, str]]:
    truncated: List[Tuple[float, str, str, str]] = []
    used = 0
    for score, pdf_id, section_id, text in chunks[:max_chunks]:
        snippet = _sentence_truncate(text, per_chunk_chars)
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
    reasoning_paths: List[List[str]] | None = None,
) -> str:
    # Limit context based on model size to avoid overlength inputs.
    filtered = _select_chunks_for_query(query, chunks)
    if any(phrase in query.lower() for phrase in ("why", "explain")):
        filtered = _focus_chunks_by_keywords(query, filtered)
    model_lower = model.lower()
    if "flan-t5-large" in model_lower:
        max_chunks, per_chunk_chars, total_chars = 4, 520, 2400
    elif "flan-t5-base" in model_lower:
        max_chunks, per_chunk_chars, total_chars = 4, 420, 2000
    else:
        max_chunks, per_chunk_chars, total_chars = 3, 320, 1400
    truncated_chunks = _truncate_chunks(filtered, max_chunks=max_chunks, per_chunk_chars=per_chunk_chars, total_chars=total_chars)
    if any(phrase in query.lower() for phrase in ("what is", "define", "definition", "what's")):
        extracted = _extract_definition(filtered)
        if extracted:
            return f"{extracted}\n\nSummary: This is the formal definition requested."
        sentence_def = _extract_definition_sentence(query, filtered)
        if sentence_def:
            return f"{sentence_def}\n\nSummary: This is a concise definition extracted from the text."
        if filtered:
            pdf_id, section_id, text = filtered[0][1], filtered[0][2], filtered[0][3]
            snippet = " ".join(text.split())[:400].rsplit(" ", 1)[0] + "..."
            return f"Closest evidence excerpt: {snippet} [{pdf_id}:{section_id}]"

    paths_block = ""
    if reasoning_paths:
        lines = [" -> ".join(path) for path in reasoning_paths]
        paths_block = "\nReasoning paths (prereq chains):\n" + "\n".join(lines)

    if provider == "openai" and openai is not None and os.environ.get("OPENAI_API_KEY"):
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        context_block = format_context(truncated_chunks)
        messages = [
            {
                "role": "system",
                "content": "You are a study assistant for math proofs. Use only the provided evidence. Cite as [pdf:section]. State if key steps are missing.",
            },
            {"role": "user", "content": f"Query: {query}\n\nEvidence:\n{context_block}{paths_block}"},
        ]
        resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=0.2)
        return resp.choices[0].message.content
    if provider == "hf" and pipeline is not None:
        context_block = format_context(truncated_chunks)
        prompt = (
            "You are a study assistant for math proofs. Use only the provided evidence. "
            "Provide a complete, self-contained answer that does not trail off. "
            "Include the formal definition if asked. End with a short summary sentence. "
            "Cite each statement as [pdf:section] at the end of the sentence. "
            "Do not copy the evidence verbatim; paraphrase it. "
            "If evidence is missing, say so before answering.\n"
            f"Query: {query}\nEvidence:\n{context_block}{paths_block}\nAnswer:"
        )
        generator = pipeline("text2text-generation", model=_normalize_hf_model(model))
        out = generator(
            prompt,
            max_new_tokens=max_tokens,
            min_new_tokens=min(64, max_tokens // 4),
            do_sample=False,
            truncation=True,
            no_repeat_ngram_size=4,
            repetition_penalty=1.15,
            num_beams=2,
            early_stopping=False,
        )
        return out[0]["generated_text"]
    # Fallback: deterministic summary
    bullet_lines = [f"- [{pdf}:{section}] {text[:200]}..." for _, pdf, section, text in truncated_chunks]
    return "Evidence summary:\n" + "\n".join(bullet_lines)
