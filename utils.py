"""
Utility layer for AI Study Assistant (Groq Edition)
MODIFICATIONS vs baseline:
    • PDF text is NEVER sent whole to the LLM
    • Chunked into ~3800-char blocks with 400-char overlap
    • Sentence-Transformer embeddings for semantic retrieval (with keyword fallback)
    • assemble_context() hard-caps characters to prevent 413 Request Entity Too Large
    • Explicit 413 / 429 / 404 / 401 handling in Groq wrapper
"""

import re
import json
import time
import random
from typing import List, Any, Dict

import numpy as np
import PyPDF2

# NEW: sentence-transformers is optional; keyword fallback always works
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

from groq import Groq


# ── Constants ─────────────────────────────────────────────────────────────
CHUNK_SIZE = 3800          # characters per chunk  (~3-4k as requested)
CHUNK_OVERLAP = 400        # characters to overlap between chunks
MAX_CONTEXT_CHARS = 20000  # hard ceiling sent to Groq (~5k tokens ≪ 12k limit)

# ── PDF handling ────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_file) -> str:
    """Safely extract text from an uploaded PDF (BytesIO)."""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        parts = []
        for page in reader.pages:
            try:
                txt = page.extract_text()
                if txt:
                    parts.append(txt)
            except Exception:
                continue
        return "\n".join(parts).strip()
    except Exception:
        return ""


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    NEW: Split text into overlapping chunks.
    Prevents sending the full PDF to the model.
    """
    if not text:
        return []
    chunks = []
    start = 0
    step = chunk_size - overlap
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        if end == length:
            break
        start += step
    return chunks


# ── Embedding & semantic search ─────────────────────────────────────────

def _load_embedding_model() -> "SentenceTransformer | None":
    """NEW: lazy-load once per process."""
    if not _ST_AVAILABLE:
        raise RuntimeError("sentence-transformers is not installed.")
    if not hasattr(_load_embedding_model, "_model"):
        _load_embedding_model._model = SentenceTransformer('all-MiniLM-L6-v2')
    return _load_embedding_model._model


def build_chunk_embeddings(chunks: List[str]):
    """
    NEW: Compute sentence embeddings for every chunk.
    Returns numpy array of shape (N, 384).
    """
    model = _load_embedding_model()
    return model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)


def get_relevant_chunks(
    query: str,
    chunks: List[str],
    top_k: int = 3,
    chunk_embeddings=None,
) -> List[str]:
    """
    NEW: Retrieve top-k most relevant chunks.
    1) Semantic cosine similarity if embeddings available
    2) Keyword overlap otherwise (zero-dependency fallback)
    """
    if not chunks:
        return []

    # -- semantic route ---------------------------------------------------
    if _ST_AVAILABLE and chunk_embeddings is not None and len(chunk_embeddings) == len(chunks):
        try:
            model = _load_embedding_model()
            q_embed = model.encode([query], convert_to_numpy=True)

            # cosine similarity via pure numpy
            norm_q = np.linalg.norm(q_embed, axis=1)[:, None]
            norm_c = np.linalg.norm(chunk_embeddings, axis=1)[None, :]
            denom = norm_q * norm_c
            denom[denom == 0] = 1e-10
            similarities = np.dot(q_embed, chunk_embeddings.T) / denom
            similarities = similarities[0]

            top_indices = np.argsort(similarities)[::-1][:top_k]
            return [chunks[int(i)] for i in top_indices]
        except Exception:
            pass  # graceful fallback

    # -- keyword fallback -------------------------------------------------
    q = query.lower()
    scored = []
    for idx, chunk in enumerate(chunks):
        score = 0
        c = chunk.lower()
        if q in c:
            score += 10
        for w in q.split():
            if len(w) > 2:
                score += c.count(w)
        scored.append((score, idx))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunks[i] for _, i in scored[:top_k]]


# ── NEW: Token-safe context assembly ────────────────────────────────────

def assemble_context(chunks: List[str], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    NEW: Join chunks with a separator, stopping before max_chars is exceeded.
    This is the single choke-point that prevents 413 errors.
    """
    if not chunks:
        return "No context provided."
    parts = []
    used = 0
    sep = "\n\n---\n\n"
    sep_len = len(sep)
    for c in chunks:
        if parts and (used + len(c) + sep_len) > max_chars:
            break
        parts.append(c)
        used += len(c) + sep_len
    return sep.join(parts) if parts else "No context provided."


# ── Groq wrapper with explicit 413 / 429 / 404 / 401 handling ───────────

def get_ai_response(
    prompt: str,
    api_key: str,
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    max_retries: int = 2,
) -> str:
    """
    MODIFIED: Call Groq with retries and explicit token-overflow handling.
    """
    for attempt in range(max_retries + 1):
        try:
            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": ("You are a precise academic assistant. "
                                    "Follow formatting instructions exactly."),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content

        except Exception as e:
            err = str(e).lower()
            if "413" in err or "too large" in err or "request too large" in err:
                return (
                    "Error 413: Request exceeded the model's token limit.\n\n"
                    "The app automatically truncates PDF context, but extremely "
                    "long prompts may still overflow. Try asking a shorter question "
                    "or splitting your material into smaller PDFs."
                )
            elif "401" in err or "unauthorized" in err or "invalid api key" in err:
                return "Error: Invalid Groq API Key. Verify at https://console.groq.com/keys"
            elif "429" in err or "rate limit" in err:
                if attempt < max_retries:
                    time.sleep(2 ** attempt + random.uniform(0, 1))
                    continue
                return (
                    "Error 429: Groq rate limit hit. Free tier is generous but has strict RPM limits. "
                    "Wait a few seconds and retry."
                )
            elif "404" in err or "not found" in err:
                return f"Error: Model '{model_name}' unavailable on Groq. Choose another model."
            else:
                if attempt == max_retries:
                    return f"Error: {str(e)}"
                time.sleep(1)
    return "Error: Unexpected failure after retries."


# ── Prompt builders (all now auto-truncate via assemble_context) ────────

def build_qa_prompt(question: str, context_chunks: List[str]) -> str:
    context = assemble_context(context_chunks, max_chars=MAX_CONTEXT_CHARS)
    return f"""Answer the user's question STRICTLY based on the uploaded material below.
If the material does not contain the answer, state: "The uploaded material does not contain specific information about this question." Then provide a brief general answer if possible.

UPLOADED MATERIAL:
{context}

USER QUESTION:
{question}

ANSWER:"""


def build_summary_prompt(text_chunks: List[str], summary_type: str = "detailed") -> str:
    context = assemble_context(text_chunks, max_chars=MAX_CONTEXT_CHARS)
    if summary_type == "short":
        instruction = "Provide a short summary (2-3 paragraphs) capturing the main ideas only."
    elif summary_type == "exam":
        instruction = (
            "Provide an exam revision summary. Include key points, important definitions, "
            "formulas, and concepts to memorize. Use bullet points and bold text."
        )
    else:
        instruction = (
            "Provide a comprehensive, detailed summary covering all major topics, "
            "sub-topics, and important details from the material."
        )
    return f"""You are an expert academic assistant. {instruction}

MATERIAL:
{context}

SUMMARY:"""


def build_quiz_prompt(text_chunks: List[str], quiz_type: str = "MCQs") -> str:
    context = assemble_context(text_chunks, max_chars=MAX_CONTEXT_CHARS)
    instruction = (
        "Generate exactly 5 multiple-choice questions (MCQs) based STRICTLY on the uploaded material above. "
        "Return ONLY a single valid JSON array. Do not wrap in markdown code fences. Do not add commentary. "
        'Each element MUST have these exact keys: "question", "options" (object with keys A, B, C, D), '
        '"correct_answer" (single string: A, B, C or D), and "explanation" (string).'
    )
    return f"""You are an expert exam setter.

{instruction}

MATERIAL:
{context}

RAW JSON OUTPUT:"""


def build_notes_prompt(text_chunks: List[str]) -> str:
    context = assemble_context(text_chunks, max_chars=MAX_CONTEXT_CHARS)
    return f"""You are an expert note-taking assistant. Create concise, well-organized short notes from the following material.
Format the notes with clear headings, bullet points, and bold key terms.

MATERIAL:
{context}

SHORT NOTES:"""


def build_explain_prompt(topic: str, context_chunks: List[str]) -> str:
    context = assemble_context(context_chunks, max_chars=MAX_CONTEXT_CHARS)
    return f"""You are a patient and knowledgeable tutor. Explain the following topic in simple, easy-to-understand language.
Use analogies and examples where possible. If the uploaded material is relevant, incorporate it into your explanation.

UPLOADED MATERIAL:
{context}

TOPIC:
{topic}

EXPLANATION:"""


def build_study_planner_prompt(exam_date_str: str, subjects_str: str, hours_per_day: int) -> str:
    # No PDF context – no truncation needed
    return f"""You are an expert academic and productivity coach. Create a detailed, day-by-day study plan.

Student Details:
- Exam Date: {exam_date_str}
- Subjects to study: {subjects_str}
- Available study hours per day: {hours_per_day}

Requirements:
- Count days remaining from today until the exam.
- Allocate time fairly across all subjects.
- Include short breaks and longer revision sessions.
- Make the plan realistic and sustainable.
- Present in a structured Markdown table or clear daily lists.

STUDY PLAN:"""


# ── Quiz JSON parser ──────────────────────────────────────────────────────

def parse_quiz_json(raw_response: str) -> List[Dict[str, Any]]:
    """Robustly extract a JSON array from an LLM response."""
    if not raw_response or not raw_response.strip():
        return []

    text = raw_response.strip()

    if "```" in text:
        fences = text.split("```")
        for part in fences:
            inner = part.strip()
            if inner.lower().startswith("json"):
                inner = inner[4:].strip()
            if inner.startswith("["):
                text = inner
                break

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed
    except json.JSONDecodeError:
        pass

    try:
        match = re.search(r'(\[\s*\{.*?\}\s*\])', text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, list):
                return parsed
    except Exception:
        pass

    return []
