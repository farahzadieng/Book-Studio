#!/usr/bin/env python3
"""Book Podcast Studio — a local, single-file Flask application.

Release: 1.0.3-planning-recovery

The app turns English PDF/EPUB nonfiction books into evidence-grounded,
documentary-style podcast scripts using local GGUF models through llama.cpp.
All intermediate state is stored in SQLite so long jobs can be paused/resumed.

Run:
    source .venv/bin/activate
    python app.py
"""

from __future__ import annotations

import gc
import json
import os
import re
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from flask import Flask, abort, jsonify, render_template_string, request, send_file, session
from werkzeug.utils import secure_filename


# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_CONFIG_PATH = BASE_DIR / "models.json"
BOOKS_DIR = BASE_DIR / "books"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "book_podcast.db"
SECRET_PATH = DATA_DIR / ".flask_secret"

ALLOWED_EXTENSIONS = {".pdf", ".epub"}
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
MAX_EPUB_UNCOMPRESSED_BYTES = 500 * 1024 * 1024
DEFAULT_CONTEXT = 8192
CHUNK_TOKENS = 2600
CHUNK_OVERLAP = 200
APP_VERSION = "1.0.3-planning-recovery"

MODEL_SLOTS = {
    "analysis": "analysis_model",
    "writing": "writing_model",
    "fast": "fast_model",
}
MODEL_SLOT_FALLBACK_PATTERNS = {
    "analysis": r"qwen2\.5.*14b",
    "writing": r"gemma-3.*12b",
    "fast": r"gemma-3.*4b",
}

for directory in (MODELS_DIR, BOOKS_DIR, OUTPUTS_DIR, DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def load_or_create_secret() -> str:
    if SECRET_PATH.exists():
        return SECRET_PATH.read_text(encoding="utf-8").strip()
    value = secrets.token_hex(32)
    SECRET_PATH.write_text(value, encoding="utf-8")
    try:
        SECRET_PATH.chmod(0o600)
    except OSError:
        pass
    return value


app = Flask(__name__)
app.config.update(
    SECRET_KEY=load_or_create_secret(),
    MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
    JSON_SORT_KEYS=False,
)

JOBS: dict[str, threading.Thread] = {}
JOBS_LOCK = threading.Lock()
PIPELINE_LOCK = threading.Lock()  # One loaded model at a time on 16 GB systems.


# ---------------------------------------------------------------------------
# Narrative contract and prompts
# ---------------------------------------------------------------------------

NARRATIVE_CONTRACT = """
NARRATIVE CONSTITUTION — mandatory:
1. Open with an unexpected hook that reframes something familiar.
2. Center two or three documented characters, institutions, forces, or
   contrasting viewpoints instead of reciting a dry timeline.
3. Attach large ideas to source-backed mini-stories and concrete particulars:
   an invention, disaster, failed operation, experiment, decision, or object.
4. Render key scenes with sensory and visual detail only when the source
   provides that detail. Never fabricate dialogue, actions, feelings, or scenery.
5. Reveal several legitimate readings when supported: historical, human,
   technological, and philosophical.
6. Show how people cling to attractive ideas when reality pushes back.
7. Keep the voice conversational, reflective, and lightly dramatic. Avoid
   textbook-summary language, hype, and canned podcast phrasing.
8. Build forward movement: hope -> practical test -> failure or redirection ->
   unforeseen consequence. Adapt the arc honestly when the source differs.
9. End with several plausible interpretations rather than a sealed verdict.

FIDELITY RULES:
- Use only the supplied book evidence. Do not import facts from memory.
- Never invent a person, event, quotation, statistic, motive, scene, or outcome.
- Paraphrases must not be presented as direct quotations.
- When the evidence is thin, write cautiously or omit the claim.
- Distinguish the author's argument from the narrator's interpretation.
""".strip()

ANALYST_SYSTEM = """You are a meticulous nonfiction book analyst and evidence
editor. Extract what the supplied passage actually supports. Do not rely on
outside knowledge. Return valid JSON only, without Markdown fences. Missing
information must be represented by empty arrays or null, never invention."""

WRITER_SYSTEM = f"""You are a documentary podcast writer and story editor.
You turn complex English nonfiction into an accurate, vivid, human narrative.

{NARRATIVE_CONTRACT}
"""


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    file_type TEXT NOT NULL,
    status TEXT NOT NULL,
    current_stage TEXT NOT NULL,
    stage_message TEXT NOT NULL DEFAULT '',
    progress REAL NOT NULL DEFAULT 0,
    duration_minutes INTEGER NOT NULL DEFAULT 60,
    target_words INTEGER NOT NULL DEFAULT 8000,
    context_size INTEGER NOT NULL DEFAULT 8192,
    analysis_model TEXT NOT NULL,
    writing_model TEXT NOT NULL,
    fast_model TEXT,
    structure_confirmed INTEGER NOT NULL DEFAULT 0,
    plan_confirmed INTEGER NOT NULL DEFAULT 0,
    pause_requested INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chapters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    chapter_index INTEGER NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    included INTEGER NOT NULL DEFAULT 1,
    token_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE(project_id, chapter_index)
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    chapter_id INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    analysis_json TEXT,
    status TEXT NOT NULL DEFAULT 'PENDING',
    UNIQUE(chapter_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS chapter_memories (
    chapter_id INTEGER PRIMARY KEY REFERENCES chapters(id) ON DELETE CASCADE,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    memory_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS book_memories (
    project_id TEXT PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
    memory_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS narrative_sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    section_index INTEGER NOT NULL,
    title TEXT NOT NULL,
    purpose TEXT NOT NULL,
    target_words INTEGER NOT NULL,
    plan_json TEXT NOT NULL,
    evidence_json TEXT,
    draft_text TEXT,
    factcheck_json TEXT,
    final_text TEXT,
    status TEXT NOT NULL DEFAULT 'PLANNED',
    UNIQUE(project_id, section_index)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_project_status
ON chunks(project_id, status);
CREATE INDEX IF NOT EXISTS idx_sections_project_status
ON narrative_sections(project_id, status);
CREATE INDEX IF NOT EXISTS idx_events_project
ON events(project_id, id DESC);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db() -> None:
    with db_connect() as conn:
        conn.executescript(SCHEMA)


def recover_interrupted_projects() -> None:
    """Expose a Resume action after the server was stopped mid-pipeline."""
    interrupted = {
        "UPLOADED", "EXTRACTING", "QUEUED", "ANALYZING", "ANALYZED",
        "PLANNING", "DRAFTING", "DRAFTED", "FACT_CHECKING",
        "FACT_CHECKED", "POLISHING",
    }
    placeholders = ",".join("?" for _ in interrupted)
    with db_connect() as conn:
        rows = conn.execute(
            f"SELECT id FROM projects WHERE status IN ({placeholders})",
            tuple(interrupted),
        ).fetchall()
        if rows:
            conn.execute(
                f"""UPDATE projects
                    SET status='PAUSED', current_stage='PAUSED', progress=0,
                        stage_message='The server stopped during processing. Resume from the last saved checkpoint.',
                        pause_requested=0, updated_at=?
                    WHERE status IN ({placeholders})""",
                (utc_now(), *tuple(interrupted)),
            )
    for row in rows:
        add_event(row["id"], "Interrupted server session detected; project made resumable", "WARNING")


def add_event(project_id: str, message: str, level: str = "INFO") -> None:
    with db_connect() as conn:
        conn.execute(
            "INSERT INTO events(project_id, level, message, created_at) VALUES(?,?,?,?)",
            (project_id, level, message, utc_now()),
        )


def update_project(project_id: str, **fields: Any) -> None:
    if not fields:
        return
    allowed = {
        "title", "status", "current_stage", "stage_message", "progress",
        "duration_minutes", "target_words", "analysis_model", "writing_model",
        "fast_model", "structure_confirmed", "plan_confirmed",
        "pause_requested", "error",
    }
    if not set(fields).issubset(allowed):
        raise ValueError("Invalid project field")
    fields["updated_at"] = utc_now()
    assignments = ", ".join(f"{key}=?" for key in fields)
    values = list(fields.values()) + [project_id]
    with db_connect() as conn:
        conn.execute(f"UPDATE projects SET {assignments} WHERE id=?", values)


def get_project(project_id: str) -> sqlite3.Row:
    with db_connect() as conn:
        row = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
    if row is None:
        raise KeyError(project_id)
    return row


def pause_requested(project_id: str) -> bool:
    try:
        return bool(get_project(project_id)["pause_requested"])
    except KeyError:
        return True


# ---------------------------------------------------------------------------
# Model discovery and safe path handling
# ---------------------------------------------------------------------------

def load_model_config() -> dict[str, str]:
    """Read models.json and return the configured path for each pipeline slot."""
    if not MODELS_CONFIG_PATH.is_file():
        return {}
    try:
        payload = json.loads(MODELS_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    configured = {}
    for slot, key in MODEL_SLOTS.items():
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            configured[slot] = value.strip()
    return configured


def resolve_model(model_path: str) -> Path:
    """Resolve a configured path, an absolute path, or a path inside models/."""
    raw = Path(model_path).expanduser()
    candidates = [raw] if raw.is_absolute() else [BASE_DIR / raw, MODELS_DIR / raw]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.suffix.lower() == ".gguf" and resolved.is_file():
            return resolved
    raise FileNotFoundError(f"GGUF model not found: {model_path}")


def discover_models() -> list[dict[str, str]]:
    """Configured models first, then every GGUF found under models/."""
    models: list[dict[str, str]] = []
    seen: set[Path] = set()
    for configured in load_model_config().values():
        try:
            resolved = resolve_model(configured)
        except FileNotFoundError:
            continue
        if resolved not in seen:
            seen.add(resolved)
            models.append({"path": configured, "name": resolved.name})
    for path in sorted(MODELS_DIR.rglob("*.gguf")):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            models.append({"path": str(resolved), "name": resolved.name})
    return models


def missing_configured_models() -> list[dict[str, str]]:
    return [
        {"slot": slot, "path": configured}
        for slot, configured in load_model_config().items()
        if not _model_exists(configured)
    ]


def _model_exists(model_path: str) -> bool:
    try:
        resolve_model(model_path)
    except FileNotFoundError:
        return False
    return True


def choose_default_model(models: list[dict[str, str]], pattern: str) -> str:
    regex = re.compile(pattern, re.I)
    for model in models:
        if regex.search(model["name"]) or regex.search(model["path"]):
            return model["path"]
    return models[0]["path"] if models else ""


def default_model_paths(models: list[dict[str, str]]) -> dict[str, str]:
    """models.json wins; filename patterns pick a default when a slot is unset."""
    available = {model["path"] for model in models}
    configured = load_model_config()
    defaults = {}
    for slot, pattern in MODEL_SLOT_FALLBACK_PATTERNS.items():
        chosen = configured.get(slot)
        defaults[slot] = chosen if chosen in available else choose_default_model(models, pattern)
    return defaults


# ---------------------------------------------------------------------------
# Text extraction and chunking
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def validate_epub_archive(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        total = sum(info.file_size for info in infos)
        if total > MAX_EPUB_UNCOMPRESSED_BYTES:
            raise ValueError("EPUB expands beyond the 500 MB safety limit")
        for info in infos:
            normalized = Path(info.filename)
            if normalized.is_absolute() or ".." in normalized.parts:
                raise ValueError("Unsafe path found inside EPUB")


def extract_epub(path: Path) -> tuple[str, list[tuple[str, str]]]:
    validate_epub_archive(path)
    from bs4 import BeautifulSoup
    from ebooklib import ITEM_DOCUMENT, epub

    book = epub.read_epub(str(path), options={"ignore_ncx": False})
    metadata = book.get_metadata("DC", "title")
    title = metadata[0][0].strip() if metadata else path.stem
    chapters: list[tuple[str, str]] = []

    for item in book.get_items_of_type(ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        for tag in soup(["script", "style", "nav", "svg"]):
            tag.decompose()
        heading = soup.find(["h1", "h2", "h3"])
        chapter_title = heading.get_text(" ", strip=True) if heading else item.get_name()
        text = clean_text(soup.get_text("\n", strip=True))
        if len(text) >= 250:
            chapters.append((chapter_title[:240], text))

    if not chapters:
        raise ValueError("No readable text chapters were found in this EPUB")
    return title, chapters


HEADING_RE = re.compile(
    r"^(?:(?:chapter|part|book)\s+(?:\d+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)\b.*|"
    r"introduction|prologue|epilogue|conclusion|afterword|preface)$",
    re.I,
)


def fallback_pdf_sections(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        candidate = line.strip()
        if 2 <= len(candidate) <= 100 and HEADING_RE.match(candidate):
            starts.append((index, candidate))

    sections: list[tuple[str, str]] = []
    if len(starts) >= 2:
        if starts[0][0] > 0:
            front = clean_text("\n".join(lines[: starts[0][0]]))
            if len(front) > 300:
                sections.append(("Front matter", front))
        for pos, (start, heading) in enumerate(starts):
            end = starts[pos + 1][0] if pos + 1 < len(starts) else len(lines)
            body = clean_text("\n".join(lines[start + 1 : end]))
            if len(body) > 300:
                sections.append((heading, body))
        return sections

    words = text.split()
    if not words:
        return []
    block_words = 9000
    for index in range(0, len(words), block_words):
        body = " ".join(words[index : index + block_words])
        sections.append((f"Detected section {index // block_words + 1}", body))
    return sections


def extract_pdf(path: Path) -> tuple[str, list[tuple[str, str]]]:
    import fitz

    document = fitz.open(path)
    if document.page_count == 0:
        raise ValueError("PDF contains no pages")
    title = (document.metadata or {}).get("title") or path.stem
    page_texts = []
    for index, page in enumerate(document):
        page_text = clean_text(page.get_text("text", sort=True))
        page_texts.append(f"[[PAGE {index + 1}]]\n{page_text}")

    total_letters = sum(sum(char.isalpha() for char in page) for page in page_texts)
    if total_letters < max(500, document.page_count * 40):
        raise ValueError(
            "This PDF appears to be scanned or image-only. MVP supports PDFs "
            "with embedded text; run OCR first and upload the searchable PDF."
        )

    toc = [entry for entry in document.get_toc(simple=True) if entry[0] == 1]
    chapters: list[tuple[str, str]] = []
    if len(toc) >= 2:
        for index, (_, chapter_title, page_number) in enumerate(toc):
            start = max(0, page_number - 1)
            end = max(start + 1, toc[index + 1][2] - 1) if index + 1 < len(toc) else len(page_texts)
            body = clean_text("\n\n".join(page_texts[start:end]))
            if len(body) > 300:
                chapters.append((chapter_title.strip()[:240], body))
    if not chapters:
        chapters = fallback_pdf_sections(clean_text("\n\n".join(page_texts)))
    document.close()
    return title.strip()[:240], chapters


class ApproxTokenizer:
    """Fallback tokenizer used only if vocab-only model loading fails."""

    def tokenize_text(self, text: str) -> list[str]:
        return re.findall(r"\S+\s*", text)

    def detokenize_text(self, tokens: Iterable[str]) -> str:
        return "".join(tokens)


class LlamaTokenizerAdapter:
    def __init__(self, model_path: Path):
        from llama_cpp import Llama
        self.model = Llama(model_path=str(model_path), vocab_only=True, verbose=False)

    def tokenize_text(self, text: str) -> list[int]:
        return self.model.tokenize(text.encode("utf-8"), add_bos=False)

    def detokenize_text(self, tokens: Iterable[int]) -> str:
        return self.model.detokenize(list(tokens)).decode("utf-8", errors="replace")

    def close(self) -> None:
        del self.model
        gc.collect()


def split_with_tokenizer(text: str, tokenizer: Any) -> list[tuple[str, int]]:
    tokens = tokenizer.tokenize_text(text)
    if not tokens:
        return []
    chunks = []
    step = CHUNK_TOKENS - CHUNK_OVERLAP
    for start in range(0, len(tokens), step):
        piece = tokens[start : start + CHUNK_TOKENS]
        value = clean_text(tokenizer.detokenize_text(piece))
        if value:
            chunks.append((value, len(piece)))
        if start + CHUNK_TOKENS >= len(tokens):
            break
    return chunks


def extraction_job(project_id: str) -> None:
    project = get_project(project_id)
    update_project(
        project_id,
        status="EXTRACTING",
        current_stage="EXTRACTING",
        stage_message="Reading the book and detecting chapters…",
        progress=2,
        error=None,
    )
    add_event(project_id, "Extraction started")
    path = Path(project["stored_path"])
    if project["file_type"] == ".pdf":
        title, chapters = extract_pdf(path)
    else:
        title, chapters = extract_epub(path)

    if not chapters:
        raise ValueError("No usable chapters were detected")
    update_project(project_id, title=title, progress=35, stage_message="Tokenizing detected chapters…")

    tokenizer: Any
    try:
        tokenizer = LlamaTokenizerAdapter(resolve_model(project["analysis_model"]))
        add_event(project_id, "Using the analysis model tokenizer for exact chunking")
    except Exception as exc:
        tokenizer = ApproxTokenizer()
        add_event(project_id, f"Tokenizer fallback used: {exc}", "WARNING")

    with db_connect() as conn:
        conn.execute("DELETE FROM chapters WHERE project_id=?", (project_id,))
        for chapter_index, (chapter_title, text) in enumerate(chapters):
            token_count = len(tokenizer.tokenize_text(text))
            cursor = conn.execute(
                "INSERT INTO chapters(project_id, chapter_index, title, text, token_count) VALUES(?,?,?,?,?)",
                (project_id, chapter_index, chapter_title or f"Chapter {chapter_index + 1}", text, token_count),
            )
            chapter_id = cursor.lastrowid
            for chunk_index, (chunk_text, count) in enumerate(split_with_tokenizer(text, tokenizer)):
                conn.execute(
                    "INSERT INTO chunks(project_id, chapter_id, chunk_index, text, token_count) VALUES(?,?,?,?,?)",
                    (project_id, chapter_id, chunk_index, chunk_text, count),
                )

    if hasattr(tokenizer, "close"):
        tokenizer.close()
    update_project(
        project_id,
        status="STRUCTURE_REVIEW",
        current_stage="STRUCTURE_REVIEW",
        stage_message=f"Review {len(chapters)} detected sections before analysis.",
        progress=100,
    )
    add_event(project_id, f"Extraction completed: {len(chapters)} sections detected")


# ---------------------------------------------------------------------------
# LLM helpers and stage workers
# ---------------------------------------------------------------------------

def load_llm(model_relative_path: str, context_size: int):
    from llama_cpp import Llama
    path = resolve_model(model_relative_path)
    return Llama(
        model_path=str(path),
        n_ctx=context_size,
        n_batch=256,
        n_gpu_layers=-1,
        verbose=False,
    )


def chat_text(
    llm: Any,
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    json_mode: bool = False,
) -> str:
    request_args = dict(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_tokens,
    )
    # llama-cpp-python translates json_object into a JSON grammar. This keeps
    # local models from emitting Markdown, commentary, or a second JSON value.
    if json_mode:
        request_args["response_format"] = {"type": "json_object"}
    try:
        result = llm.create_chat_completion(**request_args)
    except TypeError as exc:
        # Compatibility fallback for older llama-cpp-python builds. The robust
        # parser and repair pass below still protect the pipeline.
        if not json_mode or "response_format" not in str(exc):
            raise
        request_args.pop("response_format", None)
        result = llm.create_chat_completion(**request_args)
    text = result["choices"][0]["message"]["content"]
    return text.strip()


def parse_json_response(text: str) -> Any:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()
    first_error: json.JSONDecodeError | None = None
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        first_error = exc

    # json.loads rejects otherwise valid model output followed by commentary or
    # a second JSON object as "Extra data". raw_decode returns the first complete
    # value, so scan every plausible object/array start until one parses.
    starts = [index for index, char in enumerate(cleaned) if char in "{["]
    for start in starts:
        try:
            value, _end = decoder.raw_decode(cleaned, start)
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return value
    if first_error is not None:
        raise first_error
    raise json.JSONDecodeError("No JSON object or array found", cleaned, 0)


def chat_json(llm: Any, system_prompt: str, user_prompt: str, max_tokens: int = 1400) -> Any:
    raw = chat_text(
        llm,
        system_prompt,
        user_prompt + "\n\nReturn valid JSON only.",
        max_tokens=max_tokens,
        temperature=0.1,
        json_mode=True,
    )
    try:
        return parse_json_response(raw)
    except Exception:
        repair = chat_text(
            llm,
            "You repair malformed JSON. Return exactly one compact JSON object. "
            "Preserve meaning, add no facts, shorten long strings, and close every array and object.",
            "Repair this into one valid, compact JSON object:\n\n" + raw[:10000],
            max_tokens=min(2400, max_tokens + 600),
            temperature=0,
            json_mode=True,
        )
        try:
            return parse_json_response(repair)
        except Exception as repair_error:
            # Evidence extraction must remain resumable even if a small local
            # model twice fails structured output. Later synthesis can still
            # use the preserved raw text; planning performs its own validation.
            return {
                "_parse_warning": str(repair_error),
                "_raw_model_output": raw[:8000],
            }


def compact_json(value: Any, max_chars: int = 18000) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return text if len(text) <= max_chars else text[:max_chars] + "…"


def chunk_analysis_prompt(project: sqlite3.Row, chapter: sqlite3.Row, chunk: sqlite3.Row) -> str:
    return f"""BOOK: {project['title']}
CHAPTER/SECTION: {chapter['title']}
SOURCE CHUNK ID: {chunk['id']}

Extract a source-grounded evidence record with exactly these top-level keys:
central_claims, supporting_arguments, questions_and_answers, characters,
contrasting_viewpoints, concrete_examples, source_backed_scenes,
belief_vs_reality, interpretive_layers, narrative_arc_beats, connection_to_thesis,
importance_score, cautions.

Requirements:
- Every array item must be concise and traceable to this chunk.
- For examples/scenes include source_chunk_id={chunk['id']} and a short evidence
  excerpt, not a fabricated quotation.
- characters should contain name/entity, documented_traits, goal, motive,
  conflict, and role when supported.
- interpretive_layers may contain historical, human, technological,
  philosophical; omit unsupported layers.
- narrative_arc_beats may contain hope, practical_test, failure_or_redirection,
  unforeseen_consequence.
- importance_score is an integer from 1 to 5.
- Keep at most five items per array and each prose field under 60 words.

SOURCE TEXT:
---
{chunk['text']}
---
"""


def summarize_evidence(llm: Any, chapter_title: str, evidence: list[Any]) -> dict[str, Any]:
    fragments: list[Any] = []
    batch: list[Any] = []
    batch_chars = 0
    for item in evidence:
        size = len(compact_json(item, 8000))
        if batch and batch_chars + size > 15000:
            fragments.append(_summarize_evidence_batch(llm, chapter_title, batch))
            batch, batch_chars = [], 0
        batch.append(item)
        batch_chars += size
    if batch:
        fragments.append(_summarize_evidence_batch(llm, chapter_title, batch))
    if len(fragments) == 1:
        return fragments[0]
    return chat_json(
        llm,
        ANALYST_SYSTEM,
        f"""Merge these partial evidence memories for chapter {chapter_title!r}.
Deduplicate without losing source chunk IDs. Return keys: thesis_role,
arguments, characters_and_forces, worldview_conflicts, best_examples,
best_scenes, belief_vs_reality, arc_beats, interpretive_layers, open_questions,
source_chunk_ids, importance_score, cautions.

PARTIAL MEMORIES:
{compact_json(fragments, 18000)}""",
        max_tokens=1500,
    )


def _summarize_evidence_batch(llm: Any, chapter_title: str, batch: list[Any]) -> dict[str, Any]:
    return chat_json(
        llm,
        ANALYST_SYSTEM,
        f"""Create a compact chapter evidence memory for {chapter_title!r} from
the records below. Preserve source chunk IDs. Return keys: thesis_role,
arguments, characters_and_forces, worldview_conflicts, best_examples,
best_scenes, belief_vs_reality, arc_beats, interpretive_layers, open_questions,
source_chunk_ids, importance_score, cautions.

EVIDENCE RECORDS:
{compact_json(batch, 16000)}

Keep at most five items per array and each prose field under 50 words.""",
        max_tokens=1800,
    )


def build_book_memory(llm: Any, title: str, chapter_memories: list[dict[str, Any]]) -> dict[str, Any]:
    prompt = f"""Build the final editorial memory for the nonfiction book
{title!r}. Compress the chapter memories without flattening disagreement.

Return keys: central_thesis, central_question, audience_relevance,
characters_and_forces, key_arguments, worldview_conflicts, narrative_candidates,
belief_vs_reality_pattern, overarching_arc, interpretive_layers,
chapter_priorities, source_chunk_ids, unresolved_questions, factual_cautions.
For narrative_candidates preserve relevant source_chunk_ids. Select only two or
three strongest recurring characters, forces, or viewpoints for the central cast.

CHAPTER MEMORIES:
{compact_json(chapter_memories, 18000)}"""
    if len(compact_json(chapter_memories)) <= 18000:
        return chat_json(llm, ANALYST_SYSTEM, prompt, max_tokens=1900)

    partials = []
    for start in range(0, len(chapter_memories), 5):
        partials.append(
            chat_json(
                llm,
                ANALYST_SYSTEM,
                f"Summarize this group of chapter memories while preserving source IDs:\n"
                + compact_json(chapter_memories[start : start + 5], 16000),
                max_tokens=1200,
            )
        )
    return chat_json(
        llm,
        ANALYST_SYSTEM,
        f"Create the final book editorial memory using the required keys from these partials:\n"
        + compact_json(partials, 18000),
        max_tokens=1900,
    )


def stage_analysis(project_id: str) -> int:
    project = get_project(project_id)
    # A failed planning run resumes through this pipeline. If every analysis
    # checkpoint already exists, do not reload Qwen or rebuild the book memory.
    with db_connect() as conn:
        pending_chunks = conn.execute(
            """SELECT COUNT(*) AS count FROM chunks c
               JOIN chapters h ON h.id=c.chapter_id
               WHERE c.project_id=? AND h.included=1
                 AND (c.status!='ANALYZED' OR c.analysis_json IS NULL)""",
            (project_id,),
        ).fetchone()["count"]
        included_chapters = conn.execute(
            "SELECT COUNT(*) AS count FROM chapters WHERE project_id=? AND included=1",
            (project_id,),
        ).fetchone()["count"]
        chapter_memories = conn.execute(
            """SELECT COUNT(*) AS count FROM chapter_memories m
               JOIN chapters h ON h.id=m.chapter_id
               WHERE m.project_id=? AND h.included=1""",
            (project_id,),
        ).fetchone()["count"]
        has_book_memory = conn.execute(
            "SELECT 1 FROM book_memories WHERE project_id=?",
            (project_id,),
        ).fetchone() is not None
    if pending_chunks == 0 and included_chapters > 0 and chapter_memories == included_chapters and has_book_memory:
        update_project(project_id, status="ANALYZED", current_stage="ANALYZED", progress=100,
                       stage_message="Reusing the completed evidence memory for planning.", error=None)
        add_event(project_id, "Completed analysis checkpoints reused; Qwen reload skipped")
        return 0

    update_project(project_id, status="ANALYZING", current_stage="ANALYZING", progress=0,
                   stage_message="Qwen is extracting claims, characters, scenes, and evidence…", error=None)
    add_event(project_id, "Analysis model loading")
    llm = load_llm(project["analysis_model"], project["context_size"])
    try:
        with db_connect() as conn:
            chunks = conn.execute(
                """SELECT c.*, h.title AS chapter_title FROM chunks c
                   JOIN chapters h ON h.id=c.chapter_id
                   WHERE c.project_id=? AND h.included=1 ORDER BY h.chapter_index,c.chunk_index""",
                (project_id,),
            ).fetchall()
        total = max(1, len(chunks))
        for index, chunk in enumerate(chunks):
            if pause_requested(project_id):
                update_project(project_id, status="PAUSED", stage_message="Paused after the current chunk.")
                return 75
            if chunk["status"] == "ANALYZED" and chunk["analysis_json"]:
                continue
            chapter = {"title": chunk["chapter_title"]}
            result = chat_json(llm, ANALYST_SYSTEM, chunk_analysis_prompt(project, chapter, chunk), 1500)
            with db_connect() as conn:
                conn.execute(
                    "UPDATE chunks SET analysis_json=?, status='ANALYZED' WHERE id=?",
                    (json.dumps(result, ensure_ascii=False), chunk["id"]),
                )
            update_project(project_id, progress=round((index + 1) / total * 72, 1),
                           stage_message=f"Analyzed evidence chunk {index + 1} of {total}")

        with db_connect() as conn:
            chapters = conn.execute(
                "SELECT * FROM chapters WHERE project_id=? AND included=1 ORDER BY chapter_index",
                (project_id,),
            ).fetchall()
        memories = []
        for index, chapter in enumerate(chapters):
            if pause_requested(project_id):
                update_project(project_id, status="PAUSED", stage_message="Paused before chapter synthesis.")
                return 75
            with db_connect() as conn:
                existing = conn.execute("SELECT memory_json FROM chapter_memories WHERE chapter_id=?", (chapter["id"],)).fetchone()
                records = conn.execute(
                    "SELECT analysis_json FROM chunks WHERE chapter_id=? ORDER BY chunk_index",
                    (chapter["id"],),
                ).fetchall()
            if existing:
                memory = json.loads(existing["memory_json"])
            else:
                evidence = [json.loads(row["analysis_json"]) for row in records if row["analysis_json"]]
                memory = summarize_evidence(llm, chapter["title"], evidence)
                with db_connect() as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO chapter_memories(chapter_id,project_id,memory_json) VALUES(?,?,?)",
                        (chapter["id"], project_id, json.dumps(memory, ensure_ascii=False)),
                    )
            memories.append({"chapter_id": chapter["id"], "title": chapter["title"], "memory": memory})
            update_project(project_id, progress=72 + round((index + 1) / max(1, len(chapters)) * 18, 1),
                           stage_message=f"Built chapter memory {index + 1} of {len(chapters)}")

        book_memory = build_book_memory(llm, project["title"], memories)
        with db_connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO book_memories(project_id,memory_json) VALUES(?,?)",
                (project_id, json.dumps(book_memory, ensure_ascii=False)),
            )
        update_project(project_id, status="ANALYZED", current_stage="ANALYZED", progress=100,
                       stage_message="Evidence memory complete. Preparing the narrative plan…")
        add_event(project_id, "Book analysis and hierarchical memory completed")
        return 0
    finally:
        del llm
        gc.collect()


def extract_plan_sections(plan: Any) -> list[dict[str, Any]]:
    """Accept common local-model variations of the requested plan schema."""
    if not isinstance(plan, dict):
        return []
    candidate: Any = None
    for key in ("sections", "narrative_sections", "episode_sections", "outline", "story_spine", "acts"):
        if key in plan:
            candidate = plan[key]
            break
    if candidate is None:
        return []

    extracted: list[dict[str, Any]] = []

    def visit(value: Any, inherited_title: str = "", depth: int = 0) -> None:
        if depth > 3:
            return
        if isinstance(value, list):
            for item in value:
                visit(item, "", depth + 1)
            return
        if isinstance(value, str):
            extracted.append({"title": value, "purpose": "Advance the central narrative."})
            return
        if not isinstance(value, dict):
            return
        if any(key in value for key in ("title", "purpose", "argument", "target_words", "hook_or_question")):
            item = dict(value)
            if inherited_title and not item.get("title"):
                item["title"] = inherited_title
            extracted.append(item)
            return
        for nested_key in ("sections", "beats", "chapters", "items"):
            if nested_key in value:
                visit(value[nested_key], inherited_title, depth + 1)
                return
        # Some models return an object keyed by section title.
        for title, nested in value.items():
            if isinstance(nested, (dict, list, str)):
                visit(nested, str(title), depth + 1)

    visit(candidate)
    return extracted[:14]


def numeric_weight(value: Any) -> int:
    if isinstance(value, (int, float)):
        return max(1, int(value))
    match = re.search(r"\d+", str(value or ""))
    return max(1, int(match.group())) if match else 1


def normalize_plan_sections(sections: list[dict[str, Any]], target_words: int) -> list[dict[str, Any]]:
    cleaned = []
    for index, section in enumerate(sections[:14]):
        if not isinstance(section, dict):
            continue
        weight = numeric_weight(section.get("target_words") or section.get("weight") or 1)
        cleaned.append({
            **section,
            "section_index": index,
            "title": str(section.get("title") or f"Section {index + 1}")[:160],
            "purpose": str(section.get("purpose") or "Advance the central narrative.")[:1000],
            "_weight": weight,
        })
    if len(cleaned) < 5:
        raise ValueError("Narrative planner returned too few usable sections")
    total_weight = sum(item["_weight"] for item in cleaned)
    remaining = target_words
    for index, item in enumerate(cleaned):
        if index == len(cleaned) - 1:
            words = remaining
        else:
            words = max(350, round(target_words * item["_weight"] / total_weight))
            remaining -= words
        item["target_words"] = words
        item.pop("_weight", None)
    if cleaned[-1]["target_words"] < 300:
        deficit = 300 - cleaned[-1]["target_words"]
        donor = max(range(len(cleaned) - 1), key=lambda i: cleaned[i]["target_words"])
        cleaned[donor]["target_words"] -= deficit
        cleaned[-1]["target_words"] = 300
    return cleaned


def memory_brief(value: Any, limit: int = 320) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return re.sub(r"\s+", " ", text).strip()[:limit]


def collect_source_ids(value: Any) -> list[int]:
    found: list[int] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {"source_chunk_ids", "preferred_source_chunk_ids"} and isinstance(nested, list):
                found.extend(int(item) for item in nested if str(item).isdigit())
            else:
                found.extend(collect_source_ids(nested))
    elif isinstance(value, list):
        for item in value:
            found.extend(collect_source_ids(item))
    return list(dict.fromkeys(found))


def fallback_narrative_plan(book_memory: dict[str, Any]) -> dict[str, Any]:
    """Deterministic, reviewable plan used only after two malformed LLM plans."""
    thesis = memory_brief(book_memory.get("central_thesis")) or "the book's central claim"
    question = memory_brief(book_memory.get("central_question")) or "What changes when this idea meets reality?"
    cast = book_memory.get("characters_and_forces") or book_memory.get("central_cast") or []
    conflicts = memory_brief(book_memory.get("worldview_conflicts"))
    arguments = memory_brief(book_memory.get("key_arguments"))
    belief_pattern = memory_brief(book_memory.get("belief_vs_reality_pattern"))
    arc = memory_brief(book_memory.get("overarching_arc"))
    layers = book_memory.get("interpretive_layers") or []
    source_ids = collect_source_ids(book_memory)
    templates = [
        ("The Familiar Thing We Misread", "Open on a source-backed surprise and establish the central question.", 8, "hook"),
        ("The Promise", f"Explain why {thesis} first appears persuasive or necessary.", 10, "hope"),
        ("The People and the Divide", f"Introduce two or three documented forces and their conflict: {conflicts or arguments}.", 11, "characters"),
        ("When the Idea Meets the World", "Connect the argument to its first concrete test, decision, experiment, or event.", 14, "practical_test"),
        ("Evidence Pushes Back", f"Show the strongest collision between belief and reality: {belief_pattern or arc}.", 16, "failure_or_redirection"),
        ("The Change of Course", "Trace how people adapt, resist, reinterpret, or double down after the setback.", 13, "redirection"),
        ("What No One Expected", "Develop the most consequential and well-supported unforeseen result.", 11, "unforeseen_consequence"),
        ("Four Ways to Read What Happened", "Explore the supported historical, human, technological, and philosophical layers without forcing symmetry.", 10, "interpretation"),
        ("The Question That Remains", "Return to the opening image and leave several evidence-grounded interpretations open.", 7, "open_ending"),
    ]
    sections = []
    for index, (title, purpose, weight, beat) in enumerate(templates):
        preferred = source_ids[index::len(templates)][:5] or source_ids[:5]
        sections.append({
            "title": title,
            "purpose": purpose,
            "weight": weight,
            "hook_or_question": question if index in {0, len(templates) - 1} else "",
            "character_or_viewpoint": memory_brief(cast, 240),
            "argument": thesis if index in {1, 3, 4} else arguments,
            "arc_beat": beat,
            "concrete_evidence": "Retrieve the strongest source-backed example for this beat.",
            "interpretive_layer": memory_brief(layers, 220),
            "transition_in": "Continue from the preceding unresolved tension.",
            "transition_out": "Open the next question without summarizing prematurely.",
            "retrieval_terms": f"{title} {purpose} {thesis}"[:700],
            "preferred_source_chunk_ids": preferred,
            "forbidden_inventions": ["invented dialogue", "invented sensory detail", "unsupported causality"],
        })
    return {
        "theme": thesis,
        "desired_audience_effect": "Understand the book's central idea through evidence, tension, and open interpretation.",
        "central_image": memory_brief(book_memory.get("narrative_candidates")) or "A promising idea meeting resistant reality.",
        "central_question": question,
        "central_cast": cast,
        "interpretive_layers": layers,
        "sections": sections,
        "ending_strategy": "Return to the central question and leave multiple supported readings open.",
        "factual_risks": book_memory.get("factual_cautions") or [],
        "planning_fallback_used": True,
    }


def stage_planning(project_id: str) -> int:
    project = get_project(project_id)
    update_project(project_id, status="PLANNING", current_stage="PLANNING", progress=5,
                   stage_message="Gemma is designing the story spine…")
    with db_connect() as conn:
        row = conn.execute("SELECT memory_json FROM book_memories WHERE project_id=?", (project_id,)).fetchone()
    if not row:
        raise ValueError("Book memory is missing; analysis must finish first")
    book_memory = json.loads(row["memory_json"])
    llm = load_llm(project["writing_model"], project["context_size"])
    try:
        prompt = f"""Design a 50–70 minute English documentary podcast narrative
for {project['title']!r}. Target exactly {project['target_words']} words across
exactly 9 sections. This is a plan, not the script.

{NARRATIVE_CONTRACT}

Return one JSON object with keys: theme, desired_audience_effect, central_image,
central_question, central_cast, interpretive_layers, sections, ending_strategy,
factual_risks. Each sections item must contain: title, purpose, target_words,
hook_or_question, character_or_viewpoint, argument, arc_beat, concrete_evidence,
interpretive_layer, transition_in, transition_out, retrieval_terms,
preferred_source_chunk_ids, forbidden_inventions.
Keep every prose field under 35 words and every array to at most three items.

The sections must collectively move through unexpected hook -> characters and
conflict -> hope -> practical tests -> failure/redirection -> unforeseen
consequences -> multiple readings -> genuinely open ending. Do not force an arc
unsupported by the editorial memory.

BOOK EDITORIAL MEMORY:
{compact_json(book_memory, 13000)}"""
        plan = chat_json(llm, WRITER_SYSTEM, prompt, max_tokens=2600)
        raw_sections = extract_plan_sections(plan)
        if len(raw_sections) < 5:
            add_event(project_id, "Planner schema was incomplete; requesting a compact recovery plan", "WARNING")
            retry_prompt = f"""Create exactly 9 concise sections for an English
documentary podcast about {project['title']!r}. Return one JSON object with only
the key sections. Each section must have: title, purpose, weight,
hook_or_question, character_or_viewpoint, argument, arc_beat,
concrete_evidence, interpretive_layer, retrieval_terms,
preferred_source_chunk_ids, forbidden_inventions. Use no more than 25 words in
any string. Follow this progression when supported: unexpected hook, promise,
characters/conflict, practical test, evidence pushing back, redirection,
unforeseen consequence, multiple readings, open ending.

BOOK MEMORY:
{compact_json(book_memory, 12000)}"""
            retry_plan = chat_json(llm, WRITER_SYSTEM, retry_prompt, max_tokens=2400)
            retry_sections = extract_plan_sections(retry_plan)
            if len(retry_sections) >= 5:
                plan, raw_sections = retry_plan, retry_sections
            else:
                plan = fallback_narrative_plan(book_memory)
                raw_sections = plan["sections"]
                add_event(project_id, "Deterministic nine-section recovery plan created for human review", "WARNING")
        sections = normalize_plan_sections(raw_sections, project["target_words"])
        plan["sections"] = sections
        with db_connect() as conn:
            conn.execute("DELETE FROM narrative_sections WHERE project_id=?", (project_id,))
            for section in sections:
                conn.execute(
                    """INSERT INTO narrative_sections
                    (project_id,section_index,title,purpose,target_words,plan_json,status)
                    VALUES(?,?,?,?,?,?,'PLANNED')""",
                    (
                        project_id, section["section_index"], section["title"],
                        section["purpose"], section["target_words"],
                        json.dumps(section, ensure_ascii=False),
                    ),
                )
        update_project(project_id, status="PLAN_REVIEW", current_stage="PLAN_REVIEW", progress=100,
                       stage_message="Review the narrative plan before script generation.")
        add_event(project_id, f"Narrative plan created with {len(sections)} sections")
        return 0
    finally:
        del llm
        gc.collect()


STOPWORDS = {
    "the", "and", "that", "with", "from", "this", "into", "their", "about",
    "have", "will", "would", "could", "should", "what", "when", "where", "which",
    "section", "book", "story", "through", "between", "more", "than", "then",
}


def keywords(text: str) -> Counter[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", text.lower())
    return Counter(word for word in words if word not in STOPWORDS)


def retrieve_evidence(project_id: str, section: sqlite3.Row, limit: int = 7) -> list[dict[str, Any]]:
    plan = json.loads(section["plan_json"])
    query = " ".join([
        section["title"], section["purpose"], str(plan.get("argument", "")),
        str(plan.get("concrete_evidence", "")), str(plan.get("retrieval_terms", "")),
    ])
    query_terms = keywords(query)
    preferred = {int(value) for value in plan.get("preferred_source_chunk_ids", []) if str(value).isdigit()}
    with db_connect() as conn:
        rows = conn.execute(
            """SELECT c.id,c.text,c.analysis_json,h.title AS chapter_title
               FROM chunks c JOIN chapters h ON h.id=c.chapter_id
               WHERE c.project_id=? AND h.included=1 AND c.analysis_json IS NOT NULL""",
            (project_id,),
        ).fetchall()
    scored = []
    for row in rows:
        haystack = f"{row['chapter_title']} {row['analysis_json']}"
        terms = keywords(haystack)
        overlap = sum(min(count, terms.get(term, 0)) for term, count in query_terms.items())
        score = overlap + (25 if row["id"] in preferred else 0)
        scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = []
    for score, row in scored[:limit]:
        selected.append({
            "source_chunk_id": row["id"],
            "chapter": row["chapter_title"],
            "evidence_record": json.loads(row["analysis_json"]),
            "source_excerpt": row["text"][:2600],
            "retrieval_score": score,
        })
    return selected


def stage_drafting(project_id: str) -> int:
    project = get_project(project_id)
    update_project(project_id, status="DRAFTING", current_stage="DRAFTING", progress=0,
                   stage_message="Gemma is writing the documentary script…", error=None)
    llm = load_llm(project["writing_model"], project["context_size"])
    try:
        with db_connect() as conn:
            sections = conn.execute(
                "SELECT * FROM narrative_sections WHERE project_id=? ORDER BY section_index",
                (project_id,),
            ).fetchall()
        previous_tail = ""
        for index, section in enumerate(sections):
            if pause_requested(project_id):
                update_project(project_id, status="PAUSED", stage_message="Paused after the current script section.")
                return 75
            if section["draft_text"]:
                previous_tail = section["draft_text"][-1600:]
                continue
            evidence = retrieve_evidence(project_id, section)
            plan = json.loads(section["plan_json"])
            next_title = sections[index + 1]["title"] if index + 1 < len(sections) else "the open ending"
            prompt = f"""Write section {index + 1} of {len(sections)} for the
English documentary podcast script about {project['title']!r}.

SECTION TITLE: {section['title']}
SECTION PURPOSE: {section['purpose']}
TARGET LENGTH: approximately {section['target_words']} words (within 15 percent)
NEXT SECTION: {next_title}

SECTION PLAN:
{compact_json(plan, 3500)}

SOURCE EVIDENCE PACK:
{compact_json(evidence, 9000)}

PREVIOUS SECTION ENDING (for continuity; do not repeat it):
{(previous_tail[-700:] if previous_tail else '[This is the opening section.]')}

Write polished narration only. Do not include a heading, word count, citations,
production notes, bracketed sound cues, or commentary about the task. Use
rhetorical questions sparingly and answer most of them through unfolding
evidence. Make the transition toward the next section feel earned."""
            max_tokens = min(2400, max(900, int(section["target_words"] * 1.65) + 250))
            draft = chat_text(llm, WRITER_SYSTEM, prompt, max_tokens=max_tokens, temperature=0.65)
            evidence_record = {"items": evidence, "generated_word_count": len(draft.split())}
            with db_connect() as conn:
                conn.execute(
                    "UPDATE narrative_sections SET evidence_json=?,draft_text=?,status='DRAFTED' WHERE id=?",
                    (json.dumps(evidence_record, ensure_ascii=False), draft, section["id"]),
                )
            previous_tail = draft[-1600:]
            update_project(project_id, progress=round((index + 1) / len(sections) * 100, 1),
                           stage_message=f"Drafted section {index + 1} of {len(sections)}")
        update_project(project_id, status="DRAFTED", current_stage="DRAFTED", progress=100,
                       stage_message="Draft complete. Checking every claim against its evidence…")
        add_event(project_id, "First script draft completed")
        return 0
    finally:
        del llm
        gc.collect()


def stage_factcheck(project_id: str) -> int:
    project = get_project(project_id)
    update_project(project_id, status="FACT_CHECKING", current_stage="FACT_CHECKING", progress=0,
                   stage_message="Qwen is checking claims and narrative integrity…")
    llm = load_llm(project["analysis_model"], project["context_size"])
    try:
        with db_connect() as conn:
            sections = conn.execute(
                "SELECT * FROM narrative_sections WHERE project_id=? ORDER BY section_index",
                (project_id,),
            ).fetchall()
        for index, section in enumerate(sections):
            if pause_requested(project_id):
                update_project(project_id, status="PAUSED", stage_message="Paused after the current fact-check.")
                return 75
            if section["factcheck_json"]:
                continue
            evidence_for_prompt = compact_json(json.loads(section["evidence_json"] or "{}"), 9000)
            prompt = f"""Audit this documentary script section only against its
evidence pack. Return a JSON object with: factual_support_score_0_to_10,
narrative_scores, issues, required_changes, safe_to_publish.

narrative_scores must score each of the nine Narrative Constitution rules from
0 to 2 with a short reason. issues must list exact claim_or_passage, category
(unsupported, exaggerated, invented_detail, false_quote, causal_overreach,
author_narrator_confusion, continuity, repetition, narrative_weakness), severity,
evidence_chunk_ids, and correction_instruction. Do not rewrite the section.

{NARRATIVE_CONTRACT}

EVIDENCE PACK:
{evidence_for_prompt}

SCRIPT SECTION:
---
{section['draft_text']}
---"""
            report = chat_json(llm, ANALYST_SYSTEM, prompt, max_tokens=1700)
            with db_connect() as conn:
                conn.execute(
                    "UPDATE narrative_sections SET factcheck_json=?,status='CHECKED' WHERE id=?",
                    (json.dumps(report, ensure_ascii=False), section["id"]),
                )
            update_project(project_id, progress=round((index + 1) / len(sections) * 100, 1),
                           stage_message=f"Checked section {index + 1} of {len(sections)}")
        update_project(project_id, status="FACT_CHECKED", current_stage="FACT_CHECKED", progress=100,
                       stage_message="Evidence check complete. Applying targeted revisions…")
        add_event(project_id, "Factual and narrative audit completed")
        return 0
    finally:
        del llm
        gc.collect()


def stage_polishing(project_id: str) -> int:
    project = get_project(project_id)
    update_project(project_id, status="POLISHING", current_stage="POLISHING", progress=0,
                   stage_message="Gemma is applying evidence-safe revisions…")
    llm = load_llm(project["writing_model"], project["context_size"])
    try:
        with db_connect() as conn:
            sections = conn.execute(
                "SELECT * FROM narrative_sections WHERE project_id=? ORDER BY section_index",
                (project_id,),
            ).fetchall()
        previous_tail = ""
        for index, section in enumerate(sections):
            if pause_requested(project_id):
                update_project(project_id, status="PAUSED", stage_message="Paused after the current revision.")
                return 75
            if section["final_text"]:
                previous_tail = section["final_text"][-1600:]
                continue
            next_plan = json.loads(sections[index + 1]["plan_json"]) if index + 1 < len(sections) else {"title": "Open ending"}
            evidence_for_prompt = compact_json(json.loads(section["evidence_json"] or "{}"), 6000)
            audit_for_prompt = compact_json(json.loads(section["factcheck_json"] or "{}"), 3500)
            plan_for_prompt = compact_json(json.loads(section["plan_json"]), 2500)
            prompt = f"""Revise this documentary podcast section. Apply every
required factual correction and strengthen weak Narrative Constitution scores
without introducing any new facts. Preserve supported specificity. Target
approximately {section['target_words']} words.

SECTION PLAN:
{plan_for_prompt}

EVIDENCE PACK:
{evidence_for_prompt}

AUDIT REPORT:
{audit_for_prompt}

PREVIOUS FINAL SECTION ENDING:
{(previous_tail[-700:] if previous_tail else '[Opening section]')}

NEXT SECTION PLAN:
{compact_json(next_plan, 900)}

DRAFT TO REVISE:
---
{section['draft_text']}
---

Return narration only: no heading, citations, production notes, sound cues,
word count, or explanation."""
            max_tokens = min(2400, max(900, int(section["target_words"] * 1.65) + 250))
            final_text = chat_text(llm, WRITER_SYSTEM, prompt, max_tokens=max_tokens, temperature=0.45)
            with db_connect() as conn:
                conn.execute(
                    "UPDATE narrative_sections SET final_text=?,status='FINAL' WHERE id=?",
                    (final_text, section["id"]),
                )
            previous_tail = final_text[-1600:]
            update_project(project_id, progress=round((index + 1) / len(sections) * 100, 1),
                           stage_message=f"Polished section {index + 1} of {len(sections)}")
        export_project(project_id)
        update_project(project_id, status="COMPLETED", current_stage="COMPLETED", progress=100,
                       stage_message="The evidence-grounded podcast script is ready.")
        add_event(project_id, "Final script exported as Markdown and plain text")
        return 0
    finally:
        del llm
        gc.collect()


STAGE_FUNCTIONS = {
    "analysis": stage_analysis,
    "planning": stage_planning,
    "drafting": stage_drafting,
    "factcheck": stage_factcheck,
    "polishing": stage_polishing,
}


def export_paths(project: sqlite3.Row) -> tuple[Path, Path]:
    stem = secure_filename(project["title"])[:80] or project["id"]
    return OUTPUTS_DIR / f"{stem}-podcast-script.md", OUTPUTS_DIR / f"{stem}-podcast-script.txt"


def export_project(project_id: str) -> None:
    project = get_project(project_id)
    with db_connect() as conn:
        sections = conn.execute(
            "SELECT title,final_text,draft_text FROM narrative_sections WHERE project_id=? ORDER BY section_index",
            (project_id,),
        ).fetchall()
    md_path, txt_path = export_paths(project)
    md_parts = [f"# {project['title']}\n\n*Documentary podcast script*\n"]
    txt_parts = [project["title"], "Documentary podcast script", ""]
    for section in sections:
        text = section["final_text"] or section["draft_text"] or ""
        md_parts.append(f"## {section['title']}\n\n{text}\n")
        txt_parts.extend([section["title"].upper(), "", text, ""])
    md_path.write_text("\n".join(md_parts).strip() + "\n", encoding="utf-8")
    txt_path.write_text("\n".join(txt_parts).strip() + "\n", encoding="utf-8")


def run_stage_subprocess(project_id: str, stage: str) -> int:
    command = [sys.executable, str(Path(__file__).resolve()), "--stage", stage, project_id]
    completed = subprocess.run(command, cwd=BASE_DIR, check=False)
    return completed.returncode


def pipeline_before_plan_review(project_id: str) -> None:
    with PIPELINE_LOCK:
        for stage in ("analysis", "planning"):
            code = run_stage_subprocess(project_id, stage)
            if code == 75:
                return
            if code != 0:
                detail = get_project(project_id)["error"]
                raise RuntimeError(detail or f"Stage {stage} stopped with exit code {code}")


def pipeline_after_plan_review(project_id: str) -> None:
    with PIPELINE_LOCK:
        for stage in ("drafting", "factcheck", "polishing"):
            code = run_stage_subprocess(project_id, stage)
            if code == 75:
                return
            if code != 0:
                detail = get_project(project_id)["error"]
                raise RuntimeError(detail or f"Stage {stage} stopped with exit code {code}")


def launch_job(project_id: str, target: Any) -> bool:
    with JOBS_LOCK:
        current = JOBS.get(project_id)
        if current and current.is_alive():
            return False

        def runner() -> None:
            try:
                target(project_id)
            except Exception as exc:
                traceback.print_exc()
                update_project(
                    project_id,
                    status="FAILED",
                    current_stage="FAILED",
                    stage_message="Processing stopped. Review the error, then resume.",
                    error=str(exc)[:4000],
                )
                add_event(project_id, f"Processing failed: {exc}", "ERROR")
            finally:
                with JOBS_LOCK:
                    JOBS.pop(project_id, None)

        thread = threading.Thread(target=runner, name=f"project-{project_id[:8]}", daemon=True)
        JOBS[project_id] = thread
        thread.start()
        return True


# ---------------------------------------------------------------------------
# API serialization and routes
# ---------------------------------------------------------------------------

def serialize_project(project_id: str, detail: bool = True) -> dict[str, Any]:
    project = get_project(project_id)
    result = dict(project)
    result["error"] = result.get("error") or ""
    md_path, txt_path = export_paths(project)
    result["downloads"] = {"markdown": md_path.exists(), "text": txt_path.exists()}
    with JOBS_LOCK:
        result["job_running"] = bool(JOBS.get(project_id) and JOBS[project_id].is_alive())
    if not detail:
        return result
    with db_connect() as conn:
        chapters = conn.execute(
            "SELECT id,chapter_index,title,included,token_count FROM chapters WHERE project_id=? ORDER BY chapter_index",
            (project_id,),
        ).fetchall()
        sections = conn.execute(
            """SELECT id,section_index,title,purpose,target_words,status,
                      length(COALESCE(final_text,draft_text,'')) AS text_chars
               FROM narrative_sections WHERE project_id=? ORDER BY section_index""",
            (project_id,),
        ).fetchall()
        events = conn.execute(
            "SELECT level,message,created_at FROM events WHERE project_id=? ORDER BY id DESC LIMIT 30",
            (project_id,),
        ).fetchall()
    result["chapters"] = [dict(row) for row in chapters]
    result["sections"] = [dict(row) for row in sections]
    result["events"] = [dict(row) for row in events]
    return result


@app.before_request
def csrf_protect() -> None:
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        supplied = request.headers.get("X-CSRF-Token", "")
        expected = session.get("csrf_token", "")
        if not expected or not secrets.compare_digest(supplied, expected):
            abort(403, "Invalid CSRF token")


@app.get("/")
def index():
    token = session.get("csrf_token") or secrets.token_urlsafe(24)
    session["csrf_token"] = token
    models = discover_models()
    return render_template_string(
        INDEX_HTML,
        csrf_token=token,
        models=models,
        defaults=default_model_paths(models),
        config_warnings=missing_configured_models(),
        app_version=APP_VERSION,
    )


@app.get("/api/projects")
def api_projects():
    with db_connect() as conn:
        rows = conn.execute("SELECT id FROM projects ORDER BY created_at DESC").fetchall()
    return jsonify([serialize_project(row["id"], detail=False) for row in rows])


@app.get("/api/projects/<project_id>")
def api_project(project_id: str):
    try:
        return jsonify(serialize_project(project_id))
    except KeyError:
        abort(404)


def target_words_for_duration(minutes: int) -> int:
    return {50: 6800, 60: 8000, 70: 9300}.get(minutes, 8000)


@app.post("/api/projects")
def create_project():
    uploaded = request.files.get("book")
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "Choose a PDF or EPUB file."}), 400
    original = secure_filename(uploaded.filename)
    extension = Path(original).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Only PDF and EPUB files are supported."}), 400

    models = {item["path"] for item in discover_models()}
    analysis_model = request.form.get("analysis_model", "")
    writing_model = request.form.get("writing_model", "")
    fast_model = request.form.get("fast_model", "")
    if analysis_model not in models or writing_model not in models:
        return jsonify({"error": "Select valid analysis and writing models."}), 400
    if fast_model and fast_model not in models:
        return jsonify({"error": "Select a valid fast model."}), 400

    duration = int(request.form.get("duration", "60"))
    if duration not in {50, 60, 70}:
        return jsonify({"error": "Duration must be 50, 60, or 70 minutes."}), 400

    project_id = uuid.uuid4().hex
    stored = BOOKS_DIR / f"{project_id}{extension}"
    uploaded.save(stored)
    if not stored.exists() or stored.stat().st_size == 0:
        return jsonify({"error": "The uploaded file is empty."}), 400

    created = utc_now()
    with db_connect() as conn:
        conn.execute(
            """INSERT INTO projects
            (id,title,original_filename,stored_path,file_type,status,current_stage,
             stage_message,duration_minutes,target_words,context_size,
             analysis_model,writing_model,fast_model,created_at,updated_at)
            VALUES(?,?,?,?,?,'UPLOADED','UPLOADED','Upload received.',?,?,?,?,?,?,?,?)""",
            (
                project_id, Path(original).stem, original, str(stored), extension,
                duration, target_words_for_duration(duration), DEFAULT_CONTEXT,
                analysis_model, writing_model, fast_model or None, created, created,
            ),
        )
    add_event(project_id, f"Uploaded {original}")
    launch_job(project_id, extraction_job)
    return jsonify(serialize_project(project_id)), 201


@app.post("/api/projects/<project_id>/confirm-structure")
def confirm_structure(project_id: str):
    try:
        project = get_project(project_id)
    except KeyError:
        abort(404)
    if project["status"] != "STRUCTURE_REVIEW":
        return jsonify({"error": "The project is not awaiting structure review."}), 409
    payload = request.get_json(silent=True) or {}
    chapter_updates = payload.get("chapters", [])
    included_count = 0
    with db_connect() as conn:
        valid_ids = {row["id"] for row in conn.execute("SELECT id FROM chapters WHERE project_id=?", (project_id,))}
        for item in chapter_updates:
            chapter_id = int(item.get("id", 0))
            if chapter_id not in valid_ids:
                continue
            title = str(item.get("title") or "Untitled section").strip()[:240]
            included = 1 if item.get("included") else 0
            included_count += included
            conn.execute("UPDATE chapters SET title=?,included=? WHERE id=?", (title, included, chapter_id))
    if included_count == 0:
        return jsonify({"error": "Keep at least one section for analysis."}), 400
    update_project(project_id, structure_confirmed=1, pause_requested=0, error=None,
                   status="QUEUED", current_stage="QUEUED", progress=0,
                   stage_message="Queued for evidence analysis.")
    if not launch_job(project_id, pipeline_before_plan_review):
        update_project(project_id, structure_confirmed=0, status="STRUCTURE_REVIEW",
                       current_stage="STRUCTURE_REVIEW", progress=100,
                       stage_message="The previous task is still closing. Confirm again in a moment.")
        return jsonify({"error": "The previous task is still closing. Try again in a moment."}), 409
    return jsonify(serialize_project(project_id))


@app.post("/api/projects/<project_id>/confirm-plan")
def confirm_plan(project_id: str):
    try:
        project = get_project(project_id)
    except KeyError:
        abort(404)
    if project["status"] != "PLAN_REVIEW":
        return jsonify({"error": "The project is not awaiting plan review."}), 409
    payload = request.get_json(silent=True) or {}
    updates = payload.get("sections", [])
    with db_connect() as conn:
        valid = {row["id"]: row for row in conn.execute("SELECT * FROM narrative_sections WHERE project_id=?", (project_id,))}
        total_words = 0
        for item in updates:
            section_id = int(item.get("id", 0))
            if section_id not in valid:
                continue
            title = str(item.get("title") or "Untitled section").strip()[:160]
            purpose = str(item.get("purpose") or "Advance the narrative.").strip()[:1000]
            target_words = max(300, min(1400, int(item.get("target_words") or 600)))
            plan = json.loads(valid[section_id]["plan_json"])
            plan.update({"title": title, "purpose": purpose, "target_words": target_words})
            conn.execute(
                "UPDATE narrative_sections SET title=?,purpose=?,target_words=?,plan_json=? WHERE id=?",
                (title, purpose, target_words, json.dumps(plan, ensure_ascii=False), section_id),
            )
            total_words += target_words
    if total_words < 3000:
        return jsonify({"error": "The narrative plan is too short."}), 400
    update_project(project_id, target_words=total_words, plan_confirmed=1, pause_requested=0,
                   error=None, status="QUEUED", current_stage="QUEUED", progress=0,
                   stage_message="Narrative plan approved. Script generation queued.")
    if not launch_job(project_id, pipeline_after_plan_review):
        update_project(project_id, plan_confirmed=0, status="PLAN_REVIEW",
                       current_stage="PLAN_REVIEW", progress=100,
                       stage_message="The previous task is still closing. Confirm again in a moment.")
        return jsonify({"error": "The previous task is still closing. Try again in a moment."}), 409
    return jsonify(serialize_project(project_id))


@app.post("/api/projects/<project_id>/pause")
def pause_project(project_id: str):
    try:
        get_project(project_id)
    except KeyError:
        abort(404)
    update_project(project_id, pause_requested=1, stage_message="Pause requested; finishing the current model call…")
    add_event(project_id, "Pause requested")
    return jsonify(serialize_project(project_id))


@app.post("/api/projects/<project_id>/resume")
def resume_project(project_id: str):
    try:
        project = get_project(project_id)
    except KeyError:
        abort(404)
    previous_status = project["status"]
    previous_stage = project["current_stage"]
    previous_message = project["stage_message"]
    update_project(project_id, pause_requested=0, error=None, status="QUEUED", current_stage="QUEUED",
                   stage_message="Resume queued from the last checkpoint.")
    if not project["structure_confirmed"]:
        target = extraction_job
    elif not project["plan_confirmed"]:
        target = pipeline_before_plan_review
    else:
        target = pipeline_after_plan_review
    started = launch_job(project_id, target)
    if not started:
        update_project(project_id, status=previous_status, current_stage=previous_stage,
                       stage_message=previous_message)
        return jsonify({"error": "This project already has a running job."}), 409
    add_event(project_id, "Processing resumed")
    return jsonify(serialize_project(project_id))


@app.get("/api/projects/<project_id>/download/<kind>")
def download_project(project_id: str, kind: str):
    try:
        project = get_project(project_id)
    except KeyError:
        abort(404)
    md_path, txt_path = export_paths(project)
    path = md_path if kind == "markdown" else txt_path if kind == "text" else None
    if path is None or not path.exists():
        abort(404)
    return send_file(path, as_attachment=True, download_name=path.name)


@app.errorhandler(413)
def too_large(_: Any):
    return jsonify({"error": "The file exceeds the 250 MB upload limit."}), 413


# ---------------------------------------------------------------------------
# Embedded interface
# ---------------------------------------------------------------------------

INDEX_HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Book Podcast Studio</title>
  <style>
    :root {
      --ink: #172235;
      --muted: #637083;
      --fog: #eef3f7;
      --paper: #ffffff;
      --line: #cfd9e2;
      --blue: #315d7a;
      --blue-deep: #23465e;
      --amber: #d99a37;
      --coral: #c95f59;
      --success: #287560;
      --shadow: 0 20px 55px rgba(32, 54, 73, .12);
      --radius: 16px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background: linear-gradient(145deg, #e9f0f5 0%, #f7f9fb 48%, #e6edf2 100%);
      font-family: "Avenir Next", Avenir, "Helvetica Neue", sans-serif;
    }
    button, input, select, textarea { font: inherit; }
    button { cursor: pointer; }
    button:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible {
      outline: 3px solid rgba(217,154,55,.45); outline-offset: 2px;
    }
    .shell { min-height: 100vh; display: grid; grid-template-columns: 330px 1fr; }
    .sidebar {
      padding: 32px 26px;
      color: #f8fbfd;
      background: var(--ink);
      position: relative;
      overflow: hidden;
    }
    .sidebar::after {
      content: ""; position: absolute; width: 260px; height: 260px; border: 1px solid rgba(255,255,255,.08);
      border-radius: 50%; bottom: -125px; right: -115px; box-shadow: 0 0 0 40px rgba(255,255,255,.025), 0 0 0 80px rgba(255,255,255,.018);
    }
    .brand { position: relative; z-index: 1; }
    .eyebrow { font: 700 11px/1.2 "SFMono-Regular", Consolas, monospace; text-transform: uppercase; letter-spacing: .16em; color: var(--amber); }
    .brand h1 { margin: 12px 0 8px; font-size: 28px; line-height: 1.02; letter-spacing: -.035em; }
    .brand p { margin: 0; color: #b9c6d1; font-size: 13px; line-height: 1.55; }
    .new-button { width: 100%; margin: 28px 0 22px; border: 1px solid rgba(255,255,255,.18); background: rgba(255,255,255,.07); color: white; padding: 12px 14px; border-radius: 10px; text-align: left; }
    .new-button:hover { background: rgba(255,255,255,.12); }
    .project-list { display: grid; gap: 8px; position: relative; z-index: 1; }
    .project-card { border: 0; width: 100%; color: #c5d0da; background: transparent; padding: 11px 12px; border-radius: 10px; text-align: left; }
    .project-card:hover, .project-card.active { background: rgba(255,255,255,.09); color: white; }
    .project-card strong { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; }
    .project-card span { display: block; margin-top: 4px; font: 10px "SFMono-Regular", monospace; text-transform: uppercase; letter-spacing: .08em; color: #8fa1b0; }
    main { padding: 42px clamp(24px, 5vw, 76px) 72px; min-width: 0; }
    .workspace { max-width: 1080px; margin: 0 auto; }
    .hero { display: flex; align-items: flex-end; justify-content: space-between; gap: 20px; margin-bottom: 26px; }
    .hero h2 { margin: 8px 0 0; font-size: clamp(34px, 5vw, 62px); line-height: .98; letter-spacing: -.055em; max-width: 760px; }
    .hero h2 em { color: var(--blue); font-style: normal; }
    .hero-note { max-width: 245px; color: var(--muted); font-size: 13px; line-height: 1.55; border-left: 2px solid var(--amber); padding-left: 14px; }
    .panel { background: rgba(255,255,255,.93); border: 1px solid rgba(207,217,226,.9); border-radius: var(--radius); box-shadow: var(--shadow); }
    .config-warning { margin: 0 0 16px; padding: 11px 13px; border: 1px solid #e6b6b3; border-radius: 9px; background: #fdf4f3; color: #8a3430; font-size: 12px; line-height: 1.5; }
    .upload-panel { padding: clamp(24px, 4vw, 44px); }
    .drop-zone { border: 1.5px dashed #9aabba; border-radius: 13px; padding: 34px; text-align: center; background: #f8fafc; transition: .2s ease; }
    .drop-zone.drag { border-color: var(--blue); background: #edf5fa; transform: translateY(-2px); }
    .drop-zone input { width: min(100%, 420px); }
    .drop-zone h3 { margin: 0 0 8px; font-size: 21px; }
    .drop-zone p { margin: 0 0 18px; color: var(--muted); font-size: 13px; }
    .form-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 18px; margin-top: 24px; }
    .field { display: grid; gap: 7px; }
    .field label { font: 700 10px "SFMono-Regular", monospace; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); }
    .field select, .field input, .field textarea { width: 100%; border: 1px solid var(--line); background: white; color: var(--ink); border-radius: 9px; padding: 11px 12px; }
    .field textarea { min-height: 74px; resize: vertical; }
    .span-2 { grid-column: span 2; }
    .primary { border: 0; background: var(--blue); color: white; padding: 12px 18px; border-radius: 9px; font-weight: 700; box-shadow: 0 8px 18px rgba(49,93,122,.22); }
    .primary:hover { background: var(--blue-deep); }
    .secondary { border: 1px solid var(--line); background: white; color: var(--ink); padding: 11px 16px; border-radius: 9px; font-weight: 650; }
    .danger { border-color: #e6b6b3; color: #8a3430; }
    .actions { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 22px; }
    .detail-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; margin-bottom: 22px; }
    .detail-head h2 { margin: 6px 0 6px; font-size: clamp(28px,4vw,46px); line-height: 1.03; letter-spacing: -.04em; }
    .detail-head p { margin: 0; color: var(--muted); }
    .status-chip { flex: 0 0 auto; color: var(--blue-deep); background: #dbe9f1; border: 1px solid #bdd2df; padding: 8px 11px; border-radius: 999px; font: 700 10px "SFMono-Regular", monospace; letter-spacing: .07em; }
    .story-spine { display: grid; grid-template-columns: repeat(8,1fr); margin: 26px 0; padding: 18px 12px 14px; background: var(--ink); border-radius: 13px; color: white; overflow: hidden; }
    .spine-step { position: relative; text-align: center; min-width: 0; }
    .spine-step::before { content: ""; position: absolute; height: 2px; background: #536173; left: -50%; right: 50%; top: 7px; }
    .spine-step:first-child::before { display: none; }
    .spine-dot { width: 15px; height: 15px; margin: 0 auto 8px; border-radius: 50%; background: #536173; border: 3px solid var(--ink); position: relative; z-index: 1; }
    .spine-step.done .spine-dot, .spine-step.active .spine-dot { background: var(--amber); }
    .spine-step.done::before, .spine-step.active::before { background: var(--amber); }
    .spine-label { display: block; font: 9px "SFMono-Regular", monospace; color: #95a5b4; overflow: hidden; text-overflow: ellipsis; }
    .spine-step.active .spine-label { color: white; }
    .progress-wrap { margin: 20px 0; }
    .progress-meta { display: flex; justify-content: space-between; gap: 15px; color: var(--muted); font-size: 13px; margin-bottom: 8px; }
    .progress-track { height: 9px; background: #dce4eb; overflow: hidden; border-radius: 999px; }
    .progress-bar { height: 100%; background: linear-gradient(90deg,var(--blue),#5785a2,var(--amber)); transition: width .5s ease; }
    .content-panel { padding: 24px; margin-top: 18px; }
    .content-panel h3 { margin: 0 0 6px; font-size: 20px; }
    .content-panel > p { margin: 0 0 18px; color: var(--muted); font-size: 13px; }
    .review-list { display: grid; gap: 9px; max-height: 520px; overflow: auto; padding-right: 5px; }
    .review-row { display: grid; grid-template-columns: 28px 1fr 90px; gap: 10px; align-items: center; padding: 10px; background: var(--fog); border: 1px solid #dce4ea; border-radius: 9px; }
    .review-row input[type="text"], .review-row input[type="number"], .review-row textarea { width: 100%; border: 1px solid transparent; background: white; border-radius: 7px; padding: 8px; color: var(--ink); }
    .plan-row { grid-template-columns: 38px minmax(150px,.7fr) minmax(260px,1.5fr) 96px; align-items: start; }
    .plan-row textarea { min-height: 64px; }
    .index-num { font: 700 11px "SFMono-Regular", monospace; color: var(--blue); padding-top: 9px; }
    .event-list { margin: 0; padding: 0; list-style: none; display: grid; gap: 7px; }
    .event-list li { display: grid; grid-template-columns: 58px 1fr; gap: 10px; font-size: 12px; color: var(--muted); }
    .event-list b { font: 700 9px "SFMono-Regular", monospace; padding-top: 2px; color: var(--blue); }
    .error-box { margin-top: 18px; padding: 14px; border-left: 4px solid var(--coral); background: #fff0ef; color: #782f2b; white-space: pre-wrap; font-size: 13px; }
    .empty { padding: 48px; text-align: center; color: var(--muted); }
    .toast { position: fixed; right: 24px; bottom: 24px; max-width: 380px; padding: 13px 16px; color: white; background: var(--ink); border-radius: 10px; box-shadow: var(--shadow); z-index: 10; transform: translateY(120px); opacity: 0; transition: .25s ease; }
    .toast.show { transform: none; opacity: 1; }
    .toast.error { background: #7c3531; }
    .small { font-size: 12px; color: var(--muted); }
    @media (max-width: 850px) {
      .shell { grid-template-columns: 1fr; }
      .sidebar { padding: 22px; }
      .project-list { grid-template-columns: repeat(auto-fit,minmax(170px,1fr)); }
      main { padding: 28px 18px 60px; }
      .hero { align-items: flex-start; flex-direction: column; }
      .hero-note { max-width: none; }
      .story-spine { grid-template-columns: repeat(4,1fr); row-gap: 14px; }
      .spine-step:nth-child(5)::before { display: none; }
      .form-grid { grid-template-columns: 1fr; }
      .span-2 { grid-column: auto; }
      .plan-row { grid-template-columns: 32px 1fr; }
      .plan-row > *:nth-child(n+3) { grid-column: 2; }
    }
    @media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; transition: none !important; } }
  </style>
</head>
<body>
<div class="shell">
  <aside class="sidebar">
    <div class="brand">
      <div class="eyebrow">Local editorial engine</div>
      <h1>Book Podcast<br>Studio</h1>
      <p>Evidence first. Story second. Nothing leaves this machine.</p>
      <p style="margin-top:8px;font:10px 'SFMono-Regular',monospace;color:#7f93a4">Version {{ app_version }}</p>
    </div>
    <button class="new-button" id="newProject">＋ Start a new book</button>
    <div class="project-list" id="projectList"></div>
  </aside>
  <main><div class="workspace" id="workspace"></div></main>
</div>
<div class="toast" id="toast" role="status" aria-live="polite"></div>

<script>
const CSRF = {{ csrf_token|tojson }};
const MODELS = {{ models|tojson }};
const DEFAULTS = {{ defaults|tojson }};
const CONFIG_WARNINGS = {{ config_warnings|tojson }};
const STAGES = ["UPLOADED","EXTRACTING","ANALYZING","PLANNING","DRAFTING","FACT_CHECKING","POLISHING","COMPLETED"];
let activeId = null;
let pollTimer = null;

const $ = (selector, root=document) => root.querySelector(selector);
const escapeHTML = value => String(value ?? "").replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
const modelOptions = selected => MODELS.map(m => `<option value="${escapeHTML(m.path)}" ${m.path===selected?'selected':''}>${escapeHTML(m.name)}</option>`).join('');

function toast(message, isError=false) {
  const node = $('#toast'); node.textContent = message; node.className = `toast show ${isError?'error':''}`;
  setTimeout(() => node.className = 'toast', 3600);
}

async function api(url, options={}) {
  const response = await fetch(url, { ...options, headers: {"X-CSRF-Token": CSRF, ...(options.headers||{})} });
  let data = {}; try { data = await response.json(); } catch (_) {}
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

function uploadView() {
  activeId = null; clearTimeout(pollTimer);
  $('#workspace').innerHTML = `
    <section class="hero">
      <div><div class="eyebrow">New editorial project</div><h2>Turn a whole book into a <em>story worth hearing.</em></h2></div>
      <div class="hero-note">The book is read in small evidence packets. A structured memory carries its argument—not an oversized context window.</div>
    </section>
    <form class="panel upload-panel" id="uploadForm">
      ${CONFIG_WARNINGS.length ? `<div class="config-warning">${CONFIG_WARNINGS.map(w => `${escapeHTML(w.slot)}: ${escapeHTML(w.path)} is not a readable .gguf file`).join('<br>')}</div>` : ''}
      <div class="drop-zone" id="dropZone">
        <h3>Choose an English PDF or EPUB</h3>
        <p>Searchable PDFs only · up to 250 MB · processed locally</p>
        <input type="file" name="book" accept=".pdf,.epub" required>
      </div>
      <div class="form-grid">
        <div class="field"><label>Episode length</label><select name="duration"><option value="50">50 minutes · ~6,800 words</option><option value="60" selected>60 minutes · ~8,000 words</option><option value="70">70 minutes · ~9,300 words</option></select></div>
        <div class="field"><label>Context budget</label><input value="8,192 tokens · portable mode" disabled></div>
        <div class="field"><label>Analysis model</label><select name="analysis_model">${modelOptions(DEFAULTS.analysis)}</select></div>
        <div class="field"><label>Writing model</label><select name="writing_model">${modelOptions(DEFAULTS.writing)}</select></div>
        <div class="field span-2"><label>Fast fallback</label><select name="fast_model">${modelOptions(DEFAULTS.fast)}</select></div>
      </div>
      <div class="actions"><button class="primary" type="submit">Read and detect chapters</button><span class="small">You will review the structure before any long model run.</span></div>
    </form>`;
  const form = $('#uploadForm');
  form.addEventListener('submit', submitUpload);
  const zone = $('#dropZone');
  ['dragenter','dragover'].forEach(name => zone.addEventListener(name, e => { e.preventDefault(); zone.classList.add('drag'); }));
  ['dragleave','drop'].forEach(name => zone.addEventListener(name, e => { e.preventDefault(); zone.classList.remove('drag'); }));
  zone.addEventListener('drop', e => { if (e.dataTransfer.files.length) form.book.files = e.dataTransfer.files; });
}

async function submitUpload(event) {
  event.preventDefault();
  const button = event.target.querySelector('button[type=submit]'); button.disabled = true; button.textContent = 'Reading book…';
  try {
    const project = await api('/api/projects', {method:'POST', body:new FormData(event.target)});
    activeId = project.id; await loadProjects(); renderProject(project); schedulePoll(project);
  } catch (error) { toast(error.message, true); button.disabled=false; button.textContent='Read and detect chapters'; }
}

async function loadProjects() {
  const projects = await api('/api/projects');
  $('#projectList').innerHTML = projects.length ? projects.map(p => `
    <button class="project-card ${p.id===activeId?'active':''}" data-id="${p.id}"><strong>${escapeHTML(p.title)}</strong><span>${escapeHTML(p.status.replaceAll('_',' '))}</span></button>`).join('') : '<div class="small">No books yet.</div>';
  document.querySelectorAll('.project-card').forEach(button => button.onclick = () => selectProject(button.dataset.id));
}

async function selectProject(id) {
  activeId = id; clearTimeout(pollTimer);
  try { const project = await api(`/api/projects/${id}`); renderProject(project); schedulePoll(project); await loadProjects(); }
  catch (error) { toast(error.message, true); }
}

function stagePosition(project) {
  const status = project.status;
  if (status==='STRUCTURE_REVIEW') return 1;
  if (status==='PLAN_REVIEW' || status==='ANALYZED') return 3;
  if (status==='DRAFTED') return 4;
  if (status==='FACT_CHECKED') return 5;
  if (status==='PAUSED' || status==='FAILED' || status==='QUEUED') return Math.max(0, STAGES.indexOf(project.current_stage));
  return Math.max(0, STAGES.indexOf(status));
}

function spine(project) {
  const current = stagePosition(project);
  const labels = ['Upload','Extract','Analyze','Plan','Draft','Check','Polish','Ready'];
  return `<div class="story-spine" aria-label="Project progress">${labels.map((label,i)=>`<div class="spine-step ${i<current?'done':i===current?'active':''}"><div class="spine-dot"></div><span class="spine-label">${label}</span></div>`).join('')}</div>`;
}

function renderProject(project) {
  activeId = project.id;
  const busy = ['EXTRACTING','QUEUED','ANALYZING','ANALYZED','PLANNING','DRAFTING','DRAFTED','FACT_CHECKING','FACT_CHECKED','POLISHING'].includes(project.status);
  let body = '';
  if (project.status === 'STRUCTURE_REVIEW') body = structureReview(project);
  else if (project.status === 'PLAN_REVIEW') body = planReview(project);
  else if (project.status === 'COMPLETED') body = completedView(project);
  else if (project.status === 'FAILED' || project.status === 'PAUSED') body = recoveryView(project);
  else body = runningView(project);
  $('#workspace').innerHTML = `
    <section class="detail-head"><div><div class="eyebrow">${escapeHTML(project.original_filename)}</div><h2>${escapeHTML(project.title)}</h2><p>${project.duration_minutes} minute target · ${Number(project.target_words).toLocaleString()} words</p></div><span class="status-chip">${escapeHTML(project.status.replaceAll('_',' '))}</span></section>
    ${spine(project)}
    <section class="panel content-panel">
      <div class="progress-wrap"><div class="progress-meta"><span>${escapeHTML(project.stage_message)}</span><b>${Math.round(project.progress)}%</b></div><div class="progress-track"><div class="progress-bar" style="width:${Math.max(0,Math.min(100,project.progress))}%"></div></div></div>
      ${project.error ? `<div class="error-box">${escapeHTML(project.error)}</div>` : ''}
      ${body}
    </section>
    ${eventView(project.events)}`;
  bindProjectActions(project);
  if (busy) schedulePoll(project);
}

function runningView(project) {
  return `<h3>The editorial engine is working</h3><p>Closing this browser tab will not erase completed checkpoints. Keep the Terminal process running.</p><div class="actions"><button class="secondary danger" id="pauseBtn">Pause after current model call</button></div>`;
}

function structureReview(project) {
  return `<h3>Review the detected book structure</h3><p>Rename sections and exclude indexes, acknowledgements, or other material that should not shape the episode.</p>
    <div class="review-list" id="chapterList">${project.chapters.map((c,i)=>`<div class="review-row" data-id="${c.id}"><input type="checkbox" ${c.included?'checked':''} aria-label="Include section"><input type="text" value="${escapeHTML(c.title)}" aria-label="Section title"><span class="small">${Number(c.token_count).toLocaleString()} tok</span></div>`).join('')}</div>
    <div class="actions"><button class="primary" id="confirmStructure">Confirm and analyze</button><span class="small">Qwen will run after confirmation.</span></div>`;
}

function planReview(project) {
  const sum = project.sections.reduce((total,s)=>total+s.target_words,0);
  return `<h3>Review the story spine</h3><p>Adjust titles, purposes, and word budgets. The total currently equals <b id="planTotal">${sum.toLocaleString()}</b> words.</p>
    <div class="review-list" id="planList">${project.sections.map((s,i)=>`<div class="review-row plan-row" data-id="${s.id}"><span class="index-num">${String(i+1).padStart(2,'0')}</span><input type="text" value="${escapeHTML(s.title)}" aria-label="Section title"><textarea aria-label="Section purpose">${escapeHTML(s.purpose)}</textarea><input type="number" min="300" max="1400" step="50" value="${s.target_words}" aria-label="Target words"></div>`).join('')}</div>
    <div class="actions"><button class="primary" id="confirmPlan">Approve plan and write script</button><span class="small">Gemma writes, Qwen checks, then Gemma revises.</span></div>`;
}

function completedView(project) {
  return `<h3>Your documentary script is ready</h3><p>The final sections passed an evidence audit and targeted narrative revision. Keep the database if you may want to resume or inspect the project later.</p><div class="actions"><a class="primary" href="/api/projects/${project.id}/download/markdown">Download Markdown</a><a class="secondary" href="/api/projects/${project.id}/download/text">Download TXT</a></div>`;
}

function recoveryView(project) {
  return `<h3>${project.status==='PAUSED'?'Processing is paused':'Processing stopped at a checkpoint'}</h3><p>Resume continues from completed chunks or sections; it does not restart the book.</p><div class="actions"><button class="primary" id="resumeBtn">Resume processing</button></div>`;
}

function eventView(events=[]) {
  if (!events.length) return '';
  return `<section class="panel content-panel"><h3>Editorial log</h3><p>Recent local processing events.</p><ul class="event-list">${events.slice(0,12).map(e=>`<li><b>${escapeHTML(e.level)}</b><span>${escapeHTML(e.message)}</span></li>`).join('')}</ul></section>`;
}

function bindProjectActions(project) {
  const pause = $('#pauseBtn'); if (pause) pause.onclick = () => postAction(`/api/projects/${project.id}/pause`);
  const resume = $('#resumeBtn'); if (resume) resume.onclick = () => postAction(`/api/projects/${project.id}/resume`);
  const confirmStructure = $('#confirmStructure'); if (confirmStructure) confirmStructure.onclick = async () => {
    const chapters = [...document.querySelectorAll('#chapterList .review-row')].map(row => ({id:Number(row.dataset.id), included:row.querySelector('input[type=checkbox]').checked, title:row.querySelector('input[type=text]').value}));
    await postAction(`/api/projects/${project.id}/confirm-structure`, {chapters});
  };
  const inputs = document.querySelectorAll('#planList input[type=number]'); inputs.forEach(input => input.oninput = updatePlanTotal);
  const confirmPlan = $('#confirmPlan'); if (confirmPlan) confirmPlan.onclick = async () => {
    const sections = [...document.querySelectorAll('#planList .plan-row')].map(row => ({id:Number(row.dataset.id), title:row.querySelector('input[type=text]').value, purpose:row.querySelector('textarea').value, target_words:Number(row.querySelector('input[type=number]').value)}));
    await postAction(`/api/projects/${project.id}/confirm-plan`, {sections});
  };
}

function updatePlanTotal() {
  const total = [...document.querySelectorAll('#planList input[type=number]')].reduce((sum,input)=>sum+(Number(input.value)||0),0);
  const node = $('#planTotal'); if (node) node.textContent = total.toLocaleString();
}

async function postAction(url, payload) {
  try {
    const options = {method:'POST'};
    if (payload) { options.headers={'Content-Type':'application/json'}; options.body=JSON.stringify(payload); }
    const project = await api(url, options); renderProject(project); await loadProjects();
  } catch (error) { toast(error.message, true); }
}

function schedulePoll(project) {
  clearTimeout(pollTimer);
  if (!activeId || ['STRUCTURE_REVIEW','PLAN_REVIEW','COMPLETED','PAUSED','FAILED'].includes(project.status)) return;
  pollTimer = setTimeout(async () => {
    if (!activeId) return;
    try { const latest = await api(`/api/projects/${activeId}`); renderProject(latest); await loadProjects(); }
    catch (error) { toast(error.message, true); }
  }, 2500);
}

$('#newProject').onclick = uploadView;
uploadView();
loadProjects().catch(error => toast(error.message,true));
</script>
</body>
</html>'''


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def worker_main(stage: str, project_id: str) -> int:
    init_db()
    function = STAGE_FUNCTIONS.get(stage)
    if function is None:
        print(f"Unknown stage: {stage}", file=sys.stderr)
        return 2
    try:
        return function(project_id)
    except Exception as exc:
        traceback.print_exc()
        update_project(project_id, status="FAILED", current_stage="FAILED", progress=0,
                       stage_message=f"{stage.title()} failed. Review the error and resume.", error=str(exc)[:4000])
        add_event(project_id, f"Stage {stage} failed: {exc}", "ERROR")
        return 1


if __name__ == "__main__":
    init_db()
    if len(sys.argv) == 4 and sys.argv[1] == "--stage":
        raise SystemExit(worker_main(sys.argv[2], sys.argv[3]))
    recover_interrupted_projects()
    print(f"\nBook Podcast Studio {APP_VERSION}")
    print("Open http://127.0.0.1:5000 in your browser.\n")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False)
