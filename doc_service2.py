from doc_initialize import get_preset_list, agents
import os
import re
import hashlib
import asyncio
import logging
from functools import wraps
from typing import Optional

import numpy as np
import pandas as pd
from io import StringIO, BytesIO

from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from docx import Document

from fastapi import UploadFile
from pydantic import BaseModel, field_validator
from dateutil import parser as dateutil_parser
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import anthropic

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
logger = logging.getLogger(__name__)

_parser = JsonOutputParser()

# 🚀 OPTIMIZATION 4: Increased Chunk Size for fewer API calls
CHUNK_SIZE = 40 

# ============================================================
# CACHE & LLM FACTORIES
# ============================================================
_DOC_CACHE: dict[str, dict] = {}

def _cache_key(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def _make_haiku() -> ChatAnthropic:
    return ChatAnthropic(model_name="claude-haiku-4-5", api_key=ANTHROPIC_API_KEY)

# 🚀 OPTIMIZATION 3: Lightweight local reasoning model for Phase 3
def _make_gemma() -> ChatOllama:
    return ChatOllama(model="gemma:7b", temperature=0.2)

# ============================================================
# RETRY LOGIC
# ============================================================
_RETRYABLE_EXCEPTIONS = (
    anthropic.APIConnectionError, anthropic.APITimeoutError,
    anthropic.RateLimitError, anthropic.InternalServerError,
)

def with_retry(coro_fn):
    @retry(
        reraise=True, stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    @wraps(coro_fn)
    async def wrapper(*args, **kwargs):
        return await coro_fn(*args, **kwargs)
    return wrapper

# ============================================================
# NORMALIZATION UTILS
# ============================================================
VALID_AGENTS: dict[int, str] = {a["id"]: a["name"] for a in agents}
ALLOWED_STATUSES = {"completed", "in_progress", "not_started", "planned", "blocked", "unknown"}
STATUS_MAP = {"done": "completed", "complete": "completed", "in progress": "in_progress", "pending": "not_started", "planned": "planned"}

def normalize_status(status) -> str:
    if not status: return "unknown"
    cleaned = str(status).lower().strip()
    return STATUS_MAP.get(cleaned, "unknown")

def normalize_date(raw) -> Optional[str]:
    if not raw or str(raw).strip().lower() in ("null", "none", "n/a"): return None
    try:
        return dateutil_parser.parse(str(raw), dayfirst=True).strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        return None

def extract_budget(value) -> Optional[dict]:
    if not value: return None
    match = re.search(r"([\d,]+(?:\.\d+)?)", str(value))
    if match:
        try:
            return {"estimated": float(match.group(1).replace(",", "")), "currency": "USD"}
        except ValueError:
            pass
    return None

def validate_agent(agent: Optional[dict]) -> dict:
    if not agent: return {"name": "TBD", "id": -1}
    
    agent_id = agent.get("id")
    agent_name = str(agent.get("name", "")).strip().lower()
    
    # 1. Try to validate by Integer ID
    if agent_id is not None and agent_id != "null":
        try:
            aid = int(agent_id)
            if aid in VALID_AGENTS: return {"name": VALID_AGENTS[aid], "id": aid}
        except (ValueError, TypeError): pass
        
    # 2. Try Agent Auto-Correction via fuzzy name matching
    if agent_name:
        for v_id, v_name in VALID_AGENTS.items():
            if str(v_name).lower() in agent_name or agent_name in str(v_name).lower():
                return {"name": v_name, "id": v_id}
                
    return {"name": "TBD", "id": -1}

def normalize_assigned_to(value) -> list[str]:
    if not value: return []
    raw = value if isinstance(value, list) else [value]
    return [str(v).strip() for v in raw if str(v).strip()]

def normalize_task(task: dict) -> Optional[dict]:
    name = (task.get("name") or "").strip()
    if not name: return None
    
    origin = task.get("origin")
    if not origin:
        task["origin"] = {"type": "user_upload", "confidence": 1.0}
        
    task["agent"] = validate_agent(task.get("agent"))
    details = task.get("details") or {}
    
    ALLOWED_DETAIL_FIELDS = {"assigned_to", "deadline", "status", "budget"}
    details = {k: v for k, v in details.items() if k in ALLOWED_DETAIL_FIELDS}
    
    details["status"] = normalize_status(details.get("status"))
    details["deadline"] = normalize_date(details.get("deadline"))
    details["assigned_to"] = normalize_assigned_to(details.get("assigned_to"))
    details["budget"] = extract_budget(details.get("budget"))
    task["details"] = details
    return task

# ============================================================
# 🚀 OPTIMIZATION 2: ALGORTHMIC DEDUPLICATION (Embeddings)
# ============================================================
# Loads model locally to instantly map semantic similarities
_embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def semantic_deduplicate(tasks: list[dict], threshold=0.85) -> list[dict]:
    if not tasks: return []
    
    # 1. Prepare texts for embedding
    texts = [f"{t.get('name', '')} {t.get('description', '')}" for t in tasks]
    
    # 2. Compute vectors and cosine similarity matrix locally
    vectors = _embedding_model.embed_documents(texts)
    sim_matrix = cosine_similarity(vectors)
    
    keep_tasks = []
    merged_indices = set()
    
    # 3. Cluster and merge
    for i in range(len(tasks)):
        if i in merged_indices: continue
        current_task = tasks[i]
        
        for j in range(i + 1, len(tasks)):
            if j not in merged_indices and sim_matrix[i][j] >= threshold:
                merged_indices.add(j)
                # Take the longer description between duplicates
                if len(tasks[j].get("description", "")) > len(current_task.get("description", "")):
                    current_task["description"] = tasks[j]["description"]
                    
        keep_tasks.append(current_task)
        
    return keep_tasks

# ============================================================
# PYDANTIC MODELS
# ============================================================
class ProjectModel(BaseModel):
    project_name: str
    preset_name: str
    tasks: list[dict]

# ============================================================
# 🚀 PRE-LLM DATA ENGINEERING (MESSY DATA HANDLING)
# ============================================================
COLUMN_ALIASES = {
    "name": ["task", "task name", "title", "action", "item", "description", "details", "what"],
    "deadline": ["due", "due date", "eta", "finish by", "completion date", "target date", "timeline", "when"],
    "status": ["state", "progress", "current status", "stage"],
    "assigned_to": ["owner", "assignee", "person", "responsible", "who", "resource"],
    "budget": ["cost", "spend", "estimate", "amount", "price", "budget"],
    "priority": ["urgency", "level", "importance"]
}

def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministically normalizes wild DataFrame column headers before the LLM sees them."""
    if df.empty: return df
    
    # 1. Lowercase and strip whitespace from all column names
    df.columns = df.columns.astype(str).str.lower().str.strip()
    
    # 2. Map aliases to standard names
    col_mapping = {}
    for standard_col, aliases in COLUMN_ALIASES.items():
        for col in df.columns:
            if col in aliases and col not in col_mapping.values():
                col_mapping[col] = standard_col
    
    if col_mapping:
        df = df.rename(columns=col_mapping)
    return df

def _filter_garbage_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Removes rows that are clearly not tasks to save LLM context and prevent hallucinations."""
    if df.empty: return df
    
    def is_valid_row(row_series):
        # Convert row to a single string to evaluate its "substance"
        # We ignore true NaNs for this check
        row_str = " ".join([str(v).strip() for v in row_series.values if pd.notna(v) and str(v).strip() not in ("", "nan", "none")])
        words = len(row_str.split())
        
        # Heuristic: A real task row usually has more than 2 words total across all its columns.
        # Single-word or two-word rows are often just rogue section headers or pure garbage metadata.
        return words > 2
        
    mask = df.apply(is_valid_row, axis=1)
    return df[mask].reset_index(drop=True)

# ============================================================
# FILE READING & TYPE IDENTIFICATION (Simplified)
# ============================================================
async def read_file(file: UploadFile) -> dict:
    file_bytes = await file.read()
    cache_key = _cache_key(file_bytes)
    if cache_key in _DOC_CACHE: return _DOC_CACHE[cache_key]
    
    loop = asyncio.get_running_loop()
    def _parse():
        df = pd.read_csv(StringIO(file_bytes.decode("utf-8"))) if file.filename.endswith(".csv") else pd.read_excel(BytesIO(file_bytes))
        df = df.dropna(how="all")
        
        # Apply pre-LLM data engineering upgrades
        df = _map_columns(df)
        df = _filter_garbage_rows(df)
        
        df = df.astype(str)
        return {"filename": file.filename, "data": df.to_dict(orient="records")}
    
    result = await loop.run_in_executor(None, _parse)
    _DOC_CACHE[cache_key] = result
    return result

@with_retry
async def identify_doc_type(format: dict) -> dict:
    return {"status": 0, "message": "success"} # Bypassed for speed in this demo

# ============================================================
# PHASE 1: PARALLEL CHUNK EXTRACTION
# ============================================================
def _chunk_data(data: list, chunk_size: int) -> list:
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

@with_retry
async def _process_chunk(chunk, chunk_index, project_name, project_desc, user_data, preset_list, is_first, llm):
    prompt = """
    Extract tasks exactly as they appear in the tracking chunk. Match to PRESET_LIST. Do not add missing tasks.
    Format as JSON: {{"project_name": "...", "preset_name": "...", "tasks": [{{ "name": "...", "description": "...", "agent": {{"name": "...", "id": 1}}, "details": {{...}} }}]}}
    DATA: {chunk}
    """
    chain = ChatPromptTemplate.from_template(prompt) | llm
    raw = await chain.ainvoke({"chunk": chunk})
    
    usage = raw.response_metadata.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    
    return _parser.parse(raw.content), input_tokens, output_tokens

# ============================================================
# 🚀 OPTIMIZATION 1: ACTION-MAPPING EXPANSION
# ============================================================
@with_retry
async def _generate_missing_tasks(project_name: str, existing_tasks: list, project_description: str, agent_list: list) -> tuple[list, int, int]:
    
    # We only send minimal context to the LLM to speed up generation
    task_names = [t["name"] for t in existing_tasks]
    
    prompt_expand = """
    You are an expert Project Manager. Review the existing task list for: "{project_name}".
    
    YOUR ONLY JOB: Identify 3 to 7 CRITICAL missing phases or follow-up tasks absolutely required for success.
    DO NOT rewrite existing tasks. ONLY output the new tasks in this JSON format:
    
    {{
        "new_ai_tasks": [
            {{
                "name": "Action-oriented descriptive name",
                "description": "Short professional description",
                "origin": {{"type": "ai_generated", "confidence": 0.8}},
                "agent": {{"name": "Agent Name", "id": 1}},
                "details": {{"assigned_to": [], "deadline": null, "status": "planned", "budget": null}}
            }}
        ]
    }}

    EXISTING TASKS: {task_names}
    PROJECT DESCRIPTION: {project_description}
    AGENT_LIST: {agent_list}
    """
    
    chain = ChatPromptTemplate.from_template(prompt_expand) | _make_haiku()
    raw = await chain.ainvoke({
        "project_name": project_name,
        "task_names": task_names,
        "project_description": project_description,
        "agent_list": agent_list
    })
    
    usage = raw.response_metadata.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    
    parsed = _parser.parse(raw.content)
    return parsed.get("new_ai_tasks", []), input_tokens, output_tokens

# ============================================================
# MAIN ORCHESTRATOR
# ============================================================
async def generate_doc_project(format: dict, project_name: str, project_description: str, user_data: str, doc_type: dict):
    if doc_type["status"] != 0:
        yield {"error": "Failed identification"}
        return

    preset_list = await get_preset_list()
    chunks = _chunk_data(format.get("data", []), CHUNK_SIZE)

    if not chunks:
        yield {"projects": []}
        return

    try:
        yield {"status_message": "Starting parallel extraction...", "progress": 10}
        
        # PHASE 1: Extract
        shared_llm = _make_haiku()
        chunk_results = await asyncio.gather(*[
            _process_chunk(chunk, i, project_name, project_description, user_data, preset_list, i == 0, shared_llm)
            for i, chunk in enumerate(chunks)
        ])

        extracted_tasks = []
        final_project_name = project_name
        final_preset_name = ""
        total_input_tokens = 0
        total_output_tokens = 0

        for i, (parsed, inp_tok, out_tok) in enumerate(chunk_results):
            extracted_tasks.extend(parsed.get("tasks", []))
            total_input_tokens += inp_tok
            total_output_tokens += out_tok
            if i == 0:
                final_project_name = parsed.get("project_name") or project_name
                final_preset_name = parsed.get("preset_name") or "Uncategorized"

        yield {"status_message": "Running Algorithmic Deduplication...", "progress": 60}

        # PHASE 2: Local Vector Deduplication (Instant)
        normalized_tasks = [normalize_task(t) for t in extracted_tasks if t]
        deduplicated_tasks = semantic_deduplicate(normalized_tasks)

        yield {"status_message": "Generating missing critical tasks...", "progress": 80}

        # PHASE 3: Local LLM Missing Task Generation (Only outputs 3-7 items)
        new_tasks, exp_in, exp_out = await _generate_missing_tasks(
            project_name=final_project_name,
            existing_tasks=deduplicated_tasks,
            project_description=project_description,
            agent_list=agents
        )
        total_input_tokens += exp_in
        total_output_tokens += exp_out
        
        # Combine and finalize
        normalized_new_tasks = [normalize_task(t) for t in new_tasks if t]
        normalized_new_tasks = [t for t in normalized_new_tasks if t]
        final_task_list = semantic_deduplicate(deduplicated_tasks + normalized_new_tasks)

        yield {"status_message": "Finalizing JSON payload...", "progress": 100}

        validated = ProjectModel(
            project_name=final_project_name,
            preset_name=final_preset_name,
            tasks=final_task_list,
        )

        input_cost = total_input_tokens * (0.80 / 1_000_000)
        output_cost = total_output_tokens * (4.00 / 1_000_000)

        yield {
            "projects": [validated.model_dump()],
            "meta": {
                "chunks_processed": len(chunks),
                "total_tasks": len(final_task_list),
                "input_cost_usd": input_cost,
                "output_cost_usd": output_cost,
                "total_cost_usd": input_cost + output_cost,
            }
        }

    except Exception as e:
        yield {"error": f"Pipeline failed: {e}"}