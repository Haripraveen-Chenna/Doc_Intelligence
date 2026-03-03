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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from dotenv import load_dotenv

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
CHUNK_SIZE = 40 

_DOC_CACHE: dict[str, dict] = {}
_embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def _cache_key(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def _make_haiku() -> ChatAnthropic:
    return ChatAnthropic(model_name="claude-haiku-4-5", api_key=ANTHROPIC_API_KEY)

def _make_sonnet() -> ChatAnthropic:
    return ChatAnthropic(model_name="claude-sonnet-4-5", api_key=ANTHROPIC_API_KEY, max_tokens_to_sample=5000)

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
    
    if agent_id is not None and agent_id != "null":
        try:
            aid = int(agent_id)
            if aid in VALID_AGENTS: return {"name": VALID_AGENTS[aid], "id": aid}
        except (ValueError, TypeError): pass
        
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
    
    ALLOWED_DETAIL_FIELDS = {"assigned_to", "deadline", "status", "budget", "workstream"}
    details = {k: v for k, v in details.items() if k in ALLOWED_DETAIL_FIELDS}
    
    details["status"] = normalize_status(details.get("status"))
    details["deadline"] = normalize_date(details.get("deadline"))
    details["assigned_to"] = normalize_assigned_to(details.get("assigned_to"))
    details["budget"] = extract_budget(details.get("budget"))
    task["details"] = details
    return task

class OriginModel(BaseModel):
    type: str
    confidence: float

class AgentModel(BaseModel):
    name: str
    id: int

class TaskDetails(BaseModel):
    assigned_to: list[str] = []
    deadline: Optional[str] = None
    estimated_duration_days: int = 0
    status: str = "planned"
    budget_estimate: Optional[float] = None
    requires_approval: bool = False
    automation_possible: bool = False

class TaskModel(BaseModel):
    name: str
    description: str = ""
    phase: str = ""
    priority: str = "medium"
    dependencies: list[str] = []
    origin: OriginModel
    agent: Optional[AgentModel] = None
    details: TaskDetails

class PhaseModel(BaseModel):
    phase_name: str
    phase_order: int

class RiskModel(BaseModel):
    risk: str
    impact: str
    mitigation: str

class ProjectModel(BaseModel):
    project_name: str = ""
    project_type: str = ""
    project_summary: str = ""
    project_start: str = ""
    projected_end: str = ""
    event_date: Optional[str] = None
    location: str = ""
    booth_size: str = ""
    expected_attendance: int = 0
    crm_system: str = ""
    preset_name: str = ""
    phases: list[PhaseModel] = []
    tasks: list[TaskModel] = []
    risks: list[RiskModel] = []
    success_metrics: list[str] = []
    estimated_total_budget: Optional[float] = None

def _audit_and_fix_project(project: dict) -> dict:
    """Post-LLM Python validator that enforces structural and temporal integrity."""
    tasks = project.get("tasks", [])
    valid_task_names = set(str(t.get("name", "")) for t in tasks if t.get("name"))
    
    latest_deadline = project.get("project_start", "")
    event_date = project.get("event_date")

    for t in tasks:
        # Integrity Layer 1: Ghost Dependencies
        # Ensure no task depends on a task that does not exist in the array
        deps = t.get("dependencies", [])
        if isinstance(deps, list):
            t["dependencies"] = [d for d in deps if d in valid_task_names]

        # Tracking max temporal horizon
        deadline = t.get("details", {}).get("deadline")
        if deadline and deadline > latest_deadline:
            latest_deadline = deadline

    # Integrity Layer 2: Time Travel Prevention
    # Ensure the projected_end doesn't end before the last task or the event itself (cross-year bug)
    proj_end = project.get("projected_end", "")
    if latest_deadline and proj_end and latest_deadline > proj_end:
        project["projected_end"] = latest_deadline

    if event_date and project.get("projected_end", "") < event_date:
        project["projected_end"] = event_date

    project["tasks"] = tasks
    return project

# ============================================================
# 🚀 EXPERIMENTAL PRE-LLM: NLP GARBAGE CLASSIFICATION
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
    if df.empty: return df
    df.columns = df.columns.astype(str).str.lower().str.strip()
    col_mapping = {}
    for standard_col, aliases in COLUMN_ALIASES.items():
        for col in df.columns:
            if col in aliases and col not in col_mapping.values():
                col_mapping[col] = standard_col
    if col_mapping:
        df = df.rename(columns=col_mapping)
    return df

def _filter_garbage_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Uses semantic thresholding to drop non-task rows purely locally."""
    if df.empty: return df
    
    # We will score every row against a semantic hypothetical task representation
    # to filter out random metadata or short section headers
    task_hypothesis = "Actionable project task sequence assignment deadline status"
    hypothesis_vector = _embedding_model.embed_query(task_hypothesis)
    
    def is_valid_task_row(row_series):
        row_str = " ".join([str(v).strip() for v in row_series.values if pd.notna(v) and str(v).strip() not in ("", "nan", "none")])
        if len(row_str.split()) < 3: return False
        
        # Local Semantic Row Classification Check
        row_vector = _embedding_model.embed_query(row_str)
        similarity = cosine_similarity([row_vector], [hypothesis_vector])[0][0]
        
        # If it's somewhat semantically related to work/tasks, keep it
        return similarity > 0.15
        
    mask = df.apply(is_valid_task_row, axis=1)
    return df[mask].reset_index(drop=True)

# ============================================================
# FILE READING
# ============================================================
async def read_file(file: UploadFile) -> dict:
    file_bytes = await file.read()
    cache_key = _cache_key(file_bytes)
    if cache_key in _DOC_CACHE: return _DOC_CACHE[cache_key]
    
    loop = asyncio.get_running_loop()
    def _parse():
        df = pd.read_csv(StringIO(file_bytes.decode("utf-8"))) if file.filename.endswith(".csv") else pd.read_excel(BytesIO(file_bytes))
        df = df.dropna(how="all")
        df = _map_columns(df)
        df = _filter_garbage_rows(df)
        df = df.astype(str)
        return {"filename": file.filename, "data": df.to_dict(orient="records")}
    
    result = await loop.run_in_executor(None, _parse)
    _DOC_CACHE[cache_key] = result
    return result

@with_retry
async def identify_doc_type(format: dict) -> dict:
    return {"status": 0, "message": "success"}

# ============================================================
# PHASE 1: CHUNKING
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
# 🚀 EXPERIMENTAL HIERARCHICAL CLUSTERING FOR DEDUPLICATION
# ============================================================
def advanced_semantic_cluster(tasks: list[dict], distance_threshold=0.25) -> list[dict]:
    """Uses Agglomerative Clustering to merge O(N^2) duplicates into distinct groupings locally."""
    if not tasks: return []
    if len(tasks) == 1: return tasks
    
    texts = [f"{t.get('name', '')} {t.get('description', '')}" for t in tasks]
    vectors = _embedding_model.embed_documents(texts)
    
    # We use Cosine Distance for clustering
    clustering_model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='average'
    )
    clustering_model.fit(vectors)
    
    # Merge tasks within the same clusters
    clustered_tasks = {}
    for task_idx, cluster_id in enumerate(clustering_model.labels_):
        task = tasks[task_idx]
        if cluster_id not in clustered_tasks:
            clustered_tasks[cluster_id] = task
        else:
            # Simple merge: Keep longest description of the two
            existing = clustered_tasks[cluster_id]
            if len(task.get("description", "")) > len(existing.get("description", "")):
                existing["description"] = task["description"]
                existing["name"] = task["name"]
    
    return list(clustered_tasks.values())

# ============================================================
# 🚀 PHASE 3: MASTER INTELLIGENCE ENGINE (MID-TIER MODEL)
# ============================================================
@with_retry
async def _enrich_and_sequence_tasks(project_name: str, existing_tasks: list, project_description: str, agent_list: list) -> tuple[dict, int, int]:
    prompt_master = """
# 🧠 ENTERPRISE PROJECT INTELLIGENCE ENGINE — MASTER SYSTEM PROMPT

## 🔥 ROLE & DIRECTIVE
You are an **Elite Chief Operating Officer (COO) + Enterprise Systems Architect + Data Structuring Engine**.
You specialize in converting chaotic, fragmented human inputs (Excel, text dumps, meeting notes) into structurally perfect, mathematically sound, enterprise-grade execution plans.

You NEVER return commentary. You NEVER explain reasoning. You NEVER hallucinate data formats. You ALWAYS return strict, schema-compliant JSON.

## 🎯 OBJECTIVE
1. **Ingest & Comprehend:** Analyze the provided context, raw tasks, and metadata.
2. **Phase Definition:** Establish a chronological, logical phased approach (e.g., Discovery -> Planning -> Execution -> Post-Mortem).
3. **Strategic Gap Filling:** Detect missing operational dependencies (legal, financial, logistics) and inject them seamlessly.
4. **Temporal Anchoring:** Map a mathematically sound timeline. If no year is provided, default to the current year. Ensure strict chronological progression.
5. **Output Generation:** Return EVERYTHING in the defined strict JSON schema.

## 🧩 CRITICAL SYSTEM CONSTRAINTS (OBEY SANS EXCEPTION)

### 1️⃣ Identity & Scope Inference (DYNAMIC GRANULARITY)
- Deduce the true nature of the overarching initiative (e.g., Event, Software Launch, Marketing Campaign).
- **CRITICAL MULTI-PROJECT RULE & TIERING:** You MUST evaluate the scale, budget, and timeline of the raw data to determine the project's Tier. Your task generation must reflect a **sufficient operational depth** appropriate for the Tier:
  - **Tier 1 (Light/Tactical):** E.g. A single blog post, a 1-hour webinar. Output exactly ONE project array with **10-15 tasks**. Skip heavy corporate administration.
  - **Tier 2 (Core/Operational):** E.g. A minor product feature, a local event. Output exactly ONE project array with **15-25 tasks**. Include basic strategy and review phases.
  - **Tier 3 (Enterprise/Heavy):** E.g. A global Trade Show or major Company Initiative. You MUST fracture the massive workload into exactly 2 to 3 distinct departmental `projects` (e.g., Project 1: Marketing, Project 2: Sales Ops). Generate a **sufficient but highly digestible** execution blueprint (Aiming for ~10 to 15 high-value, actionable tasks per fractured project). Do not overwhelm with micro-tasks; focus on core operational steps.

### 2️⃣ Enterprise Task Generation Rules
You must expand the provided list to completely cover the operational lifecycle based on your calculated Tier.
**MANDATORY STRATEGY INJECTION:** You MUST evaluate the need for and inject the following cross-functional tasks into the appropriate project if absent:
- **Financial/Legal:** Budget approval, ROI target setting, contract review, insurance.
- **Operations/Logistics:** Inbound/outbound shipping, venue booking.
- **Marketing/Sales:** PR strategy, competitor analysis, lead follow-up systems.

### 3️⃣ Algorithmic Deduplication & Normalization
- Ruthlessly merge semantic duplicates. If "Email clients" and "Send mailer" exist, combine them into one robust task with a comprehensive description.
- Normalize all language to be active, professional, and outcome-oriented (e.g., "Finalize Booth Design" instead of "booth stuff").

### 4️⃣ Temporal Physics & Deadline Logic (ZERO TIME TRAVEL)
- If a user provides an anchor date (e.g., an event date), ALL preparatory tasks MUST mathematically precede it.
- **CRITICAL YEAR BOUNDARY LOGIC**: Pay extreme attention to the year. If tasks span from late Q4 (Nov/Dec) into early Q1 (Jan/Feb), you MUST increment the year for the Q1 tasks. Do NOT schedule a task for "Jan 2025" if the project starts in "Nov 2025". This is a fatal error.
- Calculate realistic `estimated_duration_days` based on task complexity (e.g., "Design Approval" = 3 days, "Sea Freight Shipping" = 30 days).

### 5️⃣ Ownership & Agent Routing
- Assign every single task to a specific, logical Functional Agent from the provided `AGENT_LIST` (e.g., `market_research`, `marketing`, `financial`, `legal`, `procurement`, `operations`, `product`, `business_plan`). DO NOT invent agents.

### 6️⃣ Risk Cascade & Mitigation
- Identify at least 3 high-impact risks specific to this exact project type. Do not use generic risks.
- For each risk, explicitly define the `impact` (High/Medium/Low) and a concrete, actionable `mitigation` strategy.

### 7️⃣ Success Metrics & ROI
- Define exactly how the success of this project will be measured. Use quantifiable KPIs (e.g., "Generate 150+ SQLs", "Keep total spend under $15,000", "Achieve 99.9% uptime").

### 8️⃣ JSON STRICT OUTPUT SCHEMA
Return ONLY this structure (fill in the data payload). Any deviation will cause a system crash:
```json
{{
  "projects": [
    {{
      "project_name": "",
      "project_type": "",
      "project_summary": "",
      "project_start": "YYYY-MM-DD",
      "projected_end": "YYYY-MM-DD",
      "event_date": "YYYY-MM-DD",
      "location": "",
      "booth_size": "",
      "expected_attendance": 0,
      "crm_system": "",
      "preset_name": "",
      "phases": [{{"phase_name": "", "phase_order": 1}}],
      "tasks": [
        {{
          "name": "",
          "description": "",
          "phase": "",
          "priority": "high | medium | low",
          "dependencies": [],
          "origin": {{"type": "user_upload | ai_generated", "confidence": 0.0}},
          "agent": {{"name": "", "id": 0}},
          "details": {{
            "assigned_to": [], 
            "deadline": "", 
            "estimated_duration_days": 0, 
            "status": "planned", 
            "budget_estimate": null,
            "requires_approval": false,
            "automation_possible": false
          }}
        }}
      ],
      "risks": [{{"risk": "", "impact": "", "mitigation": ""}}],
      "success_metrics": [],
      "estimated_total_budget": null
    }}
  ],
  "meta": {{
    "inferred_project_type": "",
    "input_structure_type": "structured",
    "complexity_score": 0,
    "confidence_overall": 0.0
  }}
}}
```

**DATA**:
REFERENCE PROJECT NAME: {project_name}
EXISTING TASKS: {existing_tasks}
AGENT_LIST: {agent_list}
"""
    
    chain = ChatPromptTemplate.from_template(prompt_master) | _make_sonnet()
    raw = await chain.ainvoke({"project_name": project_name, "existing_tasks": existing_tasks, "agent_list": agent_list})
    
    usage = raw.response_metadata.get("usage", {})
    parsed = _parser.parse(raw.content)
    return parsed, usage.get("input_tokens", 0), usage.get("output_tokens", 0)


# ============================================================
# 🚀 PHASE 2.5: WORKLOAD ROUTER (HAIKU)
# ============================================================
@with_retry
async def _route_and_fracture_workload(project_name: str, all_tasks: list) -> tuple[dict, int, int]:
    """Uses Haiku to quickly categorize a massive list of tasks into 1-4 distinct Project clusters before heavy Sonnet processing."""
    
    router_prompt = """
You are an Enterprise Workflow Router.
Your job is to look at a massive list of raw tasks and group them logically into distinct `projects`.
- If the list is small/tactical, group them all into 1 project.
- If the list is massive (e.g. a Trade Show), you MUST fracture them into 2-4 distinct departmental projects (e.g. Marketing, Logistics, Sales).

Return ONLY this strict JSON structure:
{{
    "projects": [
        {{
            "project_name": "Name of the distinct project (e.g., {project_name}: Marketing & PR)",
            "tasks": ["raw task 1", "raw task 2"]
        }}
    ]
}}

TASKS TO ROUTE:
{tasks}
"""
    chain = ChatPromptTemplate.from_template(router_prompt) | _make_haiku()
    raw = await chain.ainvoke({"project_name": project_name, "tasks": all_tasks})
    
    usage = raw.response_metadata.get("usage", {})
    return _parser.parse(raw.content), usage.get("input_tokens", 0), usage.get("output_tokens", 0)

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
        yield {"status_message": "Starting parallel extraction with Semantic Pre-Filtering...", "progress": 10}
        
        shared_llm = _make_haiku()
        chunk_results = await asyncio.gather(*[
            _process_chunk(chunk, i, project_name, project_description, user_data, preset_list, i == 0, shared_llm)
            for i, chunk in enumerate(chunks)
        ])

        extracted_tasks = []
        final_project_name = project_name
        final_preset_name = ""
        haiku_input_tokens = 0
        haiku_output_tokens = 0

        for i, (parsed, inp_tok, out_tok) in enumerate(chunk_results):
            extracted_tasks.extend(parsed.get("tasks", []))
            haiku_input_tokens += inp_tok
            haiku_output_tokens += out_tok
            if i == 0:
                final_project_name = parsed.get("project_name") or project_name
                final_preset_name = parsed.get("preset_name") or "Uncategorized"

        yield {"status_message": "Running Advanced Hierarchical Clustering (HDBScan Alternative)...", "progress": 60}

        normalized_tasks = [normalize_task(t) for t in extracted_tasks if t]
        normalized_tasks = [t for t in normalized_tasks if t]
        
        # 🚀 Use the new O(N log N) agglomerative clustering instead of raw Cosine Matrix
        deduplicated_tasks = advanced_semantic_cluster(normalized_tasks)

        yield {"status_message": "Master Intelligence Engine analyzing and structuring...", "progress": 80}

        # 🚀 STEP 2.5: FRACTURE THE WORKLOAD
        routed_data, route_in, route_out = await _route_and_fracture_workload(final_project_name, deduplicated_tasks)
        haiku_input_tokens += route_in
        haiku_output_tokens += route_out
        
        fractured_projects = routed_data.get("projects", [])
        if not fractured_projects: # Fallback if router fails
            fractured_projects = [{"project_name": final_project_name, "tasks": deduplicated_tasks}]

        # 🚀 STEP 3: PARALLEL SONNET EXECUTION ON FRACTURED ARRAYS
        sonnet_tasks = []
        for p in fractured_projects:
            sonnet_tasks.append(
                _enrich_and_sequence_tasks(
                    project_name=p.get("project_name", final_project_name),
                    existing_tasks=p.get("tasks", []),
                    project_description=project_description,
                    agent_list=agents
                )
            )
            
        sonnet_results = await asyncio.gather(*sonnet_tasks)

        yield {"status_message": "Finalizing optimized JSON payload...", "progress": 100}

        master_projects = []
        sonnet_in_total = 0
        sonnet_out_total = 0
        
        for (parsed_plan, s_in, s_out) in sonnet_results:
            master_projects.extend(parsed_plan.get("projects", []))
            sonnet_in_total += s_in
            sonnet_out_total += s_out

        haiku_input_cost = haiku_input_tokens * (0.80 / 1_000_000)
        haiku_output_cost = haiku_output_tokens * (4.00 / 1_000_000)
        sonnet_input_cost = sonnet_in_total * (3.00 / 1_000_000)
        sonnet_output_cost = sonnet_out_total * (15.00 / 1_000_000)

        input_cost = haiku_input_cost + sonnet_input_cost
        output_cost = haiku_output_cost + sonnet_output_cost

        # Calculate total tasks from the array generated
        
        # 🚀 POST-LLM INTEGRITY AUDIT (Dependencies & Deadlines)
        audited_projects = [_audit_and_fix_project(p) for p in master_projects]
        
        # Merge our pipeline costs onto the meta object returned by the Master Prompt
        # We take the meta from the first Sonnet result as the base
        base_meta = sonnet_results[0][0].get("meta", {}) if sonnet_results else {}
        
        meta = base_meta.copy()
        meta["chunks_processed"] = len(chunks)
        meta["input_cost_usd"] = input_cost
        meta["output_cost_usd"] = output_cost
        meta["total_cost_usd"] = input_cost + output_cost
        
        # Calculate total tasks across ALL fractured projects
        total_tasks = sum(len(p.get("tasks", [])) for p in audited_projects)
        meta["total_tasks"] = total_tasks
        
        print("\n" + "="*50)
        print(f"✅ EXCELLENCE ENGINE COMPLETE")
        print(f"📊 Total Program Tasks: {total_tasks}")
        print(f"💰 Total API Cost: ${meta['total_cost_usd']:.4f}")
        print("="*50 + "\n")

        yield {
            "projects": audited_projects,
            "meta": meta
        }

    except Exception as e:
        yield {"error": f"Pipeline v3 failed: {e}"}
