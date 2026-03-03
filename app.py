import streamlit as st
import asyncio
import json
from datetime import datetime

# Import your entire backend pipeline!
# (Assuming you saved the code we built in a file named `pipeline.py`)
from doc_tracker_v3 import read_file, identify_doc_type, generate_doc_project

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Project Extraction AI",
    page_icon="🤖",
    layout="wide"
)

# ==========================================
# RENDER: PROGRAM EXECUTIVE DASHBOARD
# ==========================================
def render_program_dashboard(data: dict):
    projects = data.get("projects", [])
    meta = data.get("meta", {})
    
    if not projects:
        st.warning("⚠️ No projects found.")
        return

    # 1. Global Program Meta Bar
    st.markdown("### 🏢 Program Executive Overview")
    with st.container(border=True):
        cols = st.columns(4)
        cols[0].metric("Fractured Projects", len(projects))
        cols[1].metric("Total Program Tasks", meta.get('total_tasks', 0))
        cols[2].metric("Chunks Analyzed", meta.get('chunks_processed', 0))
        cols[3].metric("AI Compute Cost", f"${meta.get('total_cost_usd', 0):.4f}")

    # 2. Iterate Fractured Projects
    for idx, proj in enumerate(projects):
        st.markdown(f"## 🚀 Project {idx+1}: {proj.get('project_name', 'Untitled Project')}")
        
        # Project Top-Level Metrics
        p_cols = st.columns(4)
        p_cols[0].markdown(f"**Category:** `{proj.get('preset_name', 'Uncategorized')}`")
        if proj.get('event_date'):
            p_cols[1].markdown(f"📅 **Event Date:** `{proj.get('event_date')}`")
            p_cols[2].markdown(f"📍 **Location:** `{proj.get('location', 'TBD')}`")
            p_cols[3].markdown(f"👥 **Attendance:** `{proj.get('expected_attendance', 0)}`")
            
        tasks = proj.get("tasks", [])
        if not tasks:
            st.info("No tasks extracted for this project.")
            continue
            
        st.markdown("---")
        
        # 3. Group Tasks by Phase within the Project
        phases = {}
        for task in tasks:
            phase = task.get("phase", "Uncategorized")
            if phase not in phases:
                phases[phase] = []
            phases[phase].append(task)
            
        # 4. Render Phase Accordions for this Project
        for phase_name, phase_tasks in phases.items():
            with st.expander(f"📁 **Phase: {phase_name}** ({len(phase_tasks)} Tasks)", expanded=True):
                for task in phase_tasks:
                    details = task.get("details", {})
                    
                    # Badges
                    badges = []
                    if details.get("requires_approval"): badges.append("📝 Req Approval")
                    if details.get("automation_possible"): badges.append("⚙️ Automated")
                    badge_str = " | ".join(badges)
                    
                    # Status & Agent
                    status_raw = details.get("status", "unknown")
                    status_emoji = {"completed": "✅", "in_progress": "⏳", "planned": "📅", "not_started": "⭕", "blocked": "🛑", "unknown": "❓"}.get(status_raw, "❓")
                    agent = task.get("agent", {}).get("name", "Unassigned")
                    
                    # Card Layout
                    t_cols = st.columns([1, 4, 2, 2])
                    t_cols[0].markdown(f"**{status_emoji} {status_raw.title()}**")
                    t_cols[1].markdown(f"**{task.get('name')}**<br>_{task.get('description', '')}_", unsafe_allow_html=True)
                    t_cols[2].markdown(f"**Deadline:** `{details.get('deadline', 'TBD')}`<br>**Agent:** {agent}", unsafe_allow_html=True)
                    t_cols[3].markdown(f"*{badge_str}*" if badge_str else "")
                    st.divider()
        st.markdown("<br><br>", unsafe_allow_html=True)


# ==========================================
# FRONTEND UI + BACKEND EXECUTION
# ==========================================
st.title("📄 Smart Project Tracker AI")
st.write("Upload an Excel or CSV file. Our AI will clean, deduplicate, structure, and expand your project plan.")

with st.sidebar:
    st.header("Project Details")
    project_name = st.text_input("Project Name (Optional)", value="")
    project_desc = st.text_area("Project Context (Optional)", help="Give the AI some background to help it generate missing tasks.")

uploaded_file = st.file_uploader("Upload Tracker (.csv, .xlsx, .docx)", type=["csv", "xlsx", "docx", "doc"])


@st.cache_data(show_spinner=False)
def generate_project_cached(file_bytes: bytes, filename: str, p_name: str, p_desc: str) -> dict:
    # We use a placeholder so we can stream intermediate results
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    async def run_backend():
        # Mock a FastAPI UploadFile object so you don't have to change any code in pipeline.py!
        class MockUploadFile:
            def __init__(self, name, content):
                self.filename = name
                self.content = content
            async def read(self):
                return self.content
        
        mock_file = MockUploadFile(filename, file_bytes)
        
        # Run the pipeline functions
        format_data = await read_file(mock_file)
        
        doc_type_res = await identify_doc_type(format_data)
        if doc_type_res.get("status") != 0:
            return {"error": "The AI determined this document is not a task tracker."}
        
        # Format Data implies stringifying for the LLM context
        context_str = str(format_data) 
        
        # We now expect generate_doc_project to be an async generator yielding intermediate dicts
        final_result = None
        async for interim_state in generate_doc_project(
            format=format_data,
            project_name=p_name if p_name else filename,
            project_description=p_desc,
            user_data=context_str,
            doc_type=doc_type_res
        ):
            if "status_message" in interim_state:
                status_placeholder.info(f"⏳ {interim_state['status_message']}")
                if "progress" in interim_state:
                    progress_bar.progress(interim_state["progress"])
            else:
                final_result = interim_state
                
        status_placeholder.empty()
        progress_bar.empty()
        return final_result or {"error": "Pipeline failed to return data."}
    
    return asyncio.run(run_backend())

if st.button("Process Document", type="primary") and uploaded_file is not None:
    
    with st.spinner("🤖 AI is preparing your Project"):
        try:
            # Step 1: Read the file from Streamlit's UI
            file_bytes = uploaded_file.read()
            filename = uploaded_file.name
            
            # Step 2 & 3: Execute cached pipeline!
            result_json = generate_project_cached(file_bytes, filename, project_name, project_desc)

            if "error" in result_json:
                st.error(result_json["error"])
            else:
                cost = result_json.get("meta", {}).get("total_cost_usd", 0)
                st.success(f"✅ Extracted & Structurally Validated into Program Segments! (Cost: ${cost:.4f})")
                
                # Display Results in Tabs
                tab1, tab2 = st.tabs(["� Executive Dashboard", "💻 Raw JSON"])
                
                with tab1:
                    render_program_dashboard(result_json)
                    
                with tab2:
                    st.download_button("📥 Download Master Program JSON", json.dumps(result_json, indent=2), file_name=f"{filename}_program_plan.json")
                    st.json(result_json)
                    st.json(result_json)

        except Exception as e:
            st.error(f"An error occurred during execution: {str(e)}")
