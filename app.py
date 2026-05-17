"""
AI Study Assistant – Streamlit frontend (Groq + Chunked PDF Edition)
MODIFICATIONS vs baseline:
    • PDF text chunked on upload (~3800 chars, 400 overlap)
    • Sentence embeddings cached in session_state for semantic retrieval
    • Relevant chunks sent to Groq instead of full PDF (prevents 413)
    • assemble_context() hard-caps prompt characters
    • Graceful fallback to keyword search if embeddings missing
"""

import os
import streamlit as st
from datetime import datetime

import streamlit.components.v1 as components
from utils import (
    extract_text_from_pdf,
    split_text_into_chunks,
    get_relevant_chunks,
    get_ai_response,
    build_qa_prompt,
    build_summary_prompt,
    build_quiz_prompt,
    build_notes_prompt,
    build_explain_prompt,
    build_study_planner_prompt,
    parse_quiz_json,
    build_chunk_embeddings,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# ── Page Config ──
st.set_page_config(
    page_title="Study Quest | AI Study Assistant",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global Game Theme CSS ──
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;500;700&display=swap');

    .stApp {
        background: radial-gradient(circle at 10% 20%, #0f172a 0%, #0b1026 40%, #020617 100%);
        color: #e2e8f0;
        font-family: 'Rajdhani', sans-serif;
    }

    h1, h2, h3, .main-title {
        font-family: 'Orbitron', sans-serif !important;
        background: -webkit-linear-gradient(45deg, #00fff2, #bd00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
        font-family: 'Rajdhani', sans-serif;
    }

    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #0b1120 0%, #0f172a 100%) !important;
        border-right: 1px solid #1e293b;
    }

    .stButton>button {
        background: linear-gradient(90deg, #00fff2, #008aff) !important;
        color: #000 !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        box-shadow: 0 0 15px rgba(0, 255, 242, 0.3);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 25px rgba(0, 255, 242, 0.6);
    }

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stDateInput > div > div > input {
        background-color: #0f172a !important;
        color: #00fff2 !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        font-family: 'Rajdhani', sans-serif !important;
    }

    .game-card {
        background-color: #0f172a !important;
        border: 1px solid rgba(0, 255, 242, 0.15) !important;
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 0 30px rgba(0, 255, 242, 0.04);
        margin-bottom: 1.5rem;
    }

    .option-correct {
        border: 2px solid #00ff9d;
        background: rgba(0, 255, 157, 0.08);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.8rem;
        font-weight: 600;
        color: #00ff9d;
        box-shadow: 0 0 10px rgba(0, 255, 157, 0.2);
    }
    .option-wrong {
        border: 2px solid #ff0055;
        background: rgba(255, 0, 85, 0.08);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.8rem;
        font-weight: 600;
        color: #ff0055;
        box-shadow: 0 0 10px rgba(255, 0, 85, 0.2);
    }
    .option-neutral {
        border: 1px solid #1e293b;
        background: rgba(30, 41, 59, 0.4);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.8rem;
        color: #94a3b8;
    }

    .stChatMessage { animation: fadeIn 0.3s ease-in; }
    @keyframes fadeIn { from { opacity:0; transform: translateY(8px);} to { opacity:1; transform: translateY(0);} }

    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00fff2, #bd00ff) !important;
    }
    </style>
""", unsafe_allow_html=True)

# ── Session State ──
def _init():
    defaults = dict(
        chat_history=[],
        extracted_text="",
        text_chunks=[],
        chunk_embeddings=None,   # NEW: holds numpy embeddings for semantic search
        api_key="",
        model_name="llama-3.3-70b-versatile",
        last_output="",
        pdf_loaded=False,
        quiz_game={
            "data": [],
            "index": 0,
            "score": 0,
            "streak": 0,
            "max_streak": 0,
            "correct_count": 0,
            "status": "idle",
            "selected": None,
            "checked": False,
            "total": 0,
        },
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

_init()

# ── Sidebar ──
with st.sidebar:
    st.markdown("""
        <div style="text-align:center; margin-bottom:1.5rem;">
            <h2 style="font-family:'Orbitron'; color:#00fff2; text-shadow: 0 0 10px #00fff2;">🎮 STUDY QUEST</h2>
            <p style="color:#64748b; font-size:0.9rem;">Level Up Your Learning</p>
        </div>
    """, unsafe_allow_html=True)

    # 1. API Key
    st.markdown("### 1. API Key")
    env_key = os.getenv("GROQ_API_KEY", "")
    if env_key and not st.session_state.api_key:
        st.session_state.api_key = env_key

    key_input = st.text_input(
        "Groq API Key",
        type="password",
        value=st.session_state.api_key,
        placeholder="gsk_...",
        help="Get yours at https://console.groq.com/keys",
    )
    if key_input != st.session_state.api_key:
        st.session_state.api_key = key_input

    st.divider()

    # 2. Model
    st.markdown("### 2. Model")
    KNOWN_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-specdec",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]
    use_manual = st.checkbox("Custom Model ID", key="custom_model")
    if use_manual:
        model_name = st.text_input("Model Name", value=st.session_state.model_name, label_visibility="collapsed")
    else:
        try:
            default_idx = KNOWN_MODELS.index(st.session_state.model_name)
        except ValueError:
            default_idx = 0
        model_name = st.selectbox("Pick Model", KNOWN_MODELS, index=default_idx, label_visibility="collapsed")
    st.session_state.model_name = model_name

    st.divider()

    # 3. Upload & Chunk PDF
    st.markdown("### 3. Upload Material")
    pdf_status = st.empty()
    uploaded = st.file_uploader("Drop PDFs here", type=["pdf"], accept_multiple_files=True)

    if uploaded:
        with st.status("Scanning & chunking files...", expanded=True) as status:
            all_txt, ok = "", 0
            for f in uploaded:
                txt = extract_text_from_pdf(f)
                if txt:
                    all_txt += txt + "\n\n"
                    ok += 1

            if all_txt.strip():
                # NEW: guard against massive PDFs to keep memory sane
                if len(all_txt) > 2_000_000:
                    status.warning("Very large PDF detected. Truncating to ~2M chars to prevent memory issues.")
                    all_txt = all_txt[:2_000_000]

                # Chunk text into ~3800-char slices (never store raw giant string for LLM)
                st.session_state.extracted_text = all_txt
                st.session_state.text_chunks = split_text_into_chunks(
                    all_txt, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP
                )
                st.session_state.pdf_loaded = True

                # NEW: build sentence embeddings once for fast semantic retrieval
                try:
                    with st.spinner("Building embeddings (one-time)..."):
                        st.session_state.chunk_embeddings = build_chunk_embeddings(
                            st.session_state.text_chunks
                        )
                    emb_msg = "semantic search ON"
                except Exception as e:
                    st.session_state.chunk_embeddings = None
                    if "sentence-transformers" in str(e).lower():
                        emb_msg = "semantic search OFF (install sentence-transformers)"
                    else:
                        emb_msg = f"embedding failed: {str(e)[:50]}"

                status.update(label=f"✅ {ok} PDF(s) | {len(st.session_state.text_chunks)} chunks • {emb_msg}", state="complete")
            else:
                st.session_state.extracted_text = ""
                st.session_state.text_chunks = []
                st.session_state.pdf_loaded = False
                st.session_state.chunk_embeddings = None
                status.update(label="⚠️ No extractable text", state="error")
                st.toast("Could not read text from the uploaded PDF(s).", icon="⚠️")
    else:
        st.session_state.extracted_text = ""
        st.session_state.text_chunks = []
        st.session_state.pdf_loaded = False
        st.session_state.chunk_embeddings = None
        pdf_status.info("No PDF loaded")

    st.divider()

    # 4. Mode
    st.markdown("### 4. Quest Mode")
    mode_labels = [
        "💬 Ask Questions",
        "📝 Summarize",
        "⚔️ Quiz Arena",
        "📄 Create Notes",
        "🔍 Explain Topic",
        "📅 Study Planner",
    ]
    selected_label = st.radio("Feature", mode_labels, label_visibility="collapsed")
    st.divider()

    if st.session_state.pdf_loaded:
        emb_status = "with semantic search" if st.session_state.chunk_embeddings is not None else "keyword fallback"
        st.success(f"Ready • {len(st.session_state.text_chunks)} chunks • {emb_status}")
    else:
        st.info("Awaiting upload")

# ── Main Page ──
st.markdown("""
    <div style="text-align:center; margin-bottom:3rem;">
        <h1 class="main-title" style="font-size:3rem;">AI STUDY ASSISTANT</h1>
        <p class="subtitle">Upload. Learn. Level Up.</p>
    </div>
""", unsafe_allow_html=True)

if not st.session_state.api_key:
    st.error("🔑 Enter your Groq API Key in the sidebar first.")
    st.stop()

CONFETTI_HTML = """
<script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
<script>
  confetti({ particleCount: 150, spread: 70, origin: { y: 0.6 }, colors: ['#00fff2', '#bd00ff', '#ffe600'] });
</script>
"""


def show_output(text: str, filename: str = "output.txt"):
    if not text:
        return
    st.markdown('<div class="game-card">', unsafe_allow_html=True)
    st.markdown(text)
    st.markdown('</div>', unsafe_allow_html=True)
    st.download_button("⬇️ Download .txt", text, file_name=filename, mime="text/plain", use_container_width=True)


# ── Routing ──
ss = st.session_state

if selected_label == "💬 Ask Questions":
    st.header("💬 Ask Questions")

    if not ss.pdf_loaded:
        st.info("Upload a PDF in the sidebar to chat with your material.")
    else:
        for msg in ss.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ask anything about your uploaded material...")

        if user_input:
            ss.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving relevant chunks & querying AI..."):
                    # NEW: semantic search (with keyword fallback) – only top relevant chunks sent
                    relevant = get_relevant_chunks(
                        user_input,
                        ss.text_chunks,
                        top_k=3,
                        chunk_embeddings=ss.get("chunk_embeddings"),
                    )
                    prompt = build_qa_prompt(user_input, relevant)
                    response = get_ai_response(
                        prompt,
                        ss.api_key,
                        model_name=ss.model_name,
                    )

                if response.startswith("Error"):
                    st.error(response)
                else:
                    st.markdown(response)

                if not response.startswith("Error"):
                    ss.chat_history.append({"role": "assistant", "content": response})

        if ss.chat_history:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                ss.chat_history = []
                st.rerun()

elif selected_label == "📝 Summarize":
    st.header("📝 Summarize")

    if not ss.pdf_loaded:
        st.info("Upload a PDF first.")
    else:
        s_type = st.radio("Depth", ["Short Summary", "Detailed Summary", "Exam Revision Summary"], horizontal=True)
        if st.button("Generate Summary", type="primary", use_container_width=True):
            with st.spinner("Synthesizing knowledge..."):
                key = "short" if "Short" in s_type else "exam" if "Exam" in s_type else "detailed"
                # assemble_context inside builder auto-truncates, preventing 413
                prompt = build_summary_prompt(ss.text_chunks, key)
                response = get_ai_response(prompt, ss.api_key, model_name=ss.model_name)
                ss.last_output = response
            if not ss.last_output.startswith("Error"):
                show_output(ss.last_output, f"{key}_summary.txt")
            else:
                st.error(ss.last_output)

elif selected_label == "⚔️ Quiz Arena":
    qg = ss.quiz_game

    if qg["status"] == "idle":
        st.header("⚔️ Quiz Arena")
        st.markdown('<div class="game-card">', unsafe_allow_html=True)
        st.subheader("Ready Player One?")
        st.write("Generate an MCQ challenge from your uploaded material and test your knowledge.")
        if not ss.pdf_loaded:
            st.warning("Upload a PDF in the sidebar to unlock the arena.")
        else:
            if st.button("⚡ GENERATE QUIZ", type="primary", use_container_width=True):
                with st.spinner("AI is crafting your challenge..."):
                    # assemble_context inside builder limits prompt size automatically
                    prompt = build_quiz_prompt(ss.text_chunks, "MCQs")
                    raw = get_ai_response(prompt, ss.api_key, model_name=ss.model_name)
                    if raw.startswith("Error"):
                        st.error(raw)
                    else:
                        parsed = parse_quiz_json(raw)
                        if parsed and len(parsed) > 0:
                            valid = True
                            for item in parsed:
                                if not all(k in item for k in ("question", "options", "correct_answer", "explanation")):
                                    valid = False
                                    break
                                if not all(k in item["options"] for k in ["A", "B", "C", "D"]):
                                    valid = False
                                    break
                            if valid:
                                qg["data"] = parsed
                                qg["total"] = len(parsed)
                                qg["index"] = 0
                                qg["score"] = 0
                                qg["streak"] = 0
                                qg["max_streak"] = 0
                                qg["correct_count"] = 0
                                qg["status"] = "active"
                                qg["checked"] = False
                                qg["selected"] = None
                                st.rerun()
                            else:
                                st.error("AI returned an invalid quiz format. Please try again.")
                        else:
                            st.error("Could not parse quiz. Try re-generating!")
        st.markdown('</div>', unsafe_allow_html=True)

    elif qg["status"] == "active":
        # HUD
        current_q = qg["data"][qg["index"]]
        progress = (qg["index"]) / qg["total"] if qg["total"] > 0 else 0
        st.progress(float(progress))

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""
            <div style="text-align:center;">
                <div style="font-size:0.8rem; color:#94a3b8; font-family:'Orbitron';">SCORE</div>
                <div style="font-size:2.2rem; font-weight:700; color:#00fff2; text-shadow: 0 0 12px #00fff2;">{qg['score']}</div>
            </div>
        """, unsafe_allow_html=True)
        c2.markdown(f"""
            <div style="text-align:center;">
                <div style="font-size:0.8rem; color:#94a3b8; font-family:'Orbitron';">STREAK</div>
                <div style="font-size:2.2rem; font-weight:700; color:#ffe600; text-shadow: 0 0 12px #ffe600;">{qg['streak']}🔥</div>
            </div>
        """, unsafe_allow_html=True)
        c3.markdown(f"""
            <div style="text-align:center;">
                <div style="font-size:0.8rem; color:#94a3b8; font-family:'Orbitron';">PROGRESS</div>
                <div style="font-size:2.2rem; font-weight:700; color:#bd00ff; text-shadow: 0 0 12px #bd00ff;">{qg['index']+1}/{qg['total']}</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Question card
        st.markdown(f'<div class="game-card"><h3 style="color:#ffffff; font-family:\'Rajdhani\';">{current_q["question"]}</h3></div>', unsafe_allow_html=True)

        # Options
        opts = current_q["options"]
        keys = ["A", "B", "C", "D"]
        cols = st.columns(2)

        for i, k in enumerate(keys):
            with cols[i % 2]:
                if not qg["checked"]:
                    if st.button(f"{k}. {opts[k]}", key=f"btn_{qg['index']}_{k}", use_container_width=True):
                        qg["selected"] = k
                        qg["checked"] = True
                        if k == current_q["correct_answer"]:
                            bonus = qg["streak"] * 2
                            qg["score"] += 10 + bonus
                            qg["streak"] += 1
                            qg["correct_count"] += 1
                            if qg["streak"] > qg["max_streak"]:
                                qg["max_streak"] = qg["streak"]
                        else:
                            qg["streak"] = 0
                        st.rerun()
                else:
                    is_correct = (k == current_q["correct_answer"])
                    is_selected = (k == qg["selected"])
                    if is_correct:
                        st.markdown(f'<div class="option-correct">✅ <strong>{k}.</strong> {opts[k]}</div>', unsafe_allow_html=True)
                    elif is_selected:
                        st.markdown(f'<div class="option-wrong">❌ <strong>{k}.</strong> {opts[k]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="option-neutral"><strong>{k}.</strong> {opts[k]}</div>', unsafe_allow_html=True)

        if qg["checked"]:
            if qg["selected"] == current_q["correct_answer"]:
                st.success("🎉 Correct! " + current_q.get("explanation", "Great job!"))
                components.html(CONFETTI_HTML, height=0)
            else:
                st.error(f"💥 Wrong! The correct answer was **{current_q['correct_answer']}**.")
                st.info("📖 " + current_q.get("explanation", ""))

            st.markdown("<br>", unsafe_allow_html=True)
            next_label = "Finish Quiz 🏆" if qg["index"] + 1 >= qg["total"] else "Next Question ➡️"
            if st.button(next_label, key="next_q", use_container_width=True):
                qg["index"] += 1
                qg["checked"] = False
                qg["selected"] = None
                if qg["index"] >= qg["total"]:
                    qg["status"] = "finished"
                st.rerun()

    else:  # finished
        accuracy = int((qg["correct_count"] / qg["total"]) * 100) if qg["total"] > 0 else 0
        if accuracy == 100:
            rank, color = "💎 PLATINUM", "#00fff2"
        elif accuracy >= 80:
            rank, color = "🥇 GOLD", "#ffe600"
        elif accuracy >= 60:
            rank, color = "🥈 SILVER", "#c0c0c0"
        else:
            rank, color = "🥉 BRONZE", "#cd7f32"

        st.markdown(f"""
            <div style="text-align:center; margin-top:2rem;">
                <h1 style="font-size:3.5rem; font-family:'Orbitron'; color:{color}; text-shadow: 0 0 20px {color};">
                    QUIZ COMPLETE
                </h1>
                <p style="font-size:1.2rem; color:#94a3b8;">You have conquered the arena</p>
            </div>
        """, unsafe_allow_html=True)
        st.balloons()

        col1, col2, col3 = st.columns(3)
        col1.metric("Final Score", qg["score"])
        col2.metric("Accuracy", f"{accuracy}%")
        col3.metric("Best Streak", f"{qg['max_streak']}🔥")

        st.markdown(f"""
            <div style="text-align:center; margin:2rem 0;">
                <div style="font-size:1.5rem; font-family:'Orbitron'; color:{color};">RANK ACHIEVED</div>
                <div style="font-size:2.5rem; font-weight:900; color:#ffffff;">{rank}</div>
            </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 Play Again", use_container_width=True):
            ss.quiz_game = {
                "data": [], "index": 0, "score": 0, "streak": 0,
                "max_streak": 0, "correct_count": 0, "status": "idle",
                "selected": None, "checked": False, "total": 0,
            }
            st.rerun()

elif selected_label == "📄 Create Notes":
    st.header("📄 Create Short Notes")

    if not ss.pdf_loaded:
        st.info("Upload a PDF first.")
    else:
        if st.button("Generate Notes", type="primary", use_container_width=True):
            with st.spinner("Drafting notes..."):
                # assemble_context in builder prevents 413
                prompt = build_notes_prompt(ss.text_chunks)
                response = get_ai_response(prompt, ss.api_key, model_name=ss.model_name)
                ss.last_output = response
            if not ss.last_output.startswith("Error"):
                show_output(ss.last_output, "notes.txt")
            else:
                st.error(ss.last_output)

elif selected_label == "🔍 Explain Topic":
    st.header("🔍 Explain Topic")

    topic = st.text_input("Topic / Concept", placeholder="e.g., Quantum entanglement, DBMS normalization")
    use_ctx = st.checkbox("Use uploaded material", value=True, disabled=not ss.pdf_loaded)

    if st.button("Explain", type="primary", use_container_width=True):
        if not topic.strip():
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Generating explanation..."):
                # NEW: retrieve only relevant chunks instead of whole PDF
                ctx = get_relevant_chunks(topic, ss.text_chunks, chunk_embeddings=ss.get("chunk_embeddings")) if use_ctx else []
                prompt = build_explain_prompt(topic, ctx)
                response = get_ai_response(prompt, ss.api_key, model_name=ss.model_name)
                ss.last_output = response
            if not ss.last_output.startswith("Error"):
                show_output(ss.last_output, "explanation.txt")
            else:
                st.error(ss.last_output)

elif selected_label == "📅 Study Planner":
    st.header("📅 Study Planner")

    c1, c2 = st.columns(2)
    with c1:
        exam_date = st.date_input("Exam Date", min_value=datetime.now().date())
    with c2:
        hours = st.number_input("Hours per day", min_value=1, max_value=24, value=4)

    subjects = st.text_area("Subjects (comma-separated)", placeholder="e.g., Math, Physics, Chemistry")

    if st.button("Generate Plan", type="primary", use_container_width=True):
        if not subjects.strip():
            st.warning("Please enter at least one subject.")
        else:
            with st.spinner("Building your battle plan..."):
                # No PDF context needed for planner → no token risk
                prompt = build_study_planner_prompt(str(exam_date), subjects, int(hours))
                response = get_ai_response(prompt, ss.api_key, model_name=ss.model_name)
                ss.last_output = response
            if not ss.last_output.startswith("Error"):
                show_output(ss.last_output, "study_plan.txt")
            else:
                st.error(ss.last_output)
