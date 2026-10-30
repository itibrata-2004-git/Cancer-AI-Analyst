
"""
Cancer AI Analyst
Local Agentic RAG application using Streamlit, LangGraph,
FAISS, and Ollama.

Run from the project root:
    streamlit run app/agentic_app.py
"""

import sys
import traceback
from pathlib import Path

import streamlit as st


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

# Project root:
# Cancer AI Analyst/
BASE_DIR = Path(__file__).resolve().parent.parent

# App directory:
# Cancer AI Analyst/app/
APP_DIR = Path(__file__).resolve().parent

# FAISS vector database directory
FAISS_DIR = BASE_DIR / "faiss_index"

# Uploaded PDF directory
UPLOAD_DIR = BASE_DIR / "data" / "uploaded_pdfs"

# Make the app directory available for importing agent.py
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Create upload directory if it does not exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cancer AI Analyst",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }

        [data-testid="stSidebar"] {
            background-color: #151922;
            border-right: 1px solid #2b313d;
        }

        .main-title {
            font-size: 2.5rem;
            font-weight: 800;
            color: #64b5f6;
            margin-bottom: 0;
        }

        .subtitle {
            font-size: 1rem;
            color: #aab4c3;
            margin-top: 5px;
            margin-bottom: 25px;
        }

        .info-card {
            background-color: #171d28;
            border: 1px solid #2b3545;
            padding: 18px;
            border-radius: 12px;
            margin-bottom: 15px;
        }

        .status-label {
            color: #aab4c3;
            font-size: 0.85rem;
        }

        .status-value {
            font-size: 1.1rem;
            font-weight: 600;
        }

        div[data-testid="stChatMessage"] {
            border-radius: 12px;
            padding: 10px;
        }

        .stButton button {
            border-radius: 8px;
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

if "agent_loaded" not in st.session_state:
    st.session_state.agent_loaded = False

if "agent_error" not in st.session_state:
    st.session_state.agent_error = None

if "agent_module" not in st.session_state:
    st.session_state.agent_module = None


# ============================================================
# 5. LOAD EXISTING LANGGRAPH AGENT
# ============================================================

@st.cache_resource(show_spinner=False)
def load_agent():
    """
    Import the existing agent from app/agent.py.

    The agent module should expose an object named `agent`.
    """

    import importlib.util

    agent_file = APP_DIR / "agent.py"

    if not agent_file.exists():
        raise FileNotFoundError(
            f"Agent file not found: {agent_file}"
        )

    # Load the module directly by its file path.
    # This avoids conflicts with app.py or another app package.
    spec = importlib.util.spec_from_file_location(
        "cancer_ai_agent_module",
        agent_file,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            "Could not load the agent.py module."
        )

    module = importlib.util.module_from_spec(spec)

    # Register the module before executing it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "agent"):
        raise AttributeError(
            "The file app/agent.py does not expose "
            "an object named 'agent'."
        )

    return module



def initialize_agent():
    if st.session_state.agent_loaded:
        return True

    try:
        with st.spinner("Initializing Cancer AI Analyst..."):
            module = load_agent()

        st.session_state.agent_module = module
        st.session_state.agent = module.agent
        st.session_state.agent_loaded = True
        st.session_state.agent_error = None

        return True

    except Exception as e:
        st.session_state.agent = None
        st.session_state.agent_module = None
        st.session_state.agent_loaded = False
        st.session_state.agent_error = str(e)

        return False



# ============================================================
# 6. AGENT RESPONSE HANDLER
# ============================================================

def extract_response(result):
    """
    Extract the final assistant response from a LangGraph
    create_react_agent result.
    """

    if isinstance(result, str):
        return result

    if not isinstance(result, dict):
        return str(result)

    messages = result.get("messages", [])

    if not messages:
        return str(result)

    # Search backward for the final assistant message.
    for message in reversed(messages):

        # LangChain message objects
        message_type = getattr(
            message,
            "type",
            None,
        )

        if message_type == "ai":
            content = getattr(
                message,
                "content",
                "",
            )

            if isinstance(content, str):
                return content

            if isinstance(content, list):
                parts = []

                for item in content:
                    if isinstance(item, dict):
                        text_content = item.get("text")
                        if text_content:
                            parts.append(text_content)
                    elif isinstance(item, str):
                        parts.append(item)

                if parts:
                    return "\n".join(parts)

        # Dictionary-based messages
        if isinstance(message, dict):
            role = message.get(
                "type",
                message.get("role", ""),
            )

            if role in ("ai", "assistant"):
                content = message.get("content", "")

                if isinstance(content, str):
                    return content

                if isinstance(content, list):
                    parts = []

                    for item in content:
                        if isinstance(item, dict):
                            if item.get("text"):
                                parts.append(item["text"])
                        elif isinstance(item, str):
                            parts.append(item)

                    if parts:
                        return "\n".join(parts)

    return "The agent completed its execution but returned no text response."


def ask_agent(question):
    """
    Send a question to the existing LangGraph agent.

    Expected agent interface:
        agent.invoke({"messages": [("user", question)]})
    """

    current_agent = st.session_state.agent

    if current_agent is None:
        raise RuntimeError(
            "The agent is not initialized."
        )

    result = current_agent.invoke(
        {
            "messages": [
                ("user", question)
            ]
        }
    )

    return extract_response(result)


# ============================================================
# 7. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <h2 style="color:#64b5f6;">
            🧬 Cancer AI Analyst
        </h2>
        <p style="color:#aab4c3;">
            Your local AI-powered cancer research assistant.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.subheader("⚙️ System Status")

    if st.session_state.agent_loaded:
        st.success("Agent: Initialized")
    elif st.session_state.agent_error:
        st.error("Agent: Initialization failed")
    else:
        st.info("Agent: Not initialized")

    st.caption(
        f"Project directory: {BASE_DIR}"
    )

    st.caption(
        f"FAISS directory: {FAISS_DIR}"
    )

    st.divider()

    # --------------------------------------------------------
    # PDF UPLOAD
    # --------------------------------------------------------

    st.subheader("📄 Upload Research PDFs")

    uploaded_files = st.file_uploader(
    "Select cancer research PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload PDFs to save locally and add them to the AI knowledge base.",
    key="cancer_research_pdfs"
)

if uploaded_files:

    for uploaded_file in uploaded_files:

        destination = (
            UPLOAD_DIR / Path(uploaded_file.name).name
        )

        try:
            file_bytes = uploaded_file.getvalue()

            if not destination.exists():
                with open(destination, "wb") as f:
                    f.write(file_bytes)

            if not initialize_agent():
                st.error("AI agent could not be initialized.")
                continue

            module = st.session_state.agent_module

            with st.spinner(
                f"Processing {uploaded_file.name}..."
            ):
                result = module.ingest_pdf(
                    str(destination)
                )

            if result.get("status") == "success":

                st.success(
                    f"Successfully indexed: {uploaded_file.name}"
                )

                st.info(
                    f"Chunks created: {result.get('chunks', 0)} "
                    f"| Pages: {result.get('pages', 0)}"
                )

            elif result.get("status") == "already_indexed":

                st.info(
                    f"Already indexed: {uploaded_file.name}"
                )

            else:
                st.error(
                    result.get(
                        "message",
                        f"Failed to index {uploaded_file.name}"
                    )
                )

        except Exception as e:
            st.error(
                f"Error processing {uploaded_file.name}: {e}"
            )

    st.caption(
        "Note: Saving a PDF does not automatically add it "
        "to the FAISS index. Your existing agent must have "
        "a PDF ingestion/indexing function for that."
    )

    st.divider()

    # --------------------------------------------------------
    # LOCAL PDF LIST
    # --------------------------------------------------------

    st.subheader("📚 Local Documents")

    pdf_files = sorted(
        UPLOAD_DIR.glob("*.pdf")
    )

    if pdf_files:

        st.caption(
            f"{len(pdf_files)} PDF file(s) stored"
        )

        with st.expander("View uploaded documents"):

            for pdf in pdf_files:
                st.write(f"📄 {pdf.name}")

    else:
        st.caption("No PDFs uploaded yet.")

    st.divider()

    # --------------------------------------------------------
    # CHAT CONTROLS
    # --------------------------------------------------------

    st.subheader("💬 Chat Controls")

    if st.button(
        "🗑️ Clear Chat History",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    if st.button(
        "🔄 Initialize / Retry Agent",
        use_container_width=True,
    ):
        st.session_state.agent_loaded = False
        st.session_state.agent = None
        st.session_state.agent_error = None

        # Clear cached module loader result
        load_agent.clear()

        success = initialize_agent()

        if success:
            st.success("Agent initialized successfully.")
        else:
            st.error(
                st.session_state.agent_error
                or "Agent initialization failed."
            )

    st.divider()

    st.caption(
        "Local deployment • Streamlit • LangGraph • FAISS • Ollama"
    )


# ============================================================
# 8. MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        🧬 Cancer AI Analyst
    </div>

    <div class="subtitle">
        Agentic AI assistant for cancer research,
        medical literature, and document-based question answering.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 9. TOP STATUS CARDS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="info-card">
            <div class="status-label">AI Framework</div>
            <div class="status-value">LangGraph Agent</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="info-card">
            <div class="status-label">Vector Database</div>
            <div class="status-value">FAISS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="info-card">
            <div class="status-label">LLM Runtime</div>
            <div class="status-value">Local Ollama</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 10. INITIALIZE AGENT ON FIRST RUN
# ============================================================

if not st.session_state.agent_loaded:

    with st.spinner(
        "Connecting to your local AI agent..."
    ):
        agent_ready = initialize_agent()

    if agent_ready:
        st.success(
            "Cancer AI Analyst is ready. Ask your first question below."
        )
    else:
        st.error(
            "The AI agent could not be initialized."
        )

        st.warning(
            "Check the terminal for the full error. "
            "Verify that app/agent.py, Ollama, and the FAISS "
            "index are configured correctly."
        )

        if st.session_state.agent_error:
            with st.expander(
                "View initialization error"
            ):
                st.code(
                    st.session_state.agent_error
                )


# ============================================================
# 11. CHAT INTERFACE
# ============================================================

st.subheader("💬 Ask Cancer AI Analyst")

st.caption(
    "Ask questions about cancer research, uploaded literature, "
    "or information indexed in your local knowledge base."
)


# ------------------------------------------------------------
# DISPLAY EXISTING CHAT HISTORY
# ------------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# ------------------------------------------------------------
# CHAT INPUT
# ------------------------------------------------------------

user_question = st.chat_input(
    "Ask a question about cancer research..."
)


if user_question:

    # Save and display user question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    # Check agent before execution
    if not st.session_state.agent_loaded:

        with st.chat_message("assistant"):
            st.error(
                "The agent is not available. "
                "Please check the initialization error "
                "and retry from the sidebar."
            )

    else:

        # Generate assistant response
        with st.chat_message("assistant"):

            try:

                with st.spinner(
                    "Analyzing your question..."
                ):

                    answer = ask_agent(
                        user_question
                    )

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as e:

                st.error(
                    "An error occurred while generating "
                    "the response."
                )

                with st.expander(
                    "View technical error"
                ):
                    st.code(
                        traceback.format_exc()
                    )

                st.info(
                    "Check whether Ollama is running, "
                    "the configured model is available, "
                    "and the agent uses the expected "
                    "LangGraph invocation format."
                )


# ============================================================
# 12. WELCOME SCREEN
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
        ### 👋 Welcome!

        Cancer AI Analyst is a local AI research assistant
        designed to help you explore cancer-related literature.

        **Try asking:**

        - What is the role of machine learning in cancer detection?
        - Summarize the research documents in my knowledge base.
        - Explain the difference between benign and malignant tumors.
        - What does my indexed research say about cancer biomarkers?

        <div class="info-card">
            <b>Important:</b>
            This application is intended for research and educational
            purposes. It is not a medical diagnostic tool and does
            not replace advice from a qualified healthcare professional.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#778397; font-size:0.8rem;">
        Cancer AI Analyst | Local Agentic RAG Application
        <br>
        For educational and research use only.
    </div>
    """,
    unsafe_allow_html=True,
)
