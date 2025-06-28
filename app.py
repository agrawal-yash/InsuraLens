import streamlit as st
import os
import uuid
import shutil
from dotenv import load_dotenv

# Import your custom modules
from document_processor import process_pdf, get_embedding_model
from vector_store_manager import create_vector_store, DB_DIRECTORY
from analysis_agent import get_llm, get_policy_type, get_contextual_questions, generate_analysis_and_recommendation

# --- Constants ---
TEMP_UPLOADS_DIR = "temp_uploads"

# --- Page Configuration ---
st.set_page_config(
    page_title="InsuraLens - A Multi-Insurance Clause Analyzer & Recommendation Chat Agent.",
    page_icon="🤖",
    layout="wide"
)

# --- Caching Functions for Expensive Resources ---

@st.cache_resource
def cached_get_llm():
    """Caches the LLM resource."""
    return get_llm()

@st.cache_resource
def cached_get_embedding_model():
    """Caches the embedding model resource."""
    return get_embedding_model()

# --- Session State Initialization ---

def initialize_session_state():
    """Initializes all the necessary variables in the session state."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # This dictionary will hold all state variables for cleaner access
    if 'app_state' not in st.session_state:
        st.session_state.app_state = {
            "stage": "initial",  # Stages: initial, processing, questions, report
            "uploaded_files": None,
            "policy_type": None,
            "questions": [],
            "user_answers": {},
            "final_report": None,
            "vector_store": None,
            "doc_names": []
        }

def reset_session():
    """Resets the session to its initial state."""
    session_id = st.session_state.session_id
    # Clean up the specific ChromaDB collection for this session
    collection_path = os.path.join(DB_DIRECTORY, f"chroma-collections-{session_id}")
    if os.path.exists(collection_path):
        shutil.rmtree(collection_path)
    
    # Clean up temporary uploaded files
    temp_dir_path = os.path.join(TEMP_UPLOADS_DIR, session_id)
    if os.path.exists(temp_dir_path):
        shutil.rmtree(temp_dir_path)

    # Re-initialize the session state
    st.session_state.session_id = str(uuid.uuid4()) # Get a new ID for the new session
    st.session_state.app_state = {
        "stage": "initial",
        "uploaded_files": None,
        "policy_type": None,
        "questions": [],
        "user_answers": {},
        "final_report": None,
        "vector_store": None,
        "doc_names": []
    }
    st.rerun()


# --- Main App Logic ---

# Load environment variables (for GOOGLE_API_KEY)
load_dotenv()
initialize_session_state()

# --- Sidebar for API Key and App Info ---
with st.sidebar:
    st.header("Configuration")
    st.info("This AI advisor helps you compare two insurance policies and recommends the best one for your needs.")
    
    # Allow user to input API key if not found in environment
    if 'GOOGLE_API_KEY' not in os.environ:
        google_api_key = st.text_input("Enter your Google API Key", type="password")
        if google_api_key:
            os.environ['GOOGLE_API_KEY'] = google_api_key
    
    st.button("Start New Comparison", on_click=reset_session, use_container_width=True)
    st.markdown("---")
    st.markdown("InsuraLens")


# --- Main Content Area based on Application Stage ---

st.title("InsuraLens - A Multi-Insurance Clause Analyzer & Recommendation Chat Agent.")

# Check for API Key before proceeding
if "GOOGLE_API_KEY" not in os.environ or not os.environ["GOOGLE_API_KEY"]:
    st.warning("Please enter your Google API Key in the sidebar to begin.")
    st.stop()

# STAGE 1: INITIAL - File Upload
if st.session_state.app_state["stage"] == "initial":
    st.subheader("Step 1: Upload Your Insurance Policies")
    uploaded_files = st.file_uploader(
        "Please upload 2 insurance policy documents (PDFs) for comparison.",
        type="pdf",
        accept_multiple_files=True
    )
    
    if uploaded_files and len(uploaded_files) == 2:
        st.session_state.app_state["uploaded_files"] = uploaded_files
        if st.button("Process Documents"):
            st.session_state.app_state["stage"] = "processing"
            st.rerun()
    elif uploaded_files:
        st.error("Please upload exactly two documents.")

# STAGE 2: PROCESSING - Backend work
elif st.session_state.app_state["stage"] == "processing":
    with st.spinner("Processing documents... This involves extracting text, creating chunks, and building a searchable database. Please wait."):
        # Get cached resources
        llm = cached_get_llm()
        embedding_model = cached_get_embedding_model()
        
        # Save files temporarily and get paths
        session_temp_dir = os.path.join(TEMP_UPLOADS_DIR, st.session_state.session_id)
        os.makedirs(session_temp_dir, exist_ok=True)
        
        saved_paths = []
        doc_names = []
        for file in st.session_state.app_state["uploaded_files"]:
            path = os.path.join(session_temp_dir, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            saved_paths.append(path)
            doc_names.append(file.name)
        
        st.session_state.app_state["doc_names"] = doc_names
        
        # Process PDFs and create vector store
        all_chunks = []
        full_text_for_classification = ""
        for i, path in enumerate(saved_paths):
            chunks = process_pdf(path)
            all_chunks.extend(chunks)
            if i == 0: # Use text from the first doc for classification
                full_text_for_classification = " ".join([chunk.page_content for chunk in chunks])

        # Use a unique collection name for each session
        collection_name = f"policies-{st.session_state.session_id}"
        vector_store = create_vector_store(all_chunks, embedding_model, collection_name)
        
        # Get policy type and questions
        policy_type = get_policy_type(full_text_for_classification, llm)
        questions = get_contextual_questions(policy_type, llm)
        
        # Store results in session state
        st.session_state.app_state["vector_store"] = vector_store
        st.session_state.app_state["policy_type"] = policy_type
        st.session_state.app_state["questions"] = questions
        st.session_state.app_state["stage"] = "questions"
        st.rerun()

# STAGE 3: QUESTIONS - User Input
elif st.session_state.app_state["stage"] == "questions":
    st.subheader(f"Step 2: Tell Us About Your Needs")
    st.info(f"We've identified these as **{st.session_state.app_state['policy_type']} Insurance** policies. Please answer a few questions for a personalized recommendation.")
    
    with st.form("user_questions_form"):
        user_answers = {}
        for question in st.session_state.app_state["questions"]:
            user_answers[question] = st.text_input(question)
        
        submitted = st.form_submit_button("Generate Analysis")
        if submitted:
            # Check if all questions are answered
            if all(user_answers.values()):
                st.session_state.app_state["user_answers"] = user_answers
                st.session_state.app_state["stage"] = "report"
                st.rerun()
            else:
                st.error("Please answer all questions before proceeding.")

# STAGE 4: REPORT - Display Final Analysis
elif st.session_state.app_state["stage"] == "report":
    if not st.session_state.app_state["final_report"]:
        with st.spinner("Our AI expert is analyzing your documents and profile... This may take a moment."):
            llm = cached_get_llm()
            final_report = generate_analysis_and_recommendation(
                vector_store=st.session_state.app_state["vector_store"],
                user_data=st.session_state.app_state["user_answers"],
                policy_type=st.session_state.app_state["policy_type"],
                doc_names=st.session_state.app_state["doc_names"],
                llm=llm
            )
            st.session_state.app_state["final_report"] = final_report

    st.subheader("Step 3: Your Personalized Insurance Analysis")
    st.markdown(st.session_state.app_state["final_report"])
    
    # The next step would be to add the conversational chatbot here.
    st.info("You can now ask follow-up questions (adding soon here!).")