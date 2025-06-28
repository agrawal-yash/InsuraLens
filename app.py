import streamlit as st
import os
import uuid
import shutil
from dotenv import load_dotenv

# Import your custom modules
from document_processor import process_pdf, get_embedding_model
from vector_store_manager import create_vector_store, DB_DIRECTORY
from analysis_agent import get_llm, get_policy_type, get_contextual_questions, generate_analysis_and_recommendation
from chat_agent import create_intelligent_agent_chain

# --- Constants ---
TEMP_UPLOADS_DIR = "temp_uploads"
SAMPLE_POLICY_1 = "sample-care-supreme.pdf"
SAMPLE_POLICY_2 = "sample-superstar.pdf"

# --- Page Configuration ---
st.set_page_config(
    page_title="InsuraLens: A Smart Insurance Policy Analysis, Recommendation and Conversational Agent",
    page_icon="🤖",
    layout="wide"
)

# --- Caching Functions for Expensive Resources ---

@st.cache_resource
def cached_get_llm():
    """Caches the LLM resource. This function is run only once."""
    return get_llm()

@st.cache_resource
def cached_get_embedding_model():
    """Caches the embedding model resource. This function is run only once."""
    return get_embedding_model()

# --- Session State Management ---

def initialize_session_state():
    """Initializes all the necessary variables in the session state for a new session."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'app_state' not in st.session_state:
        st.session_state.app_state = {
            "stage": "initial",
            "uploaded_files": None,
            "policy_type": None,
            "questions": [],
            "user_answers": {},
            "final_report": None,
            "vector_store": None,
            "doc_names": [],
            "saved_paths": [],
            "chat_history": [],
            "conversational_chain": None
        }

def reset_session():
    """Resets the session to its initial state and cleans up old data."""
    # --- FIX: Explicitly release objects holding file locks before deleting ---
    # These objects (especially vector_store) can hold locks on the db files.
    if "vector_store" in st.session_state.app_state:
        del st.session_state.app_state["vector_store"]
    if "conversational_chain" in st.session_state.app_state:
        del st.session_state.app_state["conversational_chain"]

    # Clean up the ChromaDB directory. This is the simplest robust way.
    if os.path.exists(DB_DIRECTORY):
        shutil.rmtree(DB_DIRECTORY)

    #clean up the chat history
    st.session_state.app_state["chat_history"] = []
    
    # Clean up temporary uploaded files
    temp_dir_path = os.path.join(TEMP_UPLOADS_DIR, st.session_state.session_id)
    if os.path.exists(temp_dir_path):
        shutil.rmtree(temp_dir_path)

    # Re-initialize the session state by clearing the app_state and getting a new ID
    # DO NOT clear st.cache_resource here, as it's shared across all sessions.
    st.session_state.session_id = str(uuid.uuid4())
    del st.session_state.app_state
    initialize_session_state() # Re-initialize with default values
    st.rerun()


# --- Main Application Logic ---

# Load environment variables (for GOOGLE_API_KEY)
load_dotenv()
initialize_session_state()

# --- Sidebar ---
with st.sidebar:
    st.title("InsuraLens")
    st.info("A Smart Insurance Policy Analysis, Recommendation and Conversational Agent")
    
    if 'GOOGLE_API_KEY' not in os.environ:
        google_api_key = st.text_input("Enter your Google API Key", type="password", key="api_key_input")
        if google_api_key:
            os.environ['GOOGLE_API_KEY'] = google_api_key
    
    st.button("Start New Comparison", use_container_width=True)
    st.markdown("---")

# --- Main Content Area ---
st.title("InsuraLens: A Smart Insurance Policy Analysis, Recommendation and Conversational Agent")
st.caption("Compare two policies, get a personalized report, and ask follow-up questions.")

# Check for API Key before proceeding
if "GOOGLE_API_KEY" not in os.environ or not os.environ["GOOGLE_API_KEY"]:
    st.warning("Please enter your Google API Key in the sidebar to begin.", icon="🔑")
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
        if st.button("Analyze Policies", use_container_width=True):
            st.session_state.app_state["stage"] = "processing"
            st.rerun()
    elif uploaded_files:
        st.error("Please upload exactly two documents to compare.")

    # --- NEW: Section for using sample documents ---
    st.divider()
    st.subheader("Don't have documents? Test with our samples:")
    
    # Check if sample files exist before showing the button
    if os.path.exists(SAMPLE_POLICY_1) and os.path.exists(SAMPLE_POLICY_2):
        if st.button("Use Sample Health Policies", use_container_width=True):
            # Set the file paths directly, bypassing the upload step
            st.session_state.app_state["file_paths_to_process"] = [SAMPLE_POLICY_1, SAMPLE_POLICY_2]
            st.session_state.app_state["stage"] = "processing"
            st.rerun()
    else:
        st.warning("Sample files not found. Please ensure the sample pdfs are in the project directory.")

# STAGE 2: PROCESSING - Backend work
elif st.session_state.app_state["stage"] == "processing":
    with st.spinner("Processing documents... This involves extracting text, creating chunks, and building a searchable database. Please wait."):
        llm = cached_get_llm()
        embedding_model = cached_get_embedding_model()
        
        # --- NEW: Logic to handle both uploaded files and sample files ---
        # Check if we are using the pre-defined sample paths
        if st.session_state.app_state.get("file_paths_to_process"):
            saved_paths = st.session_state.app_state["file_paths_to_process"]
            doc_names = [os.path.basename(p) for p in saved_paths]
        else: # Otherwise, handle user-uploaded files
            session_temp_dir = os.path.join(TEMP_UPLOADS_DIR, st.session_state.session_id)
            os.makedirs(session_temp_dir, exist_ok=True)
            
            saved_paths, doc_names = [], []
            for file in st.session_state.app_state["uploaded_files"]:
                path = os.path.join(session_temp_dir, file.name)
                with open(path, "wb") as f:
                    f.write(file.getbuffer())
                saved_paths.append(path)
                doc_names.append(file.name)
        
        st.session_state.app_state["doc_names"] = doc_names
        st.session_state.app_state["saved_paths"] = saved_paths
        
        all_chunks, full_text_for_classification = [], ""
        for i, path in enumerate(saved_paths):
            chunks = process_pdf(path)
            all_chunks.extend(chunks)
            if i == 0 and chunks:
                full_text_for_classification = " ".join([chunk.page_content for chunk in chunks])

        if not all_chunks:
            st.error("Could not extract text from the provided PDFs. Please try different documents or use the samples.")
            st.session_state.app_state["stage"] = "initial"
            st.rerun()
        
        collection_name = f"policies-{st.session_state.session_id}"
        vector_store = create_vector_store(all_chunks, embedding_model, collection_name)
        
        policy_type = get_policy_type(full_text_for_classification, llm)
        questions = get_contextual_questions(policy_type, llm)
        
        st.session_state.app_state.update({
            "vector_store": vector_store,
            "policy_type": policy_type,
            "questions": questions,
            "stage": "questions"
        })
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
            if all(user_answers.values()):
                st.session_state.app_state["user_answers"] = user_answers
                st.session_state.app_state["stage"] = "report"
                st.rerun()
            else:
                st.error("Please answer all questions before proceeding.")

# STAGE 4: REPORT & CHAT - Display Final Analysis and Start Conversation
elif st.session_state.app_state["stage"] == "report":
    if not st.session_state.app_state["final_report"]:
        with st.spinner("Our AI expert is drafting your personalized analysis... This may take a moment."):
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

    # --- Download Buttons ---
    st.subheader("Download Your Documents")
    cols = st.columns(len(st.session_state.app_state["saved_paths"]))
    for i, path in enumerate(st.session_state.app_state["saved_paths"]):
        with open(path, "rb") as f:
            with cols[i]:
                st.download_button(
                    label=f"Download {st.session_state.app_state['doc_names'][i]}",
                    data=f,
                    file_name=st.session_state.app_state['doc_names'][i],
                    mime="application/pdf",
                    use_container_width=True
                )

    st.markdown("---")
    
    st.subheader("Step 4: Ask Follow-up Questions")
    
    # --- CHAT LOGIC MOVED HERE ---
    # Initialize chain if it doesn't exist
    if not st.session_state.app_state.get("conversational_chain"):
        st.session_state.app_state["conversational_chain"] = create_intelligent_agent_chain(
            llm=cached_get_llm(),
            vector_store=st.session_state.app_state["vector_store"],
            doc_names=st.session_state.app_state["doc_names"],
            user_profile=st.session_state.app_state["user_answers"]
        )

    # Display previous messages
    if st.session_state.app_state["chat_history"]:
            # Unpack the tuple directly in the loop
            for role, content in st.session_state.app_state["chat_history"]:
                # Use the 'role' variable directly
                with st.chat_message(role):
                    # Use the 'content' variable directly
                    st.markdown(content)

    # Handle new user input
    if prompt := st.chat_input("Ask about 'policy 1', 'policy 2', or 'both'..."):
        # We need to manually add the user message to the chat history for the new chain
        st.session_state.app_state["chat_history"].append(("user", prompt))
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain = st.session_state.app_state["conversational_chain"]
                
                # The new chain expects a dictionary with specific keys
                result = chain.invoke({
                    "chat_history": st.session_state.app_state["chat_history"],
                    "input": prompt
                })
                
                response = result
                st.markdown(response)
        
        # Manually add the AI response to the history
        st.session_state.app_state["chat_history"].append(("ai", response))