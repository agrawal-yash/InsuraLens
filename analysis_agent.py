import os
import json
from typing import List, Dict
import streamlit as st

# LangChain and Gemini components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import QdrantVectorStore
from langchain.docstore.document import Document

# Import from our previous modules for the example
from document_processor import process_pdf, get_embedding_model
from vector_store_manager import create_vector_store

def get_google_api_key():
    """Get Google API key from Streamlit secrets or environment variables."""
    try:
        # Try Streamlit secrets first
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if api_key:
            return api_key
    except Exception:
        pass
    
    # Fallback to environment variable
    return os.environ.get("GOOGLE_API_KEY")

# --- LLM Configuration ---
def get_llm() -> ChatGoogleGenerativeAI:
    """Initializes and returns the Gemini Pro LLM."""
    # Get API key from secrets or environment
    api_key = get_google_api_key()
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in Streamlit secrets or environment variables. Please set it.")
    
    # Set the environment variable for the LangChain library
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Using Gemini Pro for this task
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)


# --- 1. Tool for Identifying Policy Type ---

POLICY_TYPE_PROMPT = """
Based on the following text from an insurance policy document, please identify the primary type of insurance.
The answer should be one of the following categories: Health, Motor, Life, Travel, or Property.
Return only the single word category name and nothing else.

TEXT:
"{policy_text}"
"""

def get_policy_type(policy_text: str, llm: ChatGoogleGenerativeAI) -> str:
    """Uses an LLM to classify the insurance policy type."""
    prompt = PromptTemplate.from_template(POLICY_TYPE_PROMPT)
    chain = prompt | llm | StrOutputParser()
    
    # We only need a small snippet of text to classify
    snippet = policy_text[:2000]
    
    policy_type = chain.invoke({"policy_text": snippet}).strip()
    # Basic validation
    if policy_type not in ["Health", "Motor", "Life", "Travel", "Property"]:
        return "General" # Fallback category
    return policy_type


# --- 2. Tool for Generating Contextual Questions ---

QUESTIONS_PROMPT = """
You are an expert insurance advisor assistant. Based on the provided insurance policy type, please generate a list of 3-5 essential questions to ask the user to help them choose the best policy.
These questions should gather personal information relevant to the policy type to enable a personalized recommendation.

Format the output as a valid JSON object where the key is "questions" and the value is a list of strings.
Example for Health Insurance:
{{
  "questions": [
    "What is your age?",
    "Do you have any pre-existing medical conditions? If so, please specify.",
    "Which family members need to be covered under this policy?",
    "What is your preferred hospital network (if any)?"
  ]
}}

Policy Type: {policy_type}
"""

def get_contextual_questions(policy_type: str, llm: ChatGoogleGenerativeAI) -> List[str]:
    """Generates user-specific questions based on the policy type."""
    prompt = PromptTemplate.from_template(QUESTIONS_PROMPT)
    chain = prompt | llm | StrOutputParser()
    
    response_text = chain.invoke({"policy_type": policy_type})
    
    try:
        # The LLM output might have markdown ```json ... ```, so we clean it
        json_str = response_text.strip().replace("```json", "").replace("```", "")
        questions_data = json.loads(json_str)
        return questions_data.get("questions", [])
    except json.JSONDecodeError:
        print("Error: Could not parse JSON from LLM response for questions.")
        return []

# --- 3. Main Tool for Generating the Final Report ---

ANALYSIS_PROMPT_TEMPLATE = """
You are an expert AI insurance advisor. Your task is to provide a detailed, unbiased, SIMPLE and personalized comparison of two insurance policies for a user.

**CONTEXT FROM POLICIES:**
Here is the relevant information extracted from the two policy documents. Use this as your primary source of truth.
{context}

**POLICY DOCUMENTS:**
- Policy 1 : {doc1_name}
- Policy 2 : {doc2_name}
Extract the names of the policies from the documents and use them in your analysis.

**USER PROFILE:**
The user has provided the following information about their needs. This is crucial for your recommendation.
- Policy Type: {policy_type}
- User's Answers: {user_data}

**YOUR TASK (Follow this structure precisely):**

**1. Overall Summary:**
Briefly summarize the core offering of each policy in 2-3 sentences. Dont keep in a very plain format, but make it clear and concise.

**2. Detailed Feature Comparison:**
Create a markdown table comparing the key features of both policies side-by-side. Include features like: Coverage Amount, Premium, Co-payment clauses, Waiting Periods, Key Exclusions, etc. Be factual and extract data from the context. Remember that some terms may be complex, so explain them in simple terms if necessary. Also there might be some terms that are the same but are named differently in each policy, so be sure to clarify that.
Use this exact markdown format for the table, replacing "Policy 1 Name" and "Policy 2 Name" with the actual document names:
| Feature | Policy 1 Name | Policy 2 Name |
| :--- | :--- | :--- |
| **Coverage Amount** | [Detail from Policy 1] | [Detail from Policy 2] |
| **Premium** | [Detail from Policy 1] | [Detail from Policy 2] |
| **Co-payment** | [Detail from Policy 1] | [Detail from Policy 2] |

**3. Analysis of Hidden Clauses & Red Flags:**
For each policy, identify and explain any potentially confusing, restrictive, or unfavorable clauses that a typical user might overlook. Look for things like sub-limits, strict claim conditions, or ambiguous definitions. Use a "🚩" emoji for each red flag. 

**4. Personalized Recommendation:**
Based on the user's profile and your analysis, provide a clear recommendation.
- State which policy is a better fit for the user and provide at least 3 detailed, evidence-based reasons for your choice.
- Also, explain why the other policy is less suitable for this specific user.
- Your reasoning must directly connect the policy features to the user's provided answers.

Structure your entire response in clear, easy-to-read and simple to understand markdown.
"""

def generate_analysis_and_recommendation(
    vector_store: QdrantVectorStore,
    user_data: Dict,
    policy_type: str,
    doc_names: List[str],
    llm: ChatGoogleGenerativeAI
) -> str:
    """Generates the final comprehensive analysis and recommendation report."""
    
    # 1. Retrieve relevant context from the vector store for each document
    retriever = vector_store.as_retriever(search_kwargs={"k": 10}) # k=10 for each doc
    
    # Retrieve for doc 1
    query1 = f"full summary of policy '{doc_names[0]}' for {policy_type} insurance"
    context_docs1 = retriever.get_relevant_documents(query=query1)
    
    # Retrieve for doc 2
    query2 = f"full summary of policy '{doc_names[1]}' for {policy_type} insurance"
    context_docs2 = retriever.get_relevant_documents(query=query2)

    # Combine contexts
    context_str1 = "\n\n".join([doc.page_content for doc in context_docs1])
    context_str2 = "\n\n".join([doc.page_content for doc in context_docs2])
    context_str = f"--- CONTEXT FOR {doc_names[0]} ---\n{context_str1}\n\n--- CONTEXT FOR {doc_names[1]} ---\n{context_str2}"

    # 2. Create the prompt
    prompt = PromptTemplate.from_template(ANALYSIS_PROMPT_TEMPLATE)
    
    # 3. Create the chain and invoke
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "context": context_str,
        "doc1_name": doc_names[0],
        "doc2_name": doc_names[1],
        "policy_type": policy_type,
        "user_data": json.dumps(user_data, indent=2)
    })
    
    return response