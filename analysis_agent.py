import os
import json
from typing import List, Dict
import dotenv

# LangChain and Gemini components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

# Load environment variables from .env file
dotenv.load_dotenv()

# Import from our previous modules for the example
from document_processor import process_pdf, get_embedding_model
from vector_store_manager import create_vector_store

# --- LLM Configuration ---
def get_llm() -> ChatGoogleGenerativeAI:
    """Initializes and returns the Gemini Pro LLM."""
    # Ensure the API key is set
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it.")
    
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
- Policy 1 Name: {doc1_name}
- Policy 2 Name: {doc2_name}

**USER PROFILE:**
The user has provided the following information about their needs. This is crucial for your recommendation.
- Policy Type: {policy_type}
- User's Answers: {user_data}

**YOUR TASK (Follow this structure precisely):**

**1. Overall Summary:**
Briefly summarize the core offering of each policy in 2-3 sentences.

**2. Detailed Feature Comparison:**
Create a markdown table comparing the key features of both policies side-by-side. Include features like: Coverage Amount, Premium, Co-payment clauses, Waiting Periods, Key Exclusions, etc. Be factual and extract data from the context.

**3. Analysis of Hidden Clauses & Red Flags:**
For each policy, identify and explain any potentially confusing, restrictive, or unfavorable clauses that a typical user might overlook. Look for things like sub-limits, strict claim conditions, or ambiguous definitions. Use a "🚩" emoji for each red flag.

**4. Personalized Recommendation:**
Based on the user's profile and your analysis, provide a clear recommendation.
- State which policy is a better fit for the user and provide at least 3 detailed, evidence-based reasons for your choice.
- Also, explain why the other policy is less suitable for this specific user.
- Your reasoning must directly connect the policy features to the user's provided answers.

Structure your entire response in clear, easy-to-read markdown. Also note that do not use industry jargon or complex terms. The user is not an insurance expert and needs a straightforward explanation.
"""

def generate_analysis_and_recommendation(
    vector_store: Chroma,
    user_data: Dict,
    policy_type: str,
    doc_names: List[str],
    llm: ChatGoogleGenerativeAI
) -> str:
    """Generates the final comprehensive analysis and recommendation report."""
    
    # 1. Retrieve relevant context from the vector store
    # We query for a larger 'k' value to give the LLM more context
    retriever = vector_store.as_retriever(search_kwargs={"k": 20})
    # We can join all retrieved docs into a single string
    context_docs = retriever.get_relevant_documents(query=f"full summary of {policy_type} insurance policy")
    context_str = "\n\n---\n\n".join([doc.page_content for doc in context_docs])
    
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