# In chat_agent.py

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict, Tuple, Any

# --- 1. Chain to Route Question to a Specific Document ---

# --- FIX 1: Changed {question} to {input} for consistency ---
ROUTER_PROMPT_TEMPLATE = """
You are an expert at reading a user's question and routing it to the correct document.
Based on the user's question, which of the following documents is most relevant?

Documents:
{doc_names}

If the question is about "policy 1", "the first policy", or mentions "{doc1_name}", output "{doc1_name}".
If the question is about "policy 2", "the second policy", or mentions "{doc2_name}", output "{doc2_name}".
If the question is comparative or mentions "both", output "both".
Otherwise, output "both".

Return only the document name or "both" and nothing else.

Question: {input}
"""

# --- 2. History-Aware Prompt for Reformulating Questions (No changes needed) ---
HISTORY_AWARE_PROMPT = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        (
            "user",
            "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Don't add any introductory text, just the query.",
        ),
    ]
)

# --- 3. Main Prompt for Answering the Question (No changes needed) ---
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert AI insurance assistant. Your goal is to answer the user's questions accurately and conversationally based on the provided context.

User Profile for Personalization:
{user_profile}

Instructions:
- Base your answer strictly on the provided 'Context' from the documents. Do not make up information.
- If the answer is not in the Context, state that you cannot find the information in the provided documents.
- Keep your answers concise and easy to understand.
- Refer to the User Profile when relevant (e.g., "For a person of your age...").

Context:
{context}""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ]
)

# --- Helper Function to Convert Tuples to Message Objects ---
def convert_tuples_to_messages(chat_history: List[Tuple[str, str]]):
    """Convert chat history from (role, content) tuples to LangChain message objects."""
    messages = []
    for role, content in chat_history:
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages

# --- The Main Function to Create the Intelligent Agent Chain ---

def create_intelligent_agent_chain(
    llm: ChatGoogleGenerativeAI,
    vector_store: QdrantVectorStore,
    doc_names: List[str],
    user_profile: Dict,
):
    """
    Creates a sophisticated, history-aware, and document-specific retrieval chain.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    user_profile_str = "\n".join([f"- {key}: {value}" for key, value in user_profile.items()])
    
    # 1. Document Router Chain
    doc_router_prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT_TEMPLATE)
    doc_router = (
        {"doc_names": lambda x: "\n".join(doc_names),
         "doc1_name": lambda x: doc_names[0],
         "doc2_name": lambda x: doc_names[1],
         "input": lambda x: x["input"]}
        | doc_router_prompt
        | llm
        | (lambda x: x.content.strip())
    )

    # 2. History-Aware Retriever - with message format handling
    def create_custom_history_aware_retriever(retriever):
        def retrieve_with_history(inputs: Dict[str, Any]) -> List:
            # Convert tuple chat history to message objects if needed
            if "chat_history" in inputs and inputs["chat_history"] and isinstance(inputs["chat_history"][0], tuple):
                history = convert_tuples_to_messages(inputs["chat_history"])
                modified_inputs = {**inputs, "chat_history": history}
                return history_aware_retriever.invoke(modified_inputs)
            return history_aware_retriever.invoke(inputs)
        
        history_aware_retriever = create_history_aware_retriever(llm, retriever, HISTORY_AWARE_PROMPT)
        return RunnableLambda(retrieve_with_history)

    # 3. Document-Specific Retriever
    def get_relevant_documents(inputs: dict) -> dict:
        """
        Dynamically selects a retriever based on the router's output,
        then retrieves documents.
        """
        target_doc = doc_router.invoke({"input": inputs["input"]})
        
        if target_doc != "both" and target_doc in doc_names:
            print(f"--- Routing to document: {target_doc} ---")
            # For Qdrant, we use filter in search_kwargs
            filtered_retriever = vector_store.as_retriever(
                search_kwargs={
                    "k": 10, 
                    "filter": {"source": target_doc}
                }
            )
        else:
            print("--- Routing to both documents ---")
            filtered_retriever = retriever
        
        # Now create a history-aware retriever with the chosen (or default) retriever
        specific_history_aware_retriever = create_custom_history_aware_retriever(filtered_retriever)
        return specific_history_aware_retriever.invoke(inputs)

    # 4. Answering Chain
    answer_generation_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)

    # 5. Final Conversational Chain
    conversational_retrieval_chain = create_retrieval_chain(
        retriever=RunnableLambda(get_relevant_documents),
        combine_docs_chain=answer_generation_chain
    )

    # Add the user profile to the inputs for the final chain
    chain_with_profile = RunnablePassthrough.assign(
        user_profile=lambda x: user_profile_str
    ) | conversational_retrieval_chain

    return chain_with_profile