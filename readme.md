# 🛡️InsuraLens - A Multi-Insurance Clause Analyzer & Recommendation GenAI Agent

> InsuraLens - an AI-powered insurance companion that simplifies policy comparison, identifies hidden clauses, and recommends the best plan based on your unique needs.

---

## 🚀 Overview

InsuraLens is a GenAI-powered agent designed for the BFSI (Banking, Financial Services, and Insurance) sector. It empowers users to:
- Upload two insurance policies (Health, Life, Car, Bike, Travel).
- Automatically extract and compare terms & conditions.
- Identify hidden clauses like waiting periods, co-pays, exclusions, and add-ons.
- Get personalized policy recommendations based on their specific use case.
- Interact with a conversational agent (chatbot) to clarify doubts and explore policies.

---

## 🧠 Problem Statement

Understanding insurance policy documents is challenging for most consumers. Complex jargon, hidden clauses, and misleading benefits make it difficult to make informed decisions.

**InsuraLens** solves this by acting as an intelligent intermediary between the user and the policy documents.

---

## 🎯 Key Features

✅ Multi-Insurance Support (Health, Life, Car, Bike, Travel)  
✅ Upload & Parse 2 Policy Documents (PDF)  
✅ GenAI-Powered Clause Extraction (e.g., Waiting Periods, Exclusions)  
✅ Smart Policy Comparison Based on Use Case  
✅ Follow-up Chatbot (Clause Queries, Explanations)  
✅ Visual Recommendations with Reasons  
✅ Hosted UI built with Streamlit


---

## 🖥️ Tech Stack

Streamlit, Pymupdf, Unstructured.io, sentence-transformers, ChromaDB, Gemini 2.0 Flash, Langchain


----------


## 🔍 How It Works

1.  **User selects insurance type**
    
2.  **Uploads 2 policy documents**
    
3.  **Agent extracts clauses and stores them in vector DB**
    
4.  **User enters needs (e.g., age, illness, car model)**
    
5.  **LLM compares policies, explains pros/cons, and gives recommendation**
    
6.  **User interacts via chatbot for queries like:**
    
    -   “Does Policy B cover diabetes?”
        
    -   “Which one has shorter waiting period?”
        
    -   “Is co-pay applicable in Policy A?”
        

----------

## 🧪 How to Run Locally

### 1. Clone this Repo

```bash
git clone https://github.com/agrawal-yash/InsuraLens.git
cd InsuraLens

```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Add Environment Variables

Create a `.env` file for API keys:

```
GOOGLE_API_KEY=your_api_key

```

### 5. Run the Streamlit App

```bash
streamlit run app.py

```


## 📊 Impact

-   **Improves transparency** in policy buying
    
-   **Reduces misinformation and regret** for buyers
    
-   **Empowers 40–50%** of all policyholders in India with better decision-making
    

----------

## 🧠 Future Enhancements

-   🔁 OCR for scanned PDFs
    
-   🗣️ Multilingual support (Hindi, Marathi, etc.)
    
-   📲 Mobile version (Flutter/React Native)
    
-   📩 Export report as downloadable PDF
    
-   🧩 Add more insurance domains (Home, Pet, Gadget)
    
