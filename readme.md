# 🛡️InsuraLens: A Smart Insurance Policy Analysis, Recommendation and Conversational Agent

> InsuraLens - an AI-powered insurance companion that simplifies policy comparison, identifies hidden clauses, and recommends the best plan based on your unique needs.

---

## Overview

InsuraLens is a GenAI-powered agent designed for the BFSI (Banking, Financial Services, and Insurance) sector. It empowers users to:

- Upload two insurance policies (Health, Life, Car, Bike, Travel).
- Automatically extract and compare terms & conditions.
- Identify hidden clauses like waiting periods, co-pays, exclusions, and add-ons.
- Get personalized policy recommendations based on their specific use case.
- Interact with a conversational agent (chatbot) to clarify doubts and explore policies.

---

## Problem Statement

Understanding insurance policy documents is challenging for most consumers. Complex jargon, hidden clauses, and misleading benefits make it difficult to make informed decisions.

**InsuraLens** solves this by acting as an intelligent intermediary between the user and the policy documents.

---

## Key Features

-   **Document Comparison**: Upload and compare two insurance policies side-by-side
-   **Intelligent Analysis**: AI-driven extraction of key policy features and hidden clauses
-   **Personalized Recommendations**: Customized analysis based on user needs and preferences
-   **Interactive Chat**: Ask follow-up questions about specific policy details
-   **Multiple Insurance Types**: Support for Health, Motor, Life, Travel, and Property insurance
-   **Sample Policies**: Test functionality with included sample documents

---

## System Architecture

InsuraLens uses a modular architecture combining several AI technologies:

1.  **PDF Processing**: Extracts and chunks text from insurance documents
2.  **Vector Database**: Creates searchable embeddings using Qdrant Cloud
3.  **LLM Integration**: Utilizes Google's Gemini models for intelligent analysis
4.  **Streamlit Interface**: Provides an intuitive web-based user experience  
 
---

## How It Works

1.  **User selects insurance type**
2.  **Uploads 2 policy documents**
3.  **Agent extracts clauses and stores them in Qdrant Cloud**
4.  **User enters needs (e.g., age, illness, car model)**
5.  **LLM compares policies, explains pros/cons, and gives recommendation**
6.  **User interacts via chatbot for queries like:**
- "Does Policy B cover diabetes?"
- "Which one has shorter waiting period?"
- "Is co-pay applicable in Policy A?"

---

## How to Run Locally

### 1. Clone this Repo

```bash
git  clone  https://github.com/agrawal-yash/InsuraLens.git
cd  InsuraLens
```

### 2. Create a Virtual Environment

```bash
python  -m  venv  venv

source  venv/bin/activate  # Linux/macOS
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip  install  -r  requirements.txt
```

### 4. Configure Secrets

#### For Local Development:
Create a `.streamlit/secrets.toml` file in your project directory:

```toml
# .streamlit/secrets.toml
GOOGLE_API_KEY = "your_google_api_key_here"
QDRANT_URL = "your_qdrant_cloud_url"
QDRANT_API_KEY = "your_qdrant_api_key"
```

**Important:** Never commit the `secrets.toml` file to version control!

#### For Streamlit Cloud Deployment:
1. Go to [Streamlit Cloud](https://share.streamlit.io/)
2. Deploy your app
3. In your app dashboard, click on "Settings" → "Secrets"
4. Add your secrets in this format:
```toml
GOOGLE_API_KEY = "your_google_api_key_here"
QDRANT_URL = "your_qdrant_cloud_url"
QDRANT_API_KEY = "your_qdrant_api_key"
```

#### Alternative: Environment Variables
If you prefer, you can still use a `.env` file (keep for backward compatibility):
```
GOOGLE_API_KEY=your_google_api_key
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
```

### 5. Run the Streamlit App

```bash
streamlit  run  app.py
```
The application will be accessible at http://localhost:8501

---

## Usage Guide

### 1. Upload Documents

-   Upload two insurance policy PDFs for comparison
-   Alternatively, use the provided sample documents

### 2. Answer Profile Questions

-   The system will identify the policy type and ask relevant questions
-   Your answers will help customize the analysis to your needs

### 3. Review Analysis

-   View a comprehensive side-by-side comparison
-   See highlighted red flags and hidden clauses
-   Get a personalized recommendation with justification

### 4. Ask Follow-up Questions

-   Use the chat interface to inquire about specific aspects
-   Reference policies by name (e.g., "What is the claim process for policy 1?")

---

## Impact

-  **Improves transparency** in policy buying
-  **Reduces misinformation and regret** for buyers
-  **Empowers 40–50%** of all policyholders in India with better decision-making

---

## Technical Details

### Document Processing

-   Documents are processed using PyMuPDF and LangChain's UnstructuredFileLoader
-   Text is split into chunks using RecursiveCharacterTextSplitter
-   Document metadata includes page numbers and source information

### Vector Database

-   Uses Qdrant Cloud for scalable vector storage and retrieval
-   Embedding model: "all-MiniLM-L6-v2" from HuggingFace
-   Collections are session-specific and cleaned up after use
-   Supports advanced filtering and similarity search

### LLM Integration

-   Primary model: Google's Gemini 2.0 Flash
-   Used for policy type classification, question generation, and analysis
-   Controlled temperature (0.2) for consistent outputs  

---

## 🧠 Future Enhancements

- OCR for scanned PDFs
- Multilingual support (Hindi, Marathi, etc.)
- Mobile version (Flutter/React Native)
- Export report as downloadable PDF
- Add more insurance domains (Home, Pet, Gadget)