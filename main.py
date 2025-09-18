"""
==================================================================================================
Hydra AI WhatsApp Bot - Final Consolidated Version with Advanced Visualization
==================================================================================================
Purpose:
This script launches a sophisticated, AI-powered chatbot that interacts with users via WhatsApp.
It is designed as a specialized assistant for Indian groundwater data and can handle
general knowledge questions, voice notes, and advanced data visualization.

Key Features:
- WhatsApp Integration: Connects to the Meta API to send and receive messages.
- Multi-Database RAG: Creates separate vector stores for specific district data (JSON) and
  general information (PDFs) for highly accurate, targeted answers.
- Advanced Semantic Routing: Intelligently selects the correct vector database based on the query.
- Per-User Conversational Memory: Maintains a separate, persistent conversation history for each user.
- Voice Transcription: Uses OpenAI's Whisper model to transcribe audio voice notes.
- Location Processing: Uses Nominatim to reverse geocode shared locations and fetch local data.
- User Onboarding: Greets new users and waits for a "start" command to activate the AI.
- Advanced Visualization: Generates charts and graphs from structured JSON data.

Setup Requirements:
1. `datasets` folder: Must contain the source JSON and PDF files.
2. `.env` file: Must be in the same directory with all required API keys and tokens.
3. `pip install`: All required libraries must be installed.
4. `ngrok`: Must be running to forward a public URL to the local port 5000.
==================================================================================================
"""
import os
import glob
import pandas as pd
import google.generativeai as genai
import requests
import json
import traceback
import tempfile
from flask import Flask, request, Response
from dotenv import load_dotenv
import re
from io import BytesIO

# Import Whisper for audio transcription
import whisper

# --- Imports for visualization ---
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for servers
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF

# Core LangChain imports
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Specialized LangChain component imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 1. CONFIGURATION & MODEL LOADING ---
load_dotenv()

print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded successfully.")

try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
    WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
    WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")
    if not GEMINI_API_KEY: raise ValueError("GEMINI_API_KEY not found in .env file.")
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API Key configured successfully.")
except ValueError as e:
    print(f"Error: {e}")
    GEMINI_API_KEY = None

KNOWLEDGE_BASE_DIR = "datasets"
VECTOR_STORE_PARENT_DIR = "vector_stores"

# Map districts to their specific JSON files
DISTRICT_JSON_MAP = {
    "erode": "erode_taluk_data.json",
    "namakkal": "namakkal_groundwater_rag.json",
    "thoothukudi": "thoothukudi_groundwater_rag.json",
}

# Map specific taluks to their parent district
TALUK_DISTRICT_MAP = {
    "perundurai": "erode",
    "bhavani": "erode",
    "anthiyur": "erode",
    "gobichettipalayam": "erode", 
    "sathyamangalam": "erode", 
    "kodumudi": "erode", 
    "modakurichi": "erode", 
    "thalavadi": "erode"
}

# --- NEW: Path to structured JSON data file for visualization ---
SOURCE_JSON_PATH = os.path.join(KNOWLEDGE_BASE_DIR, "groundwater_data.json") 
groundwater_data = {}  # Global variable to hold the loaded JSON data


# --- 2. DATA LOADING & VECTOR DB ---
def flatten_json(data, location_name, parent_key=''):
    """A recursive generator to flatten nested JSON and yield facts."""
    for key, value in data.items():
        new_key = f"{parent_key} {key}".strip().replace('_', ' ')
        if isinstance(value, dict):
            yield from flatten_json(value, location_name, new_key)
        elif isinstance(value, list):
            fact = f"For the location '{location_name}', the '{new_key}' are: {', '.join(map(str, value))}."
            yield Document(
                page_content=fact,
                metadata={"source": f"{location_name}.json", "location": location_name}
            )
        elif isinstance(value, (str, int, float, bool)):
            fact = f"For the location '{location_name}', the value for '{new_key}' is {value}."
            yield Document(
                page_content=fact,
                metadata={"source": f"{location_name}.json", "location": location_name}
            )

def load_and_process_json(file_path):
    """Loads a single JSON file and processes it into fact-based documents."""
    all_docs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            # Handle list of records (like Namakkal/Thoothukudi)
            if isinstance(json_data, list):
                for i, record in enumerate(json_data):
                    location_name = record.get("locationName", f"Record {i+1}")
                    all_docs.extend(list(flatten_json(record, location_name)))
            # Handle dict of records (like Erode)
            elif isinstance(json_data, dict):
                for location_name, record in json_data.items():
                     if isinstance(record, dict):
                        all_docs.extend(list(flatten_json(record, location_name.upper())))
        print(f"   Processed data from {os.path.basename(file_path)}")
    except Exception as e:
        print(f"   - SKIPPED - Error processing {os.path.basename(file_path)}: {e}")
    return all_docs

def load_pdfs_from_directory(dir_path):
    """Loads and chunks all PDF documents from a directory."""
    print("-> Loading all PDF files for the general database...")
    all_docs = []
    pdf_files = glob.glob(os.path.join(dir_path, "**/*.pdf"), recursive=True)
    if not pdf_files:
        print("   No PDF files found.")
        return []
    
    for file_path in pdf_files:
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"   Loaded {len(docs)} pages from {os.path.basename(file_path)}")
        except Exception as e:
            print(f"   - SKIPPED - Error processing {os.path.basename(file_path)}: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)
    print(f"   Split {len(all_docs)} total pages into {len(chunks)} chunks.")
    return chunks


def create_or_load_all_vector_dbs():
    """
    Creates or loads vector stores: one for each JSON file and one for all PDFs.
    Returns a dictionary of retriever objects.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    retrievers = {}
    
    if not os.path.exists(VECTOR_STORE_PARENT_DIR):
        os.makedirs(VECTOR_STORE_PARENT_DIR)

    # Create a DB for each district JSON file
    for district, filename in DISTRICT_JSON_MAP.items():
        vector_store_path = os.path.join(VECTOR_STORE_PARENT_DIR, f"{district}_db")
        json_file_path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        if os.path.exists(vector_store_path):
            print(f"\nLoading existing vector store for '{district}'...")
            db = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        else:
            print(f"\nCreating new vector store for '{district}'...")
            if not os.path.exists(json_file_path):
                print(f"   WARNING: JSON file not found at {json_file_path}. Skipping.")
                continue
            all_docs = load_and_process_json(json_file_path)
            if not all_docs: continue
            db = FAISS.from_documents(all_docs, embeddings)
            db.save_local(vector_store_path)
        retrievers[district] = db.as_retriever(search_kwargs={"k": 10})
        print(f"   Retriever for '{district}' is ready.")

    # Create one combined DB for all PDF files
    pdf_vector_store_path = os.path.join(VECTOR_STORE_PARENT_DIR, "pdf_db")
    if os.path.exists(pdf_vector_store_path):
        print("\nLoading existing vector store for PDFs...")
        pdf_db = FAISS.load_local(pdf_vector_store_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("\nCreating new vector store for all PDFs...")
        pdf_docs = load_pdfs_from_directory(KNOWLEDGE_BASE_DIR)
        if pdf_docs:
            pdf_db = FAISS.from_documents(pdf_docs, embeddings)
            pdf_db.save_local(pdf_vector_store_path)
        else:
            pdf_db = None
    if pdf_db:
        retrievers["pdf_general"] = pdf_db.as_retriever(search_kwargs={"k": 5})
        print("   Retriever for general PDF documents is ready.")
        
    return retrievers


# --- 3. AI LOGIC AND ROUTING (LCEL) ---
class SemanticRouter:
    def __init__(self, general_chain, memory_manager, district_retrievers):
        self.general_chain = general_chain
        self.memory_manager = memory_manager
        self.retrievers = district_retrievers
        router_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GEMINI_API_KEY)
        
        # Updated prompt with 'visualization' category
        routing_prompt = ChatPromptTemplate.from_template(
            """You are an expert at classifying user questions for a specialized Indian groundwater chatbot.
            Classify the question into one of three categories: 'visualization', 'groundwater_data', or 'general_knowledge'.

            **'visualization' Category:**
            Choose this if the user is asking to "create a graph", "generate a report", "show a chart", "plot the data", "make a PDF", 
            "sustainability metrics", "draft composition", or "recharge sources" for a specific location.
            - Examples: "create a graph for Erode", "generate a pdf report for Namakkal district", "show me sustainability metrics"

            **'groundwater_data' Category:**
            Choose this for specific questions about data, not for creating visuals.
            - Examples: "What is the groundwater level in Erode?", "summarize the report for Tamil Nadu"

            **'general_knowledge' Category:**
            Choose this for all other questions.
            - Examples: "hello", "who are you?", "what is the capital of Japan?"

            Return only the single category name.
            User Question: "{question}"
            Classification:"""
        )
        self.router_chain = routing_prompt | router_llm

    def route(self, user_id, question):
        chat_history = self.memory_manager.get_history(user_id)
        
        question_lower = question.lower()
        
        matched_taluk = next((taluk for taluk in TALUK_DISTRICT_MAP.keys() if taluk in question_lower), None)
        matched_district = next((dist for dist in DISTRICT_JSON_MAP.keys() if dist in question_lower), None)

        # Convert chat history to the format expected by LangChain
        from langchain_core.messages import HumanMessage, AIMessage
        formatted_history = []
        for msg in chat_history:
            if hasattr(msg, 'type'):
                if msg.type == 'human':
                    formatted_history.append(HumanMessage(content=msg.content))
                elif msg.type == 'ai':
                    formatted_history.append(AIMessage(content=msg.content))
            elif hasattr(msg, 'get_type'):
                if msg.get_type() == 'human':
                    formatted_history.append(HumanMessage(content=msg.content))
                elif msg.get_type() == 'ai':
                    formatted_history.append(AIMessage(content=msg.content))
            else:
                # Fallback: check content structure
                if hasattr(msg, 'content'):
                    if hasattr(msg, 'role') and msg.role == 'user':
                        formatted_history.append(HumanMessage(content=msg.content))
                    elif hasattr(msg, 'role') and msg.role == 'assistant':
                        formatted_history.append(AIMessage(content=msg.content))
                    else:
                        # Default to treating as AI message if we can't determine
                        formatted_history.append(AIMessage(content=str(msg)))
        
        classification = self.router_chain.invoke({"question": question}).content.strip().lower()
        print(f"--- Routing classification: '{classification}' ---")

        retriever_to_use = None
        source_name = "General AI Knowledge"

        if "visualization" in classification:
            return {"source_type": "visualization", "result": {"text": question}}
        
        elif matched_taluk:
            parent_district = TALUK_DISTRICT_MAP[matched_taluk]
            print(f"--- Routing to: '{matched_taluk}' Taluk (via '{parent_district}' RAG) ---")
            retriever_to_use = self.retrievers.get(parent_district)
            source_name = f"JSON ({parent_district.title()})"
        elif matched_district:
            print(f"--- Routing to: '{matched_district}' District RAG ---")
            retriever_to_use = self.retrievers.get(matched_district)
            source_name = f"JSON ({matched_district.title()})"
        else:
            classification = self.router_chain.invoke({"question": question}).content.strip().lower()
            print(f"--- Routing classification: '{classification}' (LLM based) ---")
            if "groundwater_data" in classification:
                print("--- Routing to: General PDF RAG ---")
                retriever_to_use = self.retrievers.get("pdf_general")
                source_name = "PDF Documents"
            else:
                result = self.general_chain.invoke({"chat_history": formatted_history, "input": question})
                self.memory_manager.save_context(user_id, {"input": question}, {"output": result.content})
                return {"source_type": "gemini", "result": {"text": result.content}}

        if retriever_to_use:
            rag_chain = get_rag_chain(retriever_to_use)
            result = rag_chain.invoke({"chat_history": formatted_history, "input": question})
            self.memory_manager.save_context(user_id, {"input": question}, {"output": result["answer"]})
            result["source_name"] = source_name
            return {"source_type": "rag", "result": result}
        else:
            return {"source_type": "gemini", "result": {"text": "I can answer questions about specific locations. Please be more specific."}}

    
def get_rag_chain(retriever):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GEMINI_API_KEY)
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and a follow up question, rephrase the follow up question to be a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a factual assistant for Indian groundwater data. Use ONLY the following pieces of retrieved context to answer the question. If the context does not contain the answer, you MUST say 'I do not have enough information in the provided documents to answer this question.'\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

def get_general_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, google_api_key=GEMINI_API_KEY)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    return prompt | llm


# Update ALL visualization functions to use BytesIO instead of temporary files

def generate_extraction_rate_chart(chart_data):
    """Generates a bar chart for extraction rates."""
    try:
        df = pd.DataFrame(chart_data['extraction_rates'])
        df = df.sort_values(by='extraction_rate', ascending=False)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.bar(df['location'], df['extraction_rate'], 
                     color=[item['color'] for item in df.to_dict('records')])
        ax.set_title('Groundwater Extraction Rate by Location', fontsize=16)
        ax.set_ylabel('Extraction Rate (%)')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Use BytesIO instead of temporary files
        from io import BytesIO
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        plt.close(fig)  # Important: close the figure to free memory
        
        img_buffer.seek(0)
        image_bytes = img_buffer.getvalue()
        img_buffer.close()
        
        return image_bytes
    except Exception as e:
        print(f"Error generating extraction rate chart: {e}")
        traceback.print_exc()
        return None

def generate_sustainability_chart(chart_data):
    """Generates a bubble chart for sustainability metrics."""
    try:
        df = pd.DataFrame(chart_data['sustainability_metrics'])
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(14, 9))
        
        scatter = ax.scatter(df['x'], df['y'], s=df['size']/50, c=df['color'], 
                           alpha=0.7, edgecolors="w", linewidth=0.5)
        
        for i, txt in enumerate(df['location']):
            ax.annotate(txt, (df['x'][i], df['y'][i]), ha='center', va='center', fontsize=9)
            
        ax.set_title('Sustainability Metrics: Extraction vs. Sustainability Index', fontsize=16)
        ax.set_xlabel('Extraction Rate (%)')
        ax.set_ylabel('Sustainability Index (Recharge/Draft %)')
        ax.grid(True)
        plt.tight_layout()
        
        # Use BytesIO instead of temporary files
        from io import BytesIO
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        plt.close(fig)  # Important: close the figure to free memory
        
        img_buffer.seek(0)
        image_bytes = img_buffer.getvalue()
        img_buffer.close()
        
        return image_bytes
    except Exception as e:
        print(f"Error generating sustainability chart: {e}")
        traceback.print_exc()
        return None

def generate_bar_chart(data, title, x_col, y_col):
    """Generates a bar chart from a DataFrame and returns the image bytes."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        data_cleaned = data.dropna(subset=[x_col, y_col])
        data_cleaned[y_col] = pd.to_numeric(data_cleaned[y_col], errors='coerce')
        data_cleaned = data_cleaned.dropna(subset=[y_col])

        data_sorted = data_cleaned.sort_values(by=y_col, ascending=False)
        
        ax.bar(data_sorted[x_col], data_sorted[y_col], color='skyblue')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Use BytesIO instead of temporary files
        from io import BytesIO
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        plt.close(fig)  # Important: close the figure to free memory
        
        img_buffer.seek(0)
        image_bytes = img_buffer.getvalue()
        img_buffer.close()
        
        return image_bytes
    except Exception as e:
        print(f"Error generating bar chart: {e}")
        traceback.print_exc()
        return None

def generate_pdf_report(data, district_name):
    """Generates a PDF summary report from a DataFrame."""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, f"Groundwater Report for {district_name.title()} District", 0, 1, 'C')
        pdf.ln(10)

        for index, row in data.iterrows():
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, str(row.get("ASSESSMENT UNIT", "N/A")), 0, 1)
            pdf.set_font("Helvetica", "", 11)
            for col, value in row.items():
                if pd.notna(value) and "Unnamed" not in str(col):
                    pdf.multi_cell(0, 7, f"  - {col}: {value}")
            pdf.ln(5)
            
        # Use BytesIO for PDF as well to avoid file issues
        from io import BytesIO
        pdf_buffer = BytesIO()
        pdf_output = pdf.output(dest='S').encode('latin-1')
        pdf_buffer.write(pdf_output)
        pdf_buffer.seek(0)
        pdf_bytes = pdf_buffer.getvalue()
        pdf_buffer.close()
        
        return pdf_bytes
    except Exception as e:
        print(f"Error generating PDF report: {e}")
        traceback.print_exc()
        return None


# --- 5. WHATSAPP SERVER LOGIC & LOCATION PROCESSING ---
app = Flask(__name__)
hydra_router = None
memory_manager = None
user_states = {}

def send_whatsapp_media(to_number, file_bytes, filename, caption, mime_type):
    """Uploads a file (image or PDF) and sends it to the user."""
    upload_url = f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_NUMBER_ID}/media"
    headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"}
    files = {'file': (filename, file_bytes, mime_type), 'messaging_product': (None, 'whatsapp')}
    
    try:
        upload_response = requests.post(upload_url, headers=headers, files=files)
        upload_response.raise_for_status()
        media_id = upload_response.json().get("id")
        if not media_id: raise Exception("Failed to get media ID.")
    except requests.exceptions.RequestException as e:
        print(f"Error uploading file: {e}\nResponse: {upload_response.text}")
        send_whatsapp_message(to_number, "Sorry, I couldn't prepare the file for sending.")
        return

    media_type = "image" if "image" in mime_type else "document"
    send_url = f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp", "to": to_number, "type": media_type,
        media_type: {"id": media_id, "caption": caption}
    }
    if media_type == "document":
        payload[media_type]["filename"] = filename

    try:
        send_response = requests.post(send_url, headers=headers, json=payload)
        send_response.raise_for_status()
        print(f"File sent to {to_number} successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error sending file message: {e}\nResponse: {send_response.text}")

def send_whatsapp_message(to_number, message):
    url = f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}", "Content-Type": "application/json"}
    data = {"messaging_product": "whatsapp", "to": to_number, "type": "text", "text": {"body": message}}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        print(f"Message sent to {to_number}: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending message: {e}")

def route(self, user_id, question):
    chat_history = self.memory_manager.get_history(user_id)
    
    question_lower = question.lower()
    
    matched_taluk = next((taluk for taluk in TALUK_DISTRICT_MAP.keys() if taluk in question_lower), None)
    matched_district = next((dist for dist in DISTRICT_JSON_MAP.keys() if dist in question_lower), None)

    # Convert chat history to the format expected by LangChain
    from langchain_core.messages import HumanMessage, AIMessage
    formatted_history = []
    for msg in chat_history:
        if hasattr(msg, 'type'):
            if msg.type == 'human':
                formatted_history.append(HumanMessage(content=msg.content))
            elif msg.type == 'ai':
                formatted_history.append(AIMessage(content=msg.content))
        elif hasattr(msg, 'get_type'):
            if msg.get_type() == 'human':
                formatted_history.append(HumanMessage(content=msg.content))
            elif msg.get_type() == 'ai':
                formatted_history.append(AIMessage(content=msg.content))
        else:
            # Fallback: check content structure
            if hasattr(msg, 'content'):
                if hasattr(msg, 'role') and msg.role == 'user':
                    formatted_history.append(HumanMessage(content=msg.content))
                elif hasattr(msg, 'role') and msg.role == 'assistant':
                    formatted_history.append(AIMessage(content=msg.content))
                else:
                    # Default to treating as AI message if we can't determine
                    formatted_history.append(AIMessage(content=str(msg)))
    
    classification = self.router_chain.invoke({"question": question}).content.strip().lower()
    print(f"--- Routing classification: '{classification}' ---")

    retriever_to_use = None
    source_name = "General AI Knowledge"

    if "visualization" in classification:
        return {"source_type": "visualization", "result": {"text": question}}
    
    elif matched_taluk:
        parent_district = TALUK_DISTRICT_MAP[matched_taluk]
        print(f"--- Routing to: '{matched_taluk}' Taluk (via '{parent_district}' RAG) ---")
        retriever_to_use = self.retrievers.get(parent_district)
        source_name = f"JSON ({parent_district.title()})"
    elif matched_district:
        print(f"--- Routing to: '{matched_district}' District RAG ---")
        retriever_to_use = self.retrievers.get(matched_district)
        source_name = f"JSON ({matched_district.title()})"
    else:
        classification = self.router_chain.invoke({"question": question}).content.strip().lower()
        print(f"--- Routing classification: '{classification}' (LLM based) ---")
        if "groundwater_data" in classification:
            print("--- Routing to: General PDF RAG ---")
            retriever_to_use = self.retrievers.get("pdf_general")
            source_name = "PDF Documents"
        else:
            result = self.general_chain.invoke({"chat_history": formatted_history, "input": question})
            self.memory_manager.save_context(user_id, {"input": question}, {"output": result.content})
            return {"source_type": "gemini", "result": {"text": result.content}}

    if retriever_to_use:
        rag_chain = get_rag_chain(retriever_to_use)
        result = rag_chain.invoke({"chat_history": formatted_history, "input": question})
        self.memory_manager.save_context(user_id, {"input": question}, {"output": result["answer"]})
        result["source_name"] = source_name
        return {"source_type": "rag", "result": result}
    else:
        return {"source_type": "gemini", "result": {"text": "I can answer questions about specific locations. Please be more specific."}}

# Update the handle_visualization_request function
def handle_visualization_request(from_number, query):
    """Handles requests to generate a graph or PDF report."""
    global groundwater_data
    
    query_lower = query.lower()
    
    # First try to use the structured JSON data for visualization
    if groundwater_data and 'chart_data' in groundwater_data:
        chart_data = groundwater_data['chart_data']
        
        image_bytes = None
        caption = "Here is the chart you requested."

        # Check for district-specific requests in the structured data
        district_in_query = next((dist for dist in DISTRICT_JSON_MAP.keys() if dist in query_lower), None)
        
        # Check for specific taluk requests
        taluk_in_query = next((taluk for taluk in TALUK_DISTRICT_MAP.keys() if taluk in query_lower), None)
        
        if taluk_in_query:
            # Filter data for the specific taluk if requested
            if 'extraction_rates' in chart_data:
                taluk_data = [item for item in chart_data['extraction_rates'] 
                            if taluk_in_query in item.get('location', '').lower()]
                if taluk_data:
                    # Create a temporary chart data structure for just this taluk
                    temp_chart_data = {'extraction_rates': taluk_data}
                    image_bytes = generate_extraction_rate_chart(temp_chart_data)
                    caption = f"This chart shows groundwater extraction rates for {taluk_in_query.title()} taluk."
                else:
                    # If no specific taluk data, fall back to the parent district
                    parent_district = TALUK_DISTRICT_MAP[taluk_in_query]
                    district_data = [item for item in chart_data['extraction_rates'] 
                                   if parent_district in item.get('location', '').lower()]
                    if district_data:
                        temp_chart_data = {'extraction_rates': district_data}
                        image_bytes = generate_extraction_rate_chart(temp_chart_data)
                        caption = f"This chart shows groundwater extraction rates for {parent_district.title()} district (which includes {taluk_in_query.title()})."
        
        elif district_in_query:
            # Filter data for the specific district if requested
            if 'extraction_rates' in chart_data:
                district_data = [item for item in chart_data['extraction_rates'] 
                               if district_in_query in item.get('location', '').lower()]
                if district_data:
                    # Create a temporary chart data structure for just this district
                    temp_chart_data = {'extraction_rates': district_data}
                    image_bytes = generate_extraction_rate_chart(temp_chart_data)
                    caption = f"This chart shows groundwater extraction rates for {district_in_query.title()} district."
        
        # General visualization requests
        elif "extraction rate" in query_lower or "extraction" in query_lower:
            image_bytes = generate_extraction_rate_chart(chart_data)
            caption = "This chart shows groundwater extraction rates across all locations."
        elif "sustainability" in query_lower:
            image_bytes = generate_sustainability_chart(chart_data)
            caption = "Sustainability metrics across locations."
        
        if image_bytes:
            send_whatsapp_media(from_number, image_bytes, "chart.png", caption, "image/png")
            return
        else:
            # If no visualization was generated, provide information about the location
            location_to_query = taluk_in_query or district_in_query
            if location_to_query:
                send_whatsapp_message(from_number, f"Looking up information about {location_to_query.title()}...")
                router_output = hydra_router.route(from_number, f"What is the groundwater data for {location_to_query}?")
                
                if router_output["source_type"] == "rag":
                    answer = router_output["result"].get('answer', 'No specific data found.')
                    send_whatsapp_message(from_number, f"Here's what I found about {location_to_query.title()}:\n\n{answer}")
                else:
                    send_whatsapp_message(from_number, f"I don't have specific visualization data for {location_to_query.title()}, but here's what I know:\n\n{router_output['result']['text']}")
                return
    
    # Fall back to RAG-based information if structured data not available
    try:
        # Check for both taluks and districts
        taluk_in_query = next((taluk for taluk in TALUK_DISTRICT_MAP.keys() if taluk in query_lower), None)
        district_in_query = next((dist for dist in DISTRICT_JSON_MAP.keys() if dist in query_lower), None)
        
        location_to_query = taluk_in_query or district_in_query
        
        if not location_to_query:
            send_whatsapp_message(from_number, "Please specify a location (e.g., 'create a graph for Erode' or 'show data for Bhavani') to generate a visual.")
            return
            
        # Use the RAG system to provide information about this location
        send_whatsapp_message(from_number, f"Looking up groundwater data for {location_to_query.title()}...")
        
        router_output = hydra_router.route(from_number, f"What is the groundwater data for {location_to_query}?")
        
        if router_output["source_type"] == "rag":
            answer = router_output["result"].get('answer', 'No specific data found.')
            source_info = f"\n\n*Source:* {router_output['result'].get('source_name', 'Local Documents')}"
            send_whatsapp_message(from_number, f"Here's what I found about {location_to_query.title()}:\n\n{answer}{source_info}")
        else:
            send_whatsapp_message(from_number, f"I don't have specific visualization data for {location_to_query.title()}, but here's what I know:\n\n{router_output['result']['text']}")

    except Exception as e:
        print(f"Error during visualization request: {e}")
        traceback.print_exc()
        send_whatsapp_message(from_number, "I couldn't create a visualization, but I can answer questions about groundwater data. Try asking me something specific.")

def get_location_details(lat, lon):
    """
    Converts latitude and longitude to find the taluk and district using Nominatim.
    Returns a dictionary with 'taluk' and 'district'.
    """
    url = f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat}&lon={lon}"
    headers = {'User-Agent': 'HydraAI/1.0'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        address = data.get("address", {})
        
        taluk = address.get("suburb") or address.get("town") or address.get("village")
        district = address.get("state_district") or address.get("county")

        return {
            "taluk": taluk.lower() if taluk else None,
            "district": district.replace(" District", "").lower() if district else None
        }
    except requests.exceptions.RequestException as e:
        print(f"Could not fetch address: {e}")
        return {"taluk": None, "district": None}

def get_media_url(media_id):
    url = f"https://graph.facebook.com/v19.0/{media_id}/"
    headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"}
    try:
        res = requests.get(url, headers=headers); res.raise_for_status(); return res.json().get("url")
    except Exception: return None

def download_media_file(media_url):
    headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"}
    try:
        res = requests.get(media_url, headers=headers); res.raise_for_status(); return res.content
    except Exception: return None

def transcribe_audio(audio_bytes):
    try:
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
            f.write(audio_bytes); temp_file_path = f.name
        result = whisper_model.transcribe(whisper.load_audio(temp_file_path))
        return result.get("text", "")
    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path): os.remove(temp_file_path)

@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    if request.method == "GET":
        if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.verify_token") == WHATSAPP_VERIFY_TOKEN:
            return request.args.get("hub.challenge"), 200
        return "Verification token mismatch", 403

    elif request.method == "POST":
        data = request.get_json()
        print(f"\n--- INCOMING ---\n{json.dumps(data, indent=2)}\n----------------\n")
        try:
            for entry in data.get("entry", []):
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    if "messages" in value:
                        for msg in value.get("messages", []):
                            from_number = msg.get("from")
                            msg_type = msg.get("type")
                            if not from_number: continue
                            
                            if msg_type == "text":
                                if msg_body := msg.get("text", {}).get("body"):
                                    process_message(from_number, msg_body)
                            
                            elif msg_type == "audio":
                                if media_id := msg.get("audio", {}).get("id"):
                                    send_whatsapp_message(from_number, "Listening... 🎤")
                                    if url := get_media_url(media_id):
                                        if audio := download_media_file(url):
                                            if text := transcribe_audio(audio):
                                                process_message(from_number, text)
                            
                            elif msg_type == "location":
                                loc = msg.get("location", {})
                                lat, lon = loc.get("latitude"), loc.get("longitude")
                                if lat and lon:
                                    send_whatsapp_message(from_number, "Got your location! Finding data for your area...")
                                    location_details = get_location_details(lat, lon)
                                    taluk = location_details.get("taluk")
                                    district = location_details.get("district")
                                    
                                    if taluk and district:
                                        query = f"What is the groundwater report for {taluk} in {district} district?"
                                        process_message(from_number, query)
                                    elif district:
                                        query = f"What is the report for {district} district?"
                                        process_message(from_number, query)
                                    else:
                                        send_whatsapp_message(from_number, "Sorry, I couldn't determine the location from your coordinates.")

        except Exception as e:
            print(f"Webhook error: {e}"); traceback.print_exc()
        return Response(status=200)
    return "Method not allowed", 405

def process_message(from_number, msg_body):
    global hydra_router, user_states
    
    if from_number not in user_states:
        user_states[from_number] = {"state": "awaiting_start"}
        send_whatsapp_message(from_number, "Welcome to Hydra AI! Type 'start' to begin.")
        return

    current_state = user_states[from_number].get("state")

    if current_state == "awaiting_start":
        if msg_body.strip().lower() == "start":
            user_states[from_number]["state"] = "active"
            send_whatsapp_message(from_number, "AI activated! How can I help? You can also share your location for a local report or request visualizations.")
        else:
            send_whatsapp_message(from_number, "Please type 'start' to begin.")
    
    elif current_state == "active":
        if hydra_router:
            router_output = hydra_router.route(from_number, msg_body)
            source_type = router_output["source_type"]
            result = router_output["result"]

            if source_type == "visualization":
                handle_visualization_request(from_number, result.get('text'))
                return
            
            elif source_type == "rag":
                answer = result.get('answer', 'Sorry, I had an issue.')
                source_info = f"\n\n*Source:* {result.get('source_name', 'Local Documents')}"
                final_message = answer + source_info
            else: # Gemini
                final_message = result.get('text', 'Sorry, I had an issue.') + "\n\n*Source:* AI Generated"
                
            send_whatsapp_message(from_number, final_message)

class PerUserMemory:
    def __init__(self):
        self.conversations = {}
    
    def get_memory(self, user_id):
        if user_id not in self.conversations:
            # Use the newer approach to avoid deprecation warnings
            from langchain.memory import ChatMessageHistory
            message_history = ChatMessageHistory()
            self.conversations[user_id] = {
                'history': message_history,
                'memory': ConversationBufferMemory(
                    chat_memory=message_history,
                    return_messages=True,
                    memory_key="chat_history"
                )
            }
        return self.conversations[user_id]
    
    def get_history(self, user_id):
        return self.get_memory(user_id)['history'].messages
    
    def save_context(self, user_id, inputs, outputs):
        memory_obj = self.get_memory(user_id)['memory']
        memory_obj.save_context(inputs, outputs)

# --- 6. MAIN APPLICATION EXECUTION ---
if __name__ == "__main__":
    if not all([GEMINI_API_KEY, WHATSAPP_ACCESS_TOKEN, WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_VERIFY_TOKEN]):
        print("Error: Missing required environment variables. Check .env file.")
    else:
        print("Initializing Hydra AI...")
        
        # Load the structured JSON data for visualization
        try:
            if os.path.exists(SOURCE_JSON_PATH):
                with open(SOURCE_JSON_PATH, 'r') as f:
                    groundwater_data = json.load(f)
                print("Source JSON data loaded successfully for visualization.")
            else:
                print(f"Visualization JSON file not found at {SOURCE_JSON_PATH}. Visualization features will be limited.")
        except Exception as e:
            print(f"Error loading visualization JSON file: {e}")
            groundwater_data = None
        
        all_retrievers = create_or_load_all_vector_dbs()

        if all_retrievers:
            memory_manager = PerUserMemory()
            general_chain = get_general_chain()
            hydra_router = SemanticRouter(general_chain, memory_manager, all_retrievers)

            print("\n" + "="*50)
            print("💧 Hydra WhatsApp Server is running.")
            print("="*50)
            print("Listening for messages on port 5000...")
            app.run(host="0.0.0.0", port=5000, debug=False)
        else:
            print("Failed to initialize vector DBs. Server not started.")