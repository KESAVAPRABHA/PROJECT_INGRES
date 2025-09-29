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
        router_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=GEMINI_API_KEY)
                # Updated prompt with 'visualization' category
        routing_prompt = ChatPromptTemplate.from_template(
    """You are an expert at classifying user questions for a specialized Indian groundwater chatbot.
    Classify the question into one of three categories: 'visualization', 'groundwater_data', or 'general_knowledge'.

    **'visualization' Category:**
    Choose this if the user is asking to "create a graph", "generate a report", "show a chart", "plot the data", "make a PDF", 
    "sustainability metrics", "draft composition", "recharge sources", "compare", or "comparison" for specific locations.
    - Examples: "create a graph for Erode", "generate a pdf report for Namakkal district", 
      "show me sustainability metrics", "compare Perundurai and Bhavani", "make a comparison chart"

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
def generate_comparison_chart(chart_data, taluk_names, metric="extraction_rate"):
    """Generates a comparison chart for multiple taluks."""
    try:
        # Filter data for the requested taluks
        filtered_data = []
        for item in chart_data['extraction_rates']:
            if any(taluk.lower() in item['location'].lower() for taluk in taluk_names):
                filtered_data.append(item)
        
        if not filtered_data:
            return None
            
        df = pd.DataFrame(filtered_data)
        df = df.sort_values(by=metric, ascending=False)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.bar(df['location'], df[metric], 
                     color=[item['color'] for item in df.to_dict('records')])
        ax.set_title(f'Comparison of {metric.replace("_", " ").title()} between {", ".join(taluk_names)}', fontsize=16)
        ax.set_ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        plt.close(fig)
        
        img_buffer.seek(0)
        image_bytes = img_buffer.getvalue()
        img_buffer.close()
        
        return image_bytes
    except Exception as e:
        print(f"Error generating comparison chart: {e}")
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
# def handle_visualization_request(from_number, query):
#     """Handles requests to generate a graph or PDF report."""
#     global groundwater_data
    
#     query_lower = query.lower()
    
#     # First try to use the structured JSON data for visualization
#     if groundwater_data and 'chart_data' in groundwater_data:
#         chart_data = groundwater_data['chart_data']
        
#         image_bytes = None
#         caption = "Here is the chart you requested."

#         # Check for district-specific requests in the structured data
#         district_in_query = next((dist for dist in DISTRICT_JSON_MAP.keys() if dist in query_lower), None)
        
#         # Check for specific taluk requests
#         taluk_in_query = next((taluk for taluk in TALUK_DISTRICT_MAP.keys() if taluk in query_lower), None)
        
#         if taluk_in_query:
#             # Filter data for the specific taluk if requested
#             if 'extraction_rates' in chart_data:
#                 taluk_data = [item for item in chart_data['extraction_rates'] 
#                             if taluk_in_query in item.get('location', '').lower()]
#                 if taluk_data:
#                     # Create a temporary chart data structure for just this taluk
#                     temp_chart_data = {'extraction_rates': taluk_data}
#                     image_bytes = generate_extraction_rate_chart(temp_chart_data)
#                     caption = f"This chart shows groundwater extraction rates for {taluk_in_query.title()} taluk."
#                 else:
#                     # If no specific taluk data, fall back to the parent district
#                     parent_district = TALUK_DISTRICT_MAP[taluk_in_query]
#                     district_data = [item for item in chart_data['extraction_rates'] 
#                                    if parent_district in item.get('location', '').lower()]
#                     if district_data:
#                         temp_chart_data = {'extraction_rates': district_data}
#                         image_bytes = generate_extraction_rate_chart(temp_chart_data)
#                         caption = f"This chart shows groundwater extraction rates for {parent_district.title()} district (which includes {taluk_in_query.title()})."
        
#         elif district_in_query:
#             # Filter data for the specific district if requested
#             if 'extraction_rates' in chart_data:
#                 district_data = [item for item in chart_data['extraction_rates'] 
#                                if district_in_query in item.get('location', '').lower()]
#                 if district_data:
#                     # Create a temporary chart data structure for just this district
#                     temp_chart_data = {'extraction_rates': district_data}
#                     image_bytes = generate_extraction_rate_chart(temp_chart_data)
#                     caption = f"This chart shows groundwater extraction rates for {district_in_query.title()} district."
        
#         # General visualization requests
#         elif "extraction rate" in query_lower or "extraction" in query_lower:
#             image_bytes = generate_extraction_rate_chart(chart_data)
#             caption = "This chart shows groundwater extraction rates across all locations."
#         elif "sustainability" in query_lower:
#             image_bytes = generate_sustainability_chart(chart_data)
#             caption = "Sustainability metrics across locations."
        
#         if image_bytes:
#             send_whatsapp_media(from_number, image_bytes, "chart.png", caption, "image/png")
#             return
#         else:
#             # If no visualization was generated, provide information about the location
#             location_to_query = taluk_in_query or district_in_query
#             if location_to_query:
#                 send_whatsapp_message(from_number, f"Looking up information about {location_to_query.title()}...")
#                 router_output = hydra_router.route(from_number, f"What is the groundwater data for {location_to_query}?")
                
#                 if router_output["source_type"] == "rag":
#                     answer = router_output["result"].get('answer', 'No specific data found.')
#                     send_whatsapp_message(from_number, f"Here's what I found about {location_to_query.title()}:\n\n{answer}")
#                 else:
#                     send_whatsapp_message(from_number, f"I don't have specific visualization data for {location_to_query.title()}, but here's what I know:\n\n{router_output['result']['text']}")
#                 return
    
#     # Fall back to RAG-based information if structured data not available
#     try:
#         # Check for both taluks and districts
#         taluk_in_query = next((taluk for taluk in TALUK_DISTRICT_MAP.keys() if taluk in query_lower), None)
#         district_in_query = next((dist for dist in DISTRICT_JSON_MAP.keys() if dist in query_lower), None)
        
#         location_to_query = taluk_in_query or district_in_query
        
#         if not location_to_query:
#             send_whatsapp_message(from_number, "Please specify a location (e.g., 'create a graph for Erode' or 'show data for Bhavani') to generate a visual.")
#             return
            
#         # Use the RAG system to provide information about this location
#         send_whatsapp_message(from_number, f"Looking up groundwater data for {location_to_query.title()}...")
        
#         router_output = hydra_router.route(from_number, f"What is the groundwater data for {location_to_query}?")
        
#         if router_output["source_type"] == "rag":
#             answer = router_output["result"].get('answer', 'No specific data found.')
#             source_info = f"\n\n*Source:* {router_output['result'].get('source_name', 'Local Documents')}"
#             send_whatsapp_message(from_number, f"Here's what I found about {location_to_query.title()}:\n\n{answer}{source_info}")
#         else:
#             send_whatsapp_message(from_number, f"I don't have specific visualization data for {location_to_query.title()}, but here's what I know:\n\n{router_output['result']['text']}")

#     except Exception as e:
#         print(f"Error during visualization request: {e}")
#         traceback.print_exc()
#         send_whatsapp_message(from_number, "I couldn't create a visualization, but I can answer questions about groundwater data. Try asking me something specific.")


# def handle_visualization_request(from_number, query):
#     """Handles requests to generate a graph or PDF report."""
#     global groundwater_data
    
#     query_lower = query.lower()
    
#     # First try to use the structured JSON data for visualization
#     if groundwater_data and 'chart_data' in groundwater_data:
#         chart_data = groundwater_data['chart_data']
        
#         image_bytes = None
#         caption = "Here is the chart you requested."

#         # Check for comparison requests (multiple taluks)
#         taluk_names = []
#         for taluk in TALUK_DISTRICT_MAP.keys():
#             if taluk in query_lower:
#                 taluk_names.append(taluk)
        
#         # If we found multiple taluks, generate a comparison chart
#         if len(taluk_names) >= 2:
#             image_bytes = generate_comparison_chart(chart_data, taluk_names)
#             caption = f"Comparison of groundwater extraction rates between {', '.join([t.title() for t in taluk_names])}"
        
#         # Check for district-specific requests in the structured data
#         district_in_query = next((dist for dist in DISTRICT_JSON_MAP.keys() if dist in query_lower), None)
        
#         # Check for specific taluk requests (single taluk)
#         taluk_in_query = next((taluk for taluk in TALUK_DISTRICT_MAP.keys() if taluk in query_lower), None)
        
#         if taluk_in_query and not image_bytes:  # Only if we haven't already generated a comparison
#             # Filter data for the specific taluk if requested
#             if 'extraction_rates' in chart_data:
#                 taluk_data = [item for item in chart_data['extraction_rates'] 
#                             if taluk_in_query in item.get('location', '').lower()]
#                 if taluk_data:
#                     # Create a temporary chart data structure for just this taluk
#                     temp_chart_data = {'extraction_rates': taluk_data}
#                     image_bytes = generate_extraction_rate_chart(temp_chart_data)
#                     caption = f"This chart shows groundwater extraction rates for {taluk_in_query.title()} taluk."
#                 else:
#                     # If no specific taluk data, fall back to the parent district
#                     parent_district = TALUK_DISTRICT_MAP[taluk_in_query]
#                     district_data = [item for item in chart_data['extraction_rates'] 
#                                    if parent_district in item.get('location', '').lower()]
#                     if district_data:
#                         temp_chart_data = {'extraction_rates': district_data}
#                         image_bytes = generate_extraction_rate_chart(temp_chart_data)
#                         caption = f"This chart shows groundwater extraction rates for {parent_district.title()} district (which includes {taluk_in_query.title()})."
        
#         elif district_in_query and not image_bytes:  # Only if we haven't already generated a comparison
#             # Filter data for the specific district if requested
#             if 'extraction_rates' in chart_data:
#                 district_data = [item for item in chart_data['extraction_rates'] 
#                                if district_in_query in item.get('location', '').lower()]
#                 if district_data:
#                     # Create a temporary chart data structure for just this district
#                     temp_chart_data = {'extraction_rates': district_data}
#                     image_bytes = generate_extraction_rate_chart(temp_chart_data)
#                     caption = f"This chart shows groundwater extraction rates for {district_in_query.title()} district."
        
#         # General visualization requests
#         elif ("extraction rate" in query_lower or "extraction" in query_lower) and not image_bytes:
#             image_bytes = generate_extraction_rate_chart(chart_data)
#             caption = "This chart shows groundwater extraction rates across all locations."
#         elif "sustainability" in query_lower and not image_bytes:
#             image_bytes = generate_sustainability_chart(chart_data)
#             caption = "Sustainability metrics across locations."
        
#         if image_bytes:
#             send_whatsapp_media(from_number, image_bytes, "chart.png", caption, "image/png")
#             return
#         else:
#             # If no visualization was generated, provide information about the location
#             location_to_query = taluk_in_query or district_in_query
#             if location_to_query:
#                 send_whatsapp_message(from_number, f"Looking up information about {location_to_query.title()}...")
#                 router_output = hydra_router.route(from_number, f"What is the groundwater data for {location_to_query}?")
                
#                 if router_output["source_type"] == "rag":
#                     answer = router_output["result"].get('answer', 'No specific data found.')
#                     send_whatsapp_message(from_number, f"Here's what I found about {location_to_query.title()}:\n\n{answer}")
#                 else:
#                     send_whatsapp_message(from_number, f"I don't have specific visualization data for {location_to_query.title()}, but here's what I know:\n\n{router_output['result']['text']}")
#                 return
    
#     # Fall back to RAG-based information if structured data not available
#     try:
#         # Check for both taluks and districts
#         taluk_names = []
#         for taluk in TALUK_DISTRICT_MAP.keys():
#             if taluk in query_lower:
#                 taluk_names.append(taluk)
        
#         district_in_query = next((dist for dist in DISTRICT_JSON_MAP.keys() if dist in query_lower), None)
        
#         location_to_query = taluk_names[0] if taluk_names else district_in_query
        
#         if not location_to_query:
#             send_whatsapp_message(from_number, "Please specify a location (e.g., 'create a graph for Erode' or 'show data for Bhavani and Perundurai') to generate a visual.")
#             return
            
#         # Use the RAG system to provide information about this location
#         if len(taluk_names) >= 2:
#             # For multiple taluks, provide information about each
#             for taluk in taluk_names:
#                 send_whatsapp_message(from_number, f"Looking up groundwater data for {taluk.title()}...")
#                 router_output = hydra_router.route(from_number, f"What is the groundwater data for {taluk}?")
                
#                 if router_output["source_type"] == "rag":
#                     answer = router_output["result"].get('answer', 'No specific data found.')
#                     source_info = f"\n\n*Source:* {router_output['result'].get('source_name', 'Local Documents')}"
#                     send_whatsapp_message(from_number, f"Here's what I found about {taluk.title()}:\n\n{answer}{source_info}")
#                 else:
#                     send_whatsapp_message(from_number, f"I don't have specific visualization data for {taluk.title()}, but here's what I know:\n\n{router_output['result']['text']}")
#         else:
#             # For single location
#             send_whatsapp_message(from_number, f"Looking up groundwater data for {location_to_query.title()}...")
#             router_output = hydra_router.route(from_number, f"What is the groundwater data for {location_to_query}?")
            
#             if router_output["source_type"] == "rag":
#                 answer = router_output["result"].get('answer', 'No specific data found.')
#                 source_info = f"\n\n*Source:* {router_output['result'].get('source_name', 'Local Documents')}"
#                 send_whatsapp_message(from_number, f"Here's what I found about {location_to_query.title()}:\n\n{answer}{source_info}")
#             else:
#                 send_whatsapp_message(from_number, f"I don't have specific visualization data for {location_to_query.title()}, but here's what I know:\n\n{router_output['result']['text']}")

#     except Exception as e:
#         print(f"Error during visualization request: {e}")
# def handle_visualization_request(from_number, query):
#     """Handles requests to generate comprehensive analysis charts for single locations."""
#     global groundwater_data
    
#     query_lower = query.lower()
    
#     # First try to use the structured JSON data for visualization
#     if groundwater_data and 'flattened_data' in groundwater_data:
        
#         image_bytes = None
#         caption = "Here is the analysis you requested."

#         # Find the requested location (taluk or district)
#         location_found = None
        
#         # Check for specific taluk requests first
#         for taluk in TALUK_DISTRICT_MAP.keys():
#             if taluk in query_lower:
#                 location_found = taluk
#                 break
        
#         # If no taluk found, check for district requests
#         if not location_found:
#             for district in DISTRICT_JSON_MAP.keys():
#                 if district in query_lower:
#                     location_found = district
#                     break
        
#         if location_found:
#             # Determine which type of analysis to generate
#             if any(keyword in query_lower for keyword in ['comprehensive', 'detailed', 'full', 'complete']):
#                 # Generate comprehensive 6-panel analysis
#                 image_bytes = generate_single_location_analysis_chart(groundwater_data, location_found)
#                 caption = f"Comprehensive groundwater analysis for {location_found.title()} showing all key metrics including extraction status, water balance, draft composition, sustainability metrics, firka distribution, and recharge vs draft comparison."
            
#             elif any(keyword in query_lower for keyword in ['key', 'important', 'critical', 'main', 'summary']):
#                 # Generate focused 5-metric summary
#                 image_bytes = generate_key_metrics_summary_chart(groundwater_data, location_found)
#                 caption = f"Key groundwater metrics for {location_found.title()} showing the 5 most important indicators: extraction rate, safety margin, sustainability index, net balance, and firka risk score."
            
#             else:
#                 # Default to key metrics for single location requests
#                 image_bytes = generate_key_metrics_summary_chart(groundwater_data, location_found)
#                 caption = f"Key groundwater metrics analysis for {location_found.title()} showing the 5 most critical indicators for water management decision-making."
        
#         # If specific location visualization was generated, send it
#         if image_bytes:
#             send_whatsapp_media(from_number, image_bytes, "groundwater_analysis.png", caption, "image/png")
            
#             # Send additional textual summary
#             send_location_analysis_summary(from_number, groundwater_data, location_found)
#             return
        
#         # Fallback to original chart_data if available
#         elif 'chart_data' in groundwater_data:
#             chart_data = groundwater_data['chart_data']
            
#             # General visualization requests using existing functions
#             if ("extraction rate" in query_lower or "extraction" in query_lower):
#                 image_bytes = generate_extraction_rate_chart(chart_data)
#                 caption = "Groundwater extraction rates across all locations."
#             elif "sustainability" in query_lower:
#                 image_bytes = generate_sustainability_chart(chart_data)
#                 caption = "Sustainability metrics across all locations."
            
#             if image_bytes:
#                 send_whatsapp_media(from_number, image_bytes, "chart.png", caption, "image/png")
#                 return
    
#     # Fall back to RAG-based information if no structured data or visualization generated
#     try:
#         # Check for location in query
#         location_to_query = None
#         for taluk in TALUK_DISTRICT_MAP.keys():
#             if taluk in query_lower:
#                 location_to_query = taluk
#                 break
        
#         if not location_to_query:
#             for district in DISTRICT_JSON_MAP.keys():
#                 if district in query_lower:
#                     location_to_query = district
#                     break
        
#         if location_to_query:
#             # Use RAG system to provide information about this location
#             send_whatsapp_message(from_number, f"Looking up groundwater data for {location_to_query.title()}...")
#             router_output = hydra_router.route(from_number, f"What is the groundwater data for {location_to_query}?")
            
#             if router_output["source_type"] == "rag":
#                 answer = router_output["result"].get('answer', 'No specific data found.')
#                 source_info = f"\n\n*Source:* {router_output['result'].get('source_name', 'Local Documents')}"
#                 send_whatsapp_message(from_number, f"Here's what I found about {location_to_query.title()}:\n\n{answer}{source_info}")
#             else:
#                 send_whatsapp_message(from_number, f"I don't have specific visualization data for {location_to_query.title()}, but here's what I know:\n\n{router_output['result']['text']}")
#         else:
#             send_whatsapp_message(from_number, "Please specify a location (e.g., 'show analysis for Erode', 'key metrics for Bhavani', or 'comprehensive report for Perundurai') to generate a detailed analysis.")

#     except Exception as e:
#         print(f"Error during visualization request: {e}")
#         traceback.print_exc()
#         send_whatsapp_message(from_number, "I couldn't create a visualization, but I can answer questions about groundwater data. Try asking me something specific.")


# def send_location_analysis_summary(from_number, groundwater_data, location_name):
#     """Sends a textual summary of the key findings for a location."""
#     try:
#         flattened_data = groundwater_data.get('flattened_data', [])
        
#         location_data = None
#         for item in flattened_data:
#             if location_name.lower() in item['location'].lower():
#                 location_data = item
#                 break
        
#         if not location_data:
#             return
        
#         summary = f"📊 **Analysis Summary: {location_data['location'].title()}**\n\n"
        
#         # Overall Status
#         status_emoji = {
#             'safe': '✅',
#             'semi_critical': '⚠️', 
#             'critical': '🔶',
#             'over_exploited': '🚨'
#         }
        
#         emoji = status_emoji.get(location_data['overall_status'], '📍')
#         summary += f"{emoji} **Overall Status:** {location_data['overall_status'].replace('_', ' ').title()}\n\n"
        
#         # Key Metrics
#         summary += f"🔹 **Key Metrics:**\n"
#         summary += f"   • Extraction Rate: {location_data['extraction_rate_total']:.1f}%\n"
#         summary += f"   • Safety Margin: {location_data['safety_margin']:.1f}%\n"
#         summary += f"   • Sustainability Index: {location_data['sustainability_index']:.1f}\n"
#         summary += f"   • Net Water Balance: {location_data['net_balance']:,.0f} ha-m\n\n"
        
#         # Risk Assessment
#         total_firkas = (location_data['over_exploited_firkas'] + 
#                        location_data['semi_critical_firkas'] + 
#                        location_data['safe_firkas'])
        
#         summary += f"🔹 **Risk Assessment:**\n"
#         summary += f"   • Total Firkas: {total_firkas}\n"
#         summary += f"   • Over-exploited: {location_data['over_exploited_firkas']}\n"
#         summary += f"   • Semi-critical: {location_data['semi_critical_firkas']}\n"
#         summary += f"   • Safe: {location_data['safe_firkas']}\n\n"
        
#         # Recommendations
#         summary += f"💡 **Key Insights:**\n"
        
#         if location_data['extraction_rate_total'] > 100:
#             summary += f"   • ⚠️ High extraction rate requires immediate intervention\n"
#         elif location_data['extraction_rate_total'] > 90:
#             summary += f"   • ⚠️ Approaching critical extraction levels\n"
#         else:
#             summary += f"   • ✅ Extraction rate within manageable limits\n"
        
#         if location_data['net_balance'] < 0:
#             summary += f"   • 🚨 Water deficit of {abs(location_data['net_balance']):,.0f} ha-m\n"
#         else:
#             summary += f"   • ✅ Water surplus of {location_data['net_balance']:,.0f} ha-m\n"
        
#         if location_data['over_exploited_firkas'] > 0:
#             summary += f"   • 🔴 {location_data['over_exploited_firkas']} firka(s) need urgent attention\n"
        
#         send_whatsapp_message(from_number, summary)
        
#     except Exception as e:
#         print(f"Error sending location analysis summary: {e}")
#         traceback.print_exc()
#         send_whatsapp_message(from_number, "I couldn't create a visualization, but I can answer questions about groundwater data. Try asking me something specific.")

def handle_visualization_request(from_number, query):
    """Handles requests to generate a graph or PDF report."""
    global groundwater_data
    
    query_lower = query.lower()
    
    # Check for PDF report requests first
    if any(keyword in query_lower for keyword in ['pdf', 'report', 'generate report', 'download']):
        # Find the requested location
        location_found = None
        
        # Check for specific taluk requests first
        for taluk in TALUK_DISTRICT_MAP.keys():
            if taluk in query_lower:
                location_found = taluk
                break
        
        # If no taluk found, check for district requests
        if not location_found:
            for district in DISTRICT_JSON_MAP.keys():
                if district in query_lower:
                    location_found = district
                    break
        
        if location_found:
            send_whatsapp_message(from_number, f"📋 Generating comprehensive PDF report for {location_found.title()}...")
            
            # Generate PDF report
            pdf_bytes = generate_taluk_pdf_report(groundwater_data, location_found)
            
            if pdf_bytes:
                filename = f"Groundwater_Report_{location_found.title()}.pdf"
                caption = f"📊 Comprehensive Groundwater Report for {location_found.title()}\n\nThis report contains detailed analysis, key metrics, and recommendations for water management."
                
                send_whatsapp_media(from_number, pdf_bytes, filename, caption, "application/pdf")
                return
            else:
                send_whatsapp_message(from_number, f"Sorry, I couldn't generate a PDF report for {location_found.title()}. The data might not be available.")
                return
    
    # Rest of the existing visualization handling code remains the same...
    # [Keep all the existing chart generation code here]
    
    # First try to use the structured JSON data for visualization
    if groundwater_data and 'flattened_data' in groundwater_data:
        
        image_bytes = None
        caption = "Here is your water analysis report."
        chart_type = 'key_metrics'  # default

        # Find the requested location (taluk or district)
        location_found = None
        
        # Check for specific taluk requests first
        for taluk in TALUK_DISTRICT_MAP.keys():
            if taluk in query_lower:
                location_found = taluk
                break
        
        # If no taluk found, check for district requests
        if not location_found:
            for district in DISTRICT_JSON_MAP.keys():
                if district in query_lower:
                    location_found = district
                    break
        
        if location_found:
            # Determine which type of analysis to generate
            if any(keyword in query_lower for keyword in ['comprehensive', 'detailed', 'full', 'complete']):
                # Generate comprehensive 6-panel analysis
                image_bytes = generate_single_location_analysis_chart(groundwater_data, location_found)
                caption = f"Complete water analysis for {location_found.title()}. This shows 6 different aspects of your area's water situation."
                chart_type = 'comprehensive'
            
            elif any(keyword in query_lower for keyword in ['key', 'important', 'critical', 'main', 'summary']):
                # Generate focused 5-metric summary
                image_bytes = generate_key_metrics_summary_chart(groundwater_data, location_found)
                caption = f"Key water facts for {location_found.title()}. These are the 5 most important things to know about water in your area."
                chart_type = 'key_metrics'
            
            else:
                # Default to key metrics for single location requests
                image_bytes = generate_key_metrics_summary_chart(groundwater_data, location_found)
                caption = f"Water situation summary for {location_found.title()}. This shows the most important water facts for your area."
                chart_type = 'key_metrics'
        
        # If specific location visualization was generated, send it
        if image_bytes:
            send_whatsapp_media(from_number, image_bytes, "water_report.png", caption, "image/png")
            
            # Send simple chart explanation
            send_simple_chart_explanation(from_number, location_found, chart_type)
            
            # Send simple textual summary
            send_location_analysis_summary(from_number, groundwater_data, location_found)
            return
        
        # Fallback to original chart_data if available
        elif 'chart_data' in groundwater_data:
            chart_data = groundwater_data['chart_data']
            
            # General visualization requests using existing functions
            if ("extraction rate" in query_lower or "extraction" in query_lower):
                image_bytes = generate_extraction_rate_chart(chart_data)
                caption = "Water usage levels across different areas. Higher bars mean more water is being used."
            elif "sustainability" in query_lower:
                image_bytes = generate_sustainability_chart(chart_data)
                caption = "Long-term water health across different areas. This shows which areas can maintain their water supply."
            
            if image_bytes:
                send_whatsapp_media(from_number, image_bytes, "water_chart.png", caption, "image/png")
                send_whatsapp_message(from_number, "This chart shows water information across multiple areas. Each point or bar represents a different location.")
                return
    
    # Fall back to RAG-based information if no structured data or visualization generated
    try:
        # Check for location in query
        location_to_query = None
        for taluk in TALUK_DISTRICT_MAP.keys():
            if taluk in query_lower:
                location_to_query = taluk
                break
        
        if not location_to_query:
            for district in DISTRICT_JSON_MAP.keys():
                if district in query_lower:
                    location_to_query = district
                    break
        
        if location_to_query:
            # Use RAG system to provide information about this location
            send_whatsapp_message(from_number, f"Looking up water information for {location_to_query.title()}...")
            router_output = hydra_router.route(from_number, f"What is the groundwater data for {location_to_query}?")
            
            if router_output["source_type"] == "rag":
                answer = router_output["result"].get('answer', 'No specific data found.')
                source_info = f"\n\n*Source:* {router_output['result'].get('source_name', 'Local Documents')}"
                send_whatsapp_message(from_number, f"Here's what I found about water in {location_to_query.title()}:\n\n{answer}{source_info}")
            else:
                send_whatsapp_message(from_number, f"I don't have specific water charts for {location_to_query.title()}, but here's what I know:\n\n{router_output['result']['text']}")
        else:
            send_whatsapp_message(from_number, "Please tell me which area you want to know about. For example: 'show water report for Erode' or 'generate PDF report for Bhavani'")

    except Exception as e:
        print(f"Error during visualization request: {e}")
        traceback.print_exc()
        send_whatsapp_message(from_number, "I'm having trouble creating your water report right now. You can still ask me questions about water in your area and I'll try to help.")

def send_location_analysis_summary(from_number, groundwater_data, location_name):
    """Sends a simple, user-friendly summary that anyone can understand."""
    try:
        flattened_data = groundwater_data.get('flattened_data', [])
        
        location_data = None
        for item in flattened_data:
            if location_name.lower() in item['location'].lower():
                location_data = item
                break
        
        if not location_data:
            return
        
        # Create simple, easy-to-understand summary
        summary = f"💧 *Water Report: {location_data['location'].title()}*\n\n"
        
        # Overall Status in simple terms
        status_messages = {
            'safe': '✅ *Water Situation:* Good - No major concerns',
            'semi_critical': '⚠️ *Water Situation:* Watch carefully - Some problems starting',
            'critical': '🔶 *Water Situation:* Serious concerns - Action needed soon',
            'over_exploited': '🚨 *Water Situation:* Very serious - Urgent action required'
        }
        
        summary += status_messages.get(location_data['overall_status'], '📍 *Water Situation:* Under review') + '\n\n'
        
        # Explain what's happening in simple terms
        extraction_rate = location_data['extraction_rate_total']
        
        summary += "*What's happening with water:*\n"
        
        # Water usage explanation
        if extraction_rate > 100:
            summary += f"• We're using {extraction_rate:.0f}% of available water (using too much!)\n"
            summary += "• This means we're taking out more water than nature puts back in\n"
        elif extraction_rate > 90:
            summary += f"• We're using {extraction_rate:.0f}% of available water (almost too much)\n"
            summary += "• We need to be very careful about water use\n"
        elif extraction_rate > 70:
            summary += f"• We're using {extraction_rate:.0f}% of available water (need to watch)\n"
            summary += "• Water use is getting high but still manageable\n"
        else:
            summary += f"• We're using {extraction_rate:.0f}% of available water (good level)\n"
            summary += "• Water use is at a safe level\n"
        
        summary += "\n"
        
        # Water balance in simple terms
        net_balance = location_data['net_balance']
        if net_balance > 0:
            summary += f"• We have extra water: {net_balance:,.0f} units more than we use\n"
        else:
            summary += f"• We have a water shortage: {abs(net_balance):,.0f} units less than we need\n"
        
        summary += "\n"
        
        # Area problems in simple terms
        total_areas = (location_data['over_exploited_firkas'] + 
                      location_data['semi_critical_firkas'] + 
                      location_data['safe_firkas'])
        
        problem_areas = location_data['over_exploited_firkas']
        warning_areas = location_data['semi_critical_firkas']
        safe_areas = location_data['safe_firkas']
        
        summary += f"*Areas in {location_data['location'].title()}:*\n"
        summary += f"• Total areas checked: {total_areas}\n"
        
        if problem_areas > 0:
            summary += f"• Areas with serious water problems: {problem_areas}\n"
        if warning_areas > 0:
            summary += f"• Areas that need watching: {warning_areas}\n"
        if safe_areas > 0:
            summary += f"• Areas with good water situation: {safe_areas}\n"
        
        summary += "\n"
        
        # Simple recommendations
        summary += "*What this means for you:*\n"
        
        if extraction_rate > 100:
            summary += "• Water situation is very serious - save water wherever possible\n"
            summary += "• Contact local authorities about water conservation\n"
        elif extraction_rate > 90:
            summary += "• Start saving water now to prevent bigger problems\n"
            summary += "• Avoid wasting water in daily activities\n"
        elif extraction_rate > 70:
            summary += "• Be mindful of water use - don't waste it\n"
            summary += "• Consider water-saving methods\n"
        else:
            summary += "• Water situation is currently good\n"
            summary += "• Continue using water responsibly\n"
        
        if problem_areas > 0:
            summary += f"• {problem_areas} area(s) need immediate attention from authorities\n"
        
        # Water usage breakdown in simple terms
        total_usage = location_data['total_draft']
        farm_usage = (location_data['agriculture_draft'] / total_usage) * 100
        home_usage = (location_data['domestic_draft'] / total_usage) * 100
        industry_usage = (location_data['industry_draft'] / total_usage) * 100
        
        summary += f"\n*Where water is being used:*\n"
        summary += f"• Farming: {farm_usage:.0f}%\n"
        summary += f"• Homes & drinking: {home_usage:.0f}%\n"
        summary += f"• Industries: {industry_usage:.0f}%\n"
        
        send_whatsapp_message(from_number, summary)
        
    except Exception as e:
        print(f"Error sending simple location summary: {e}")
        import traceback
        traceback.print_exc()

def get_simple_status_message(overall_status, extraction_rate):
    """Returns a simple explanation of the water situation."""
    if overall_status == 'safe':
        return "Water situation is good. Keep using water wisely."
    elif overall_status == 'semi_critical':
        return "Water levels are getting concerning. Time to start saving water."
    elif overall_status == 'critical':
        return "Water situation is serious. Save water and avoid waste."
    elif overall_status == 'over_exploited':
        return "Water crisis situation. Every drop counts - save water immediately."
    else:
        return "Water situation is being monitored."

def send_simple_chart_explanation(from_number, location_name, chart_type):
    """Sends a simple explanation of what the chart shows."""
    explanations = {
        'comprehensive': f"""
🔍 *Understanding Your Water Chart for {location_name.title()}*

Your chart shows 6 important things about water in your area:

1️⃣ *Circle Chart (top left):* Shows how much water we're using
   • Green = Good, Yellow = Careful, Red = Problem

2️⃣ *Bar Chart (top middle):* Compares water coming in vs going out
   • Blue = Water available, Red = Water used

3️⃣ *Pie Chart (top right):* Shows who uses the most water
   • Purple = Farms, Blue = Homes, Orange = Factories

4️⃣ *Side Bars (bottom left):* Different measurements of water health
   • Longer bars = Better situation

5️⃣ *Stacked Bar (bottom middle):* Problem areas in your region
   • Red = Serious problems, Yellow = Watch carefully, Green = All good

6️⃣ *Comparison Bars (bottom right):* Water coming in vs going out
   • Green = Water from rain/rivers, Red = Water we use
        """,
        
        'key_metrics': f"""
📊 *Understanding Your Water Summary for {location_name.title()}*

Your chart shows the 5 most important water facts:

🔹 *How much water we use* - Lower is better
🔹 *Safety cushion* - Higher is better (how much extra water we have)
🔹 *Long-term health* - Higher numbers mean water will last longer
🔹 *Water balance* - Positive numbers are good (extra water)
🔹 *Problem areas score* - Lower is better (fewer problem areas)

*Colors mean:*
• Green bars = Good situation
• Yellow bars = Be careful
• Orange bars = Getting serious
• Red bars = Need urgent action
        """
    }
    
    explanation = explanations.get(chart_type, "This chart shows important water information for your area.")
    send_whatsapp_message(from_number, explanation)

def generate_single_location_analysis_chart(groundwater_data, location_name):
    """
    Generates a comprehensive multi-feature analysis chart for a single location
    showing the 5 most important groundwater metrics for users.
    """
    try:
        flattened_data = groundwater_data.get('flattened_data', [])
        
        # Find data for the requested location
        location_data = None
        for item in flattened_data:
            if location_name.lower() in item['location'].lower():
                location_data = item
                break
        
        if not location_data:
            print(f"No data found for location: {location_name}")
            return None
        
        # Create a 2x3 subplot layout for comprehensive analysis
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))
        
        location_title = location_data['location'].title()
        
        # 1. Extraction Rate Status (Gauge Chart)
        extraction_rate = location_data['extraction_rate_total']
        
        # Create a gauge-like visualization for extraction rate
        colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545']  # green, yellow, orange, red
        ranges = [70, 90, 100, 200]  # safe, semi-critical, critical, over-exploited
        
        if extraction_rate <= 70:
            color = colors[0]
            status = "Safe"
        elif extraction_rate <= 90:
            color = colors[1]
            status = "Semi-Critical"
        elif extraction_rate <= 100:
            color = colors[2]
            status = "Critical"
        else:
            color = colors[3]
            status = "Over-Exploited"
        
        # Gauge visualization
        wedges = [70, 20, 10, 100]  # proportional segments
        wedge_colors = colors
        ax1.pie(wedges, colors=wedge_colors, startangle=180, counterclock=False)
        ax1.add_patch(plt.Circle((0, 0), 0.5, color='white'))
        ax1.text(0, 0, f'{extraction_rate:.1f}%\n{status}', ha='center', va='center', fontsize=12, fontweight='bold')
        ax1.set_title('Extraction Rate Status', fontweight='bold')
        
        # 2. Water Balance (Bar Chart)
        water_metrics = ['Total\nAvailability', 'Total\nDraft', 'Net\nBalance']
        water_values = [location_data['total_availability'], 
                       location_data['total_draft'], 
                       location_data['net_balance']]
        
        colors_water = ['#3498db', '#e74c3c', '#2ecc71' if location_data['net_balance'] > 0 else '#e74c3c']
        bars = ax2.bar(water_metrics, water_values, color=colors_water)
        ax2.set_title('Water Balance (ha-m)', fontweight='bold')
        ax2.set_ylabel('Volume (ha-m)')
        
        # Add value labels on bars
        for bar, value in zip(bars, water_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (max(water_values) * 0.01),
                    f'{value:,.0f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Draft Composition (Pie Chart)
        draft_categories = ['Agriculture', 'Domestic', 'Industry']
        draft_values = [location_data['agriculture_draft'], 
                       location_data['domestic_draft'], 
                       location_data['industry_draft']]
        draft_colors = ['#8e44ad', '#3498db', '#f39c12']
        
        ax3.pie(draft_values, labels=draft_categories, autopct='%1.1f%%', 
                colors=draft_colors, startangle=90)
        ax3.set_title('Water Draft Composition', fontweight='bold')
        
        # 4. Sustainability Metrics (Horizontal Bar)
        sustainability_metrics = ['Extraction Rate', 'Sustainability Index', 'Safety Margin', 'Stress Level Score']
        
        # Normalize stress level to a score (lower is better)
        stress_mapping = {'low_stress': 25, 'moderate_stress': 50, 'high_stress': 75, 'critical_stress': 100}
        stress_score = stress_mapping.get(location_data['stress_level'], 50)
        
        sustain_values = [extraction_rate, 
                         min(location_data['sustainability_index'], 300),  # cap at 300 for visualization
                         max(location_data['safety_margin'], -100),  # cap at -100 for visualization
                         stress_score]
        
        # Normalize values to 0-100 scale for comparison
        normalized_values = [
            extraction_rate,
            (location_data['sustainability_index'] / 300) * 100,
            ((location_data['safety_margin'] + 100) / 200) * 100,  # shift and scale
            100 - stress_score  # invert so higher is better
        ]
        
        colors_sustain = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']
        bars_h = ax4.barh(sustainability_metrics, normalized_values, color=colors_sustain)
        ax4.set_title('Sustainability Metrics (Normalized 0-100)', fontweight='bold')
        ax4.set_xlabel('Score')
        ax4.set_xlim(0, 100)
        
        # Add value labels
        for bar, original_val, metric in zip(bars_h, sustain_values, sustainability_metrics):
            width = bar.get_width()
            if metric == 'Sustainability Index':
                label = f'{original_val:.1f}'
            elif metric == 'Safety Margin':
                label = f'{original_val:.1f}%'
            elif metric == 'Stress Level Score':
                label = location_data['stress_level'].replace('_', ' ').title()
            else:
                label = f'{original_val:.1f}%'
            
            ax4.text(width + 2, bar.get_y() + bar.get_height()/2.,
                    label, ha='left', va='center', fontsize=9)
        
        # 5. Firka Status Distribution (Stacked Bar)
        firka_categories = ['Over-Exploited', 'Semi-Critical', 'Safe']
        firka_counts = [location_data['over_exploited_firkas'],
                       location_data['semi_critical_firkas'],
                       location_data['safe_firkas']]
        firka_colors = ['#dc3545', '#ffc107', '#28a745']
        
        ax5.bar(['Firka Status'], [sum(firka_counts)], color='lightgray', alpha=0.3)
        
        bottom = 0
        for count, color, label in zip(firka_counts, firka_colors, firka_categories):
            if count > 0:
                ax5.bar(['Firka Status'], [count], bottom=bottom, color=color, label=label)
                ax5.text(0, bottom + count/2, str(count), ha='center', va='center', fontweight='bold')
                bottom += count
        
        ax5.set_title('Firka Status Distribution', fontweight='bold')
        ax5.set_ylabel('Number of Firkas')
        ax5.legend()
        ax5.set_ylim(0, sum(firka_counts) + 1)
        
        # 6. Recharge vs Draft Comparison
        recharge_draft = ['Total Recharge', 'Total Draft']
        recharge_draft_values = [location_data['total_recharge'], location_data['total_draft']]
        colors_rd = ['#2ecc71', '#e74c3c']
        
        bars_rd = ax6.bar(recharge_draft, recharge_draft_values, color=colors_rd)
        ax6.set_title('Recharge vs Draft (ha-m)', fontweight='bold')
        ax6.set_ylabel('Volume (ha-m)')
        
        # Add value labels
        for bar, value in zip(bars_rd, recharge_draft_values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + (max(recharge_draft_values) * 0.01),
                    f'{value:,.0f}', ha='center', va='bottom', fontsize=9)
        
        # Add balance indicator
        balance_ratio = location_data['total_recharge'] / location_data['total_draft']
        balance_text = f"Balance Ratio: {balance_ratio:.2f}"
        if balance_ratio > 1.2:
            balance_color = '#2ecc71'
            balance_status = "(Surplus)"
        elif balance_ratio > 1.0:
            balance_color = '#f39c12'
            balance_status = "(Balanced)"
        else:
            balance_color = '#e74c3c'
            balance_status = "(Deficit)"
        
        ax6.text(0.5, 0.95, f"{balance_text} {balance_status}", 
                transform=ax6.transAxes, ha='center', va='top', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=balance_color, alpha=0.7),
                fontweight='bold', color='white')
        
        # Main title
        fig.suptitle(f'Comprehensive Groundwater Analysis: {location_title}', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save to BytesIO
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        img_buffer.seek(0)
        image_bytes = img_buffer.getvalue()
        img_buffer.close()
        
        return image_bytes
        
    except Exception as e:
        print(f"Error generating single location analysis chart: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_key_metrics_summary_chart(groundwater_data, location_name):
    """
    Generates a focused chart showing only the 5 most critical metrics for users.
    """
    try:
        flattened_data = groundwater_data.get('flattened_data', [])
        
        # Find data for the requested location
        location_data = None
        for item in flattened_data:
            if location_name.lower() in item['location'].lower():
                location_data = item
                break
        
        if not location_data:
            return None
        
        # Create a single chart with 5 key metrics
        fig, ax = plt.subplots(figsize=(14, 8))
        
        location_title = location_data['location'].title()
        
        # Define the 5 most important metrics for users
        metrics = ['Extraction\nRate (%)', 'Safety\nMargin (%)', 'Sustainability\nIndex', 
                  'Net Balance\n(ha-m)', 'Firka Risk\nScore']
        
        # Calculate firka risk score (0-100, higher = more risk)
        total_firkas = (location_data['over_exploited_firkas'] + 
                       location_data['semi_critical_firkas'] + 
                       location_data['safe_firkas'])
        
        if total_firkas > 0:
            risk_score = ((location_data['over_exploited_firkas'] * 100 + 
                          location_data['semi_critical_firkas'] * 50) / total_firkas)
        else:
            risk_score = 0
        
        values = [
            location_data['extraction_rate_total'],
            location_data['safety_margin'],
            location_data['sustainability_index'],
            location_data['net_balance'] / 1000,  # Convert to thousands for readability
            risk_score
        ]
        
        # Color code based on performance (green = good, red = bad)
        def get_metric_color(metric, value):
            if metric == 'Extraction\nRate (%)':
                if value <= 70: return '#28a745'
                elif value <= 90: return '#ffc107'
                elif value <= 100: return '#fd7e14'
                else: return '#dc3545'
            elif metric == 'Safety\nMargin (%)':
                if value >= 30: return '#28a745'
                elif value >= 10: return '#ffc107'
                elif value >= 0: return '#fd7e14'
                else: return '#dc3545'
            elif metric == 'Sustainability\nIndex':
                if value >= 150: return '#28a745'
                elif value >= 100: return '#ffc107'
                elif value >= 75: return '#fd7e14'
                else: return '#dc3545'
            elif metric == 'Net Balance\n(ha-m)':
                if value >= 2: return '#28a745'
                elif value >= 0: return '#ffc107'
                elif value >= -1: return '#fd7e14'
                else: return '#dc3545'
            elif metric == 'Firka Risk\nScore':
                if value <= 20: return '#28a745'
                elif value <= 40: return '#ffc107'
                elif value <= 70: return '#fd7e14'
                else: return '#dc3545'
            return '#3498db'
        
        colors = [get_metric_color(metric, value) for metric, value in zip(metrics, values)]
        
        # Create bars
        bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Customize the chart
        ax.set_title(f'Key Groundwater Metrics: {location_title}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value, metric in zip(bars, values, metrics):
            height = bar.get_height()
            
            # Format label based on metric type
            if 'ha-m' in metric:
                label = f'{value*1000:,.0f}'  # Convert back to original scale
            elif 'Score' in metric:
                label = f'{value:.0f}'
            else:
                label = f'{value:.1f}'
            
            # Position label
            if height >= 0:
                va = 'bottom'
                y_pos = height + (max(values) * 0.02)
            else:
                va = 'top'
                y_pos = height - (max(values) * 0.02)
                
            ax.text(bar.get_x() + bar.get_width()/2., y_pos, label,
                   ha='center', va=va, fontweight='bold', fontsize=10)
        
        # Add status legend
        status_text = f"Overall Status: {location_data['overall_status'].replace('_', ' ').title()}"
        status_color = get_metric_color('Extraction\nRate (%)', location_data['extraction_rate_total'])
        
        ax.text(0.02, 0.98, status_text, transform=ax.transAxes, 
               bbox=dict(boxstyle="round,pad=0.5", facecolor=status_color, alpha=0.8),
               fontsize=12, fontweight='bold', color='white', va='top')
        
        plt.tight_layout()
        
        # Save to BytesIO
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        img_buffer.seek(0)
        image_bytes = img_buffer.getvalue()
        img_buffer.close()
        
        return image_bytes
        
    except Exception as e:
        print(f"Error generating key metrics chart: {e}")
        import traceback
        traceback.print_exc()
        return None


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

def generate_taluk_pdf_report(groundwater_data, taluk_name):
    """Generates a comprehensive PDF report for a specific taluk using reportlab."""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch, cm
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from datetime import datetime
        from io import BytesIO
        import textwrap
        
        # Find the taluk data
        flattened_data = groundwater_data.get('flattened_data', [])
        taluk_data = None
        
        for item in flattened_data:
            if taluk_name.lower() in item['location'].lower():
                taluk_data = item
                break
        
        if not taluk_data:
            print(f"No data found for taluk: {taluk_name}")
            return None
        
        # Create PDF buffer
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=72)
        
        styles = getSampleStyleSheet()
        
        # Create custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # center
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=12
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6
        )
        
        # Build story (content)
        story = []
        
        # Title
        story.append(Paragraph(f"GROUNDWATER REPORT: {taluk_data['location'].upper()}", title_style))
        
        # Metadata
        story.append(Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
        story.append(Paragraph(f"<b>District:</b> Erode", normal_style))
        story.append(Spacer(1, 20))
        
        # 1. Executive Summary
        story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
        
        status_descriptions = {
            'safe': 'GOOD',
            'semi_critical': 'WARNING', 
            'critical': 'CRITICAL',
            'over_exploited': 'EMERGENCY'
        }
        
        status_desc = status_descriptions.get(taluk_data['overall_status'], 'UNKNOWN')
        
        extraction_rate = taluk_data['extraction_rate_total']
        status_text = ""
        if extraction_rate > 100:
            status_text = f"CRITICAL - Using {extraction_rate:.1f}% of available water (over-exploited)"
        elif extraction_rate > 90:
            status_text = f"HIGH - Using {extraction_rate:.1f}% of available water (approaching critical)"
        else:
            status_text = f"ACCEPTABLE - Using {extraction_rate:.1f}% of available water"
        
        net_balance = taluk_data['net_balance']
        balance_text = f"Water surplus: {net_balance:,.0f} ha-m" if net_balance >= 0 else f"Water deficit: {abs(net_balance):,.0f} ha-m"
        
        summary_text = f"""
        <b>Overall Status:</b> {status_desc} - {taluk_data['overall_status'].replace('_', ' ').title()}
        <b>Extraction Rate:</b> {status_text}
        <b>Water Balance:</b> {balance_text}
        """
        
        story.append(Paragraph(summary_text, normal_style))
        story.append(Spacer(1, 15))
        
        # 2. Key Metrics Table
        story.append(Paragraph("KEY PERFORMANCE INDICATORS", heading_style))
        
        metrics_data = [
            ['Metric', 'Value', 'Status'],
            ['Extraction Rate', f"{taluk_data['extraction_rate_total']:.1f}%", 
             'CRITICAL' if extraction_rate > 100 else 'HIGH' if extraction_rate > 90 else 'ACCEPTABLE'],
            ['Safety Margin', f"{taluk_data['safety_margin']:.1f}%", 
             'GOOD' if taluk_data['safety_margin'] > 30 else 'LOW' if taluk_data['safety_margin'] > 10 else 'CRITICAL'],
            ['Sustainability Index', f"{taluk_data['sustainability_index']:.1f}", 
             'GOOD' if taluk_data['sustainability_index'] > 150 else 'FAIR' if taluk_data['sustainability_index'] > 100 else 'POOR'],
            ['Net Water Balance', f"{taluk_data['net_balance']:,.0f} ha-m", 
             'SURPLUS' if taluk_data['net_balance'] >= 0 else 'DEFICIT'],
            ['Total Availability', f"{taluk_data['total_availability']:,.0f} ha-m", ''],
            ['Total Draft', f"{taluk_data['total_draft']:,.0f} ha-m", ''],
            ['Total Recharge', f"{taluk_data['total_recharge']:,.0f} ha-m", '']
        ]
        
        table = Table(metrics_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(Spacer(1, 15))
        
        # 3. Water Usage Breakdown
        story.append(Paragraph("WATER USAGE DISTRIBUTION", heading_style))
        
        total_draft = taluk_data['total_draft']
        agriculture_pct = (taluk_data['agriculture_draft'] / total_draft) * 100
        domestic_pct = (taluk_data['domestic_draft'] / total_draft) * 100
        industry_pct = (taluk_data['industry_draft'] / total_draft) * 100
        
        usage_data = [
            ['Sector', 'Percentage', 'Volume (ha-m)'],
            ['Agriculture', f"{agriculture_pct:.1f}%", f"{taluk_data['agriculture_draft']:,.0f}"],
            ['Domestic', f"{domestic_pct:.1f}%", f"{taluk_data['domestic_draft']:,.0f}"],
            ['Industry', f"{industry_pct:.1f}%", f"{taluk_data['industry_draft']:,.0f}"],
            ['TOTAL', '100.0%', f"{total_draft:,.0f}"]
        ]
        
        usage_table = Table(usage_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        usage_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, -1), (-1, -1), colors.darkgrey),
            ('TEXTCOLOR', (0, -1), (-1, -1), colors.white),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(usage_table)
        story.append(Spacer(1, 15))
        
        # 4. Firka Status
        story.append(Paragraph("AREA-WISE STATUS (FIRKAS)", heading_style))
        
        total_firkas = (taluk_data['over_exploited_firkas'] + 
                       taluk_data['semi_critical_firkas'] + 
                       taluk_data['safe_firkas'])
        
        firka_data = [
            ['Status', 'Number of Areas', 'Percentage'],
            ['Over-exploited', str(taluk_data['over_exploited_firkas']), 
             f"{(taluk_data['over_exploited_firkas']/total_firkas*100):.1f}%"],
            ['Semi-critical', str(taluk_data['semi_critical_firkas']), 
             f"{(taluk_data['semi_critical_firkas']/total_firkas*100):.1f}%"],
            ['Safe', str(taluk_data['safe_firkas']), 
             f"{(taluk_data['safe_firkas']/total_firkas*100):.1f}%"],
            ['TOTAL', str(total_firkas), '100.0%']
        ]
        
        firka_table = Table(firka_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        firka_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, -1), (-1, -1), colors.darkgrey),
            ('TEXTCOLOR', (0, -1), (-1, -1), colors.white),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(firka_table)
        story.append(Spacer(1, 15))
        
        # 5. Recommendations
        story.append(Paragraph("RECOMMENDATIONS", heading_style))
        
        recommendations = []
        
        if taluk_data['extraction_rate_total'] > 100:
            recommendations = [
                "🚨 URGENT ACTION REQUIRED:",
                "• Implement immediate water conservation measures",
                "• Convert to drip irrigation for agriculture",
                "• Enforce strict industrial water usage limits",
                "• Consider temporary water rationing",
                "• Promote rainwater harvesting systems",
                "• Conduct public awareness campaigns"
            ]
        elif taluk_data['extraction_rate_total'] > 90:
            recommendations = [
                "HIGH PRIORITY: Water conservation needed",
                "• Promote efficient irrigation techniques",
                "• Conduct water audit for major consumers",
                "• Develop drought contingency plans",
                "• Invest in water recycling systems",
                "• Monitor groundwater levels weekly"
            ]
        else:
            recommendations = [
                "MAINTAIN GOOD PRACTICES:",
                "• Continue regular water monitoring",
                "• Promote sustainable water use habits",
                "• Maintain water conservation infrastructure",
                "• Prepare long-term water security plans",
                "• Encourage community participation"
            ]
        
        for rec in recommendations:
            if rec.startswith("🚨") or rec.startswith("•"):
                story.append(Paragraph(rec, normal_style))
            else:
                story.append(Paragraph(f"<b>{rec}</b>", normal_style))
        
        story.append(Spacer(1, 15))
        
        # 6. Contact Information
        story.append(Paragraph("CONTACT INFORMATION", heading_style))
        
        contacts = [
            "• District Water Management Department: 044-XXXX-XXXX",
            "• Groundwater Authority Helpdesk: 1800-XXX-XXXX",
            "• Emergency Water Services: 1077 (Toll-free)",
            "• Environmental Protection Agency: 1800-XXX-XXXX",
            "• Local Municipal Office: Contact your ward office"
        ]
        
        for contact in contacts:
            story.append(Paragraph(contact, normal_style))
        
        # Footer
        story.append(Spacer(1, 20))
        story.append(Paragraph("<i>Generated by Hydra AI Groundwater Monitoring System - Comprehensive Water Resource Analysis</i>", 
                             ParagraphStyle('Footer', parent=styles['Italic'], fontSize=8, alignment=1)))
        
        # Build PDF
        doc.build(story)
        pdf_buffer.seek(0)
        
        print(f"PDF generated successfully for {taluk_name}")
        return pdf_buffer.getvalue()
        
    except Exception as e:
        print(f"Error generating PDF report: {e}")
        import traceback
        traceback.print_exc()
        return None

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