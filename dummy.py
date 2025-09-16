"""
==================================================================================================
Hydra AI WhatsApp Bot - Final Version with Taluk-Level Location Processing
==================================================================================================
This script launches a sophisticated, AI-powered chatbot that interacts with users via WhatsApp.
This final version can process shared WhatsApp locations, perform reverse geocoding to find the
user's specific taluk and district, and automatically query the relevant RAG database.
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

# Import Whisper for audio transcription
import whisper

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
            if isinstance(json_data, list):
                for i, record in enumerate(json_data):
                    location_name = record.get("locationName", f"Record {i+1}")
                    all_docs.extend(list(flatten_json(record, location_name)))
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
        routing_prompt = ChatPromptTemplate.from_template(
            """Classify the user's question as 'groundwater_data' or 'general_knowledge'.
            - 'groundwater_data': Questions about 'groundwater', 'aquifers', water levels, data, reports, or general water-related topics.
            - 'general_knowledge': All other questions.
            Return only the category name. Question: "{question}" Classification:"""
        )
        self.router_chain = routing_prompt | router_llm

    def route(self, user_id, question):
        chat_history = self.memory_manager.get_history(user_id)
        
        question_lower = question.lower()
        
        matched_taluk = next((taluk for taluk in TALUK_DISTRICT_MAP.keys() if taluk in question_lower), None)
        matched_district = next((dist for dist in DISTRICT_JSON_MAP.keys() if dist in question_lower), None)

        retriever_to_use = None
        source_name = "General AI Knowledge"

        if matched_taluk:
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
                result = self.general_chain.invoke({"chat_history": chat_history, "input": question})
                self.memory_manager.save_context(user_id, {"input": question}, {"output": result.content})
                return {"source_type": "gemini", "result": {"text": result.content}}

        if retriever_to_use:
            rag_chain = get_rag_chain(retriever_to_use)
            result = rag_chain.invoke({"chat_history": chat_history, "input": question})
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

class PerUserMemory:
    def __init__(self):
        self.conversations = {}
    def get_memory(self, user_id):
        if user_id not in self.conversations:
            self.conversations[user_id] = ConversationBufferMemory(return_messages=True)
        return self.conversations[user_id]
    def get_history(self, user_id):
        return self.get_memory(user_id).chat_memory.messages
    def save_context(self, user_id, inputs, outputs):
        self.get_memory(user_id).save_context(inputs, outputs)


# --- 4. WHATSAPP SERVER LOGIC & LOCATION PROCESSING ---
app = Flask(__name__)
hydra_router = None
memory_manager = None
user_states = {}

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

# --- NEW: Geocoding Function using Nominatim ---
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
        
        # Keys can vary, so we check a few common ones for taluk/sub-district
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
        res = requests.get(url, headers=headers); res.raise_for_status(); return res.content
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
                                        # Formulate a highly specific query
                                        query = f"What is the groundwater report for {taluk} in {district} district?"
                                        process_message(from_number, query)
                                    elif district:
                                        # Fallback to district if taluk not found
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
            send_whatsapp_message(from_number, "AI activated! How can I help? You can also share your location for a local report.")
        else:
            send_whatsapp_message(from_number, "Please type 'start' to begin.")
    
    elif current_state == "active":
        if hydra_router:
            router_output = hydra_router.route(from_number, msg_body)
            source_type = router_output["source_type"]
            result = router_output["result"]
            
            if source_type == "rag":
                answer = result.get('answer', 'Sorry, I had an issue.')
                source_info = f"\n\n*Source:* {result.get('source_name', 'Local Documents')}"
                final_message = answer + source_info
            else: # Gemini
                final_message = result.get('text', 'Sorry, I had an issue.') + "\n\n*Source:* AI Generated"
                
            send_whatsapp_message(from_number, final_message)

# --- 5. MAIN APPLICATION EXECUTION ---
if __name__ == "__main__":
    if not all([GEMINI_API_KEY, WHATSAPP_ACCESS_TOKEN, WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_VERIFY_TOKEN]):
        print("Error: Missing required environment variables. Check .env file.")
    else:
        print("Initializing Hydra AI...")
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