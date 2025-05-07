import streamlit as st
import requests
import os
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Set page config
st.set_page_config(page_title="เที่ยวเชียงใหม่ AI", layout="wide")

# Load API key from .env file

# Load environment variables
load_dotenv()

# Retrieve the API key from the environment variable
groq_api_key = os.getenv("GROQ_API_KEY")

# Check if the API key is loaded
if not groq_api_key:
    st.error("API key not found. Please set GROQ_API_KEY in your .env file.")

# Sidebar
with st.sidebar:
    st.title("🏔️ เที่ยวเชียงใหม่ AI")
    st.markdown("---")
    st.markdown("ระบบ AI แนะนำการท่องเที่ยวเชียงใหม่แบบส่วนตัว")
    st.markdown("พัฒนาโดยใช้เทคโนโลยี RAG และ Groq LLM API")
    st.markdown("---")
    
    # Add some useful information in sidebar
    st.subheader("เกี่ยวกับระบบ")
    st.markdown("""
    - ถามคำถามเกี่ยวกับการท่องเที่ยวเชียงใหม่ได้ทุกอย่าง
    - ข้อมูลจากแหล่งท่องเที่ยวที่น่าเชื่อถือ
    - ระบบตอบกลับอัตโนมัติด้วย AI
    """)

# Main content
st.title("🌴 เที่ยวเชียงใหม่กับ AI ผู้ช่วยส่วนตัว")
st.markdown("ถามคำถามเกี่ยวกับการท่องเที่ยวเชียงใหม่ได้เลย!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for vector database
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

# Function to initialize vector database
@st.cache_resource
def initialize_vectordb():
    try:
        # List of popular Chiang Mai tourism websites
        urls = [
            "https://www.tourismthailand.org/Destinations/Provinces/Chiang-Mai/217",
            "https://www.chiangmaitourist.com/",
            "https://www.lonelyplanet.com/thailand/chiang-mai-province/chiang-mai", 
            "https://wikitravel.org/en/Chiang_Mai",
            "https://www.tripadvisor.com/Tourism-g293917-Chiang_Mai-Vacations.html"
        ]
        
        # Load documents from URLs
        documents = []
        for url in urls:
            loader = WebBaseLoader(url)
            data = loader.load()
            documents.extend(data)
            
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # Initialize embeddings model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create vector database
        vectordb = FAISS.from_documents(chunks, embeddings)
        
        return vectordb
    
    except Exception as e:
        st.error(f"Error initializing vector database: {str(e)}")
        return None

# Load vector database if not already loaded
if st.session_state.vectordb is None:
    with st.spinner("กำลังโหลดข้อมูลการท่องเที่ยวเชียงใหม่..."):
        st.session_state.vectordb = initialize_vectordb()

# Function to generate response using Groq API with RAG
def generate_response(query, context, api_key):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        system_prompt = f"""คุณเป็น AI ผู้เชี่ยวชาญด้านการท่องเที่ยวเชียงใหม่ 
        ให้ข้อมูลที่เป็นประโยชน์ ถูกต้อง และเป็นกันเองกับนักท่องเที่ยว 
        ตอบคำถามโดยใช้ข้อมูลที่ให้มาเป็นหลัก 
        โดยข้อมูลด้านล่างเป็นข้อมูลอ้างอิงสำหรับตอบคำถาม:
        
        {context}
        
        ตอบคำถามเป็นภาษาไทย แสดงข้อมูลที่เป็นประโยชน์ต่อนักท่องเที่ยว ถ้าไม่มีข้อมูลในบริบทที่ให้มา ตอบตามความรู้ทั่วไปเกี่ยวกับเชียงใหม่"""
        
        data = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.5,
            "max_tokens": 1024,
        }
        
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                                headers=headers, 
                                data=json.dumps(data))
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            # ในกรณีที่ API มีปัญหา ให้ใช้ข้อความตอบกลับสำรอง
            return f"ขออภัยค่ะ ระบบมีปัญหาเล็กน้อย กรุณาลองใหม่อีกครั้งในอีกสักครู่ (Error: {response.status_code})"
    
    except Exception as e:
        return "ขออภัยค่ะ เกิดข้อผิดพลาดในการเชื่อมต่อกับระบบ กรุณาลองใหม่อีกครั้ง"

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
user_query = st.chat_input("ถามอะไรเกี่ยวกับเชียงใหม่ก็ได้เลย...")

# Generate and display response
if user_query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Display assistant response with spinner
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("กำลังคิดคำตอบ...")
        
        # Search vector database for relevant context
        if st.session_state.vectordb:
            docs = st.session_state.vectordb.similarity_search(user_query, k=5)
            context = "\n\n".join([doc.page_content for doc in docs])
        else:
            context = "ไม่พบข้อมูลในฐานข้อมูล"
        
        # Generate response
        response = generate_response(user_query, context, groq_api_key)
        message_placeholder.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display welcome message if no messages yet
if not st.session_state.messages:
    st.info("👋 สวัสดีครับ! ผมเป็น AI ผู้ช่วยแนะนำการท่องเที่ยวเชียงใหม่ ถามอะไรเกี่ยวกับการท่องเที่ยวเชียงใหม่ได้เลยครับ")

# Add footer
st.markdown("---")
st.markdown("© 2025 เที่ยวเชียงใหม่ AI - ระบบแนะนำการท่องเที่ยวอัจฉริยะ")