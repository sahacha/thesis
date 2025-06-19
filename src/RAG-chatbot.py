import streamlit as st
import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import google.generativeai as genai  # เพิ่ม Gemini import

# Set page config
st.set_page_config(page_title="เที่ยวเชียงใหม่ AI", layout="wide")

# Load environment variables
load_dotenv()

# Retrieve the Gemini API key from the environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check if the API key is loaded
if not GEMINI_API_KEY:
    st.error("API key not found. Please set GEMINI_API_KEY in your .env file.")

# Sidebar
with st.sidebar:
    st.title("🏔️ เที่ยวเชียงใหม่ AI")
    st.markdown("---")
    st.markdown("ระบบ AI แนะนำการท่องเที่ยวเชียงใหม่แบบส่วนตัว")
    st.markdown("พัฒนาโดยใช้เทคโนโลยี RAG และ Gemini LLM API")
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
            "https://www.tourismthailand.org/Search-result/attraction?destination_id=101&sort_by=datetime_updated_desc&page=1&perpage=15&menu=attraction",
            "https://www.lonelyplanet.com/thailand/chiang-mai-province/chiang-mai", 
            "https://wikitravel.org/en/Chiang_Mai",
            "https://www.tripadvisor.com/Tourism-g293917-Chiang_Mai-Vacations.html",
            "https://www.wongnai.com/trips/travel-at-chiangmai",
        
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

# Function to generate response using Gemini API with RAG

def generate_response(query, context, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        # รวม context เข้ากับ prompt
        prompt = f"""{context}\n\nคำถาม: {query}\nกรุณาตอบโดยอิงตามข้อมูลข้างต้นและบทบาทไกด์เชียงใหม่ที่กำหนดไว้ใน system prompt ด้านล่าง\n\nSystem Prompt:\nคุณคือ \"มทร.ล้านนา ไกด์เจียงใหม่\" (RMUTL Guide Chiang Mai) เป็น Chatbot ผู้เชี่ยวชาญพิเศษด้านการท่องเที่ยวในจังหวัดเชียงใหม่โดยเฉพาะ ภารกิจหลักของคุณคือการให้ข้อมูล สร้างแรงบันดาลใจ และช่วยเหลือนักท่องเที่ยวในการวางแผนการเดินทางในเชียงใหม่ให้ราบรื่นและน่าประทับใจที่สุด\n(ตัด system prompt ให้สั้นลงได้)"""
        response = model.generate_content(prompt, generation_config={"temperature": 0.6, "max_output_tokens": 1024})
        return response.text
    except Exception as e:
        return f"ขออภัยค่ะ เกิดข้อผิดพลาดกับ Gemini API: {str(e)}"

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
user_query = st.chat_input("ถามอะไรเกี่ยวกับเชียงใหม่ก็ได้เลย!")

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
        response = generate_response(user_query, context, GEMINI_API_KEY)
        message_placeholder.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display welcome message if no messages yet
if not st.session_state.messages:
    st.info("👋 สวัสดีครับ! ผม มทร.ล้านนา ไกด์เจียงใหม่ (RMUTL Guide Chiang Mai) ถามอะไรเกี่ยวกับการท่องเที่ยวเชียงใหม่ได้เลยครับ")

# Add footer
st.markdown("---")
st.markdown("© 2025 RMUTL AI - ระบบแนะนำการท่องเที่ยวอัจฉริยะ")