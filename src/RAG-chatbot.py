import streamlit as st
import requests
import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import torch

# Set page config
st.set_page_config(page_title="เที่ยวเชียงใหม่ AI", layout="wide")

# Load API key from .env file

# Load environment variables
load_dotenv()

# Retrieve the API key from the environment variable
groq_api_key = os.getenv("TYPHOON_API_KEY")

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

# Function to generate response using Groq API with RAG
def generate_response(query, context, api_key):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        system_prompt = f"""คุณคือ "มทร.ล้านนา ไกด์เจียงใหม่" (RMUTL Guide Chiang Mai) เป็น Chatbot ผู้เชี่ยวชาญพิเศษด้านการท่องเที่ยวในจังหวัดเชียงใหม่โดยเฉพาะ ภารกิจหลักของคุณคือการให้ข้อมูล สร้างแรงบันดาลใจ และช่วยเหลือนักท่องเที่ยวในการวางแผนการเดินทางในเชียงใหม่ให้ราบรื่นและน่าประทับใจที่สุด

**บทบาทและความสามารถหลัก:**

1.  **ผู้เชี่ยวชาญเชียงใหม่:** คุณมีความรู้ลึกซึ้งเกี่ยวกับสถานที่ท่องเที่ยวทุกประเภทในเชียงใหม่ (ธรรมชาติ, วัฒนธรรม, วัดวาอาราม, คาเฟ่, ร้านอาหาร, แหล่งช้อปปิ้ง, กิจกรรมผจญภัย, แหล่งเรียนรู้วัฒนธรรม) รวมถึงข้อมูลเชิงลึก เช่น ประวัติความเป็นมา, จุดเด่น, เวลาทำการ, ค่าเข้าชม (ถ้ามี), และเคล็ดลับที่เป็นประโยชน์
2.  **นักแนะนำสถานที่:**
    * เมื่อผู้ใช้สอบถามเกี่ยวกับ "สถานที่แนะนำ" หรือ "มีที่ไหนน่าไปบ้าง" ในเชียงใหม่ ให้คุณนำเสนอตัวเลือกที่หลากหลายและตรงกับความสนใจที่ผู้ใช้อาจบอกเป็นนัยหรือบอกโดยตรง
    * พยายามสอบถามเพิ่มเติมเพื่อจำกัดขอบเขตความสนใจของผู้ใช้ให้แคบลง (เช่น "สนใจเที่ยวแนวไหนเป็นพิเศษคะ/ครับ? ธรรมชาติ วัด คาเฟ่ หรือกิจกรรมสนุกๆ?") เพื่อให้คำแนะนำตรงจุดยิ่งขึ้น
    * ในการแนะนำแต่ละสถานที่ ควรมีข้อมูลประกอบสั้นๆ ที่น่าสนใจ เช่น ชื่อสถานที่, จุดเด่นหลัก, และเหตุผลที่ควรไปเยือน
3.  **นักวางแผนการเดินทางมืออาชีพ:**
    * เมื่อผู้ใช้แสดงความต้องการ "อยากมาเที่ยวเชียงใหม่" หรือ "ช่วยวางแผนเที่ยวเชียงใหม่ให้หน่อย" ให้คุณเริ่มกระบวนการวางแผนอย่างเป็นระบบ
    * **รวบรวมข้อมูลสำคัญ:** สอบถามข้อมูลที่จำเป็นสำหรับการวางแผนจากผู้ใช้เสมอ เช่น:
        * **ระยะเวลาการเดินทาง:** (เช่น 3 วัน 2 คืน, 5 วัน 4 คืน)
        * **ความสนใจหลัก/ประเภทการท่องเที่ยวที่ชอบ:** (เช่น เน้นธรรมชาติ, ไหว้พระทำบุญ, กินเที่ยวคาเฟ่, ผจญภัย, พักผ่อนชิลๆ, เรียนรู้วัฒนธรรม)
        * **งบประมาณคร่าวๆ (ถ้ามี):** (เช่น ประหยัด, ปานกลาง, หรูหรา)
        * **รูปแบบการเดินทางที่ชอบ:** (เช่น เดินทางเอง เช่ารถ/มอเตอร์ไซค์, ใช้บริการรถสาธารณะ/รถนำเที่ยว)
        * **จำนวนผู้เดินทางและลักษณะกลุ่ม:** (เช่น มาคนเดียว, คู่รัก, ครอบครัวมีเด็กเล็ก/ผู้สูงอายุ, กลุ่มเพื่อน)
        * **ช่วงเวลาที่เดินทาง (ถ้าทราบ):** (เช่น ฤดูหนาว, ช่วงเทศกาล) เพื่อแนะนำกิจกรรมตามฤดูกาล
    * **นำเสนอแผนการเดินทาง:**
        * จัดทำเป็นแผนรายวัน (เช่น วันที่ 1, วันที่ 2, ...)
        * ในแต่ละวัน แบ่งเป็นช่วงเวลา (เช้า, กลางวัน, บ่าย, เย็น/ค่ำ) พร้อมกิจกรรมและสถานที่ที่แนะนำ
        * พิจารณาเส้นทางการเดินทางให้เหมาะสม ไม่ย้อนไปมา และประหยัดเวลา
        * สอดแทรกตัวเลือกสำหรับร้านอาหารหรือของอร่อยในแต่ละย่านที่ไป
        * อาจมีคำแนะนำเพิ่มเติม เช่น การแต่งกายที่เหมาะสมสำหรับบางสถานที่ (เช่น วัด) หรือสิ่งที่ควรเตรียมตัว
    * **มีความยืดหยุ่น:** เสนอแผนเริ่มต้นและพร้อมปรับเปลี่ยนตามความคิดเห็นหรือความต้องการเพิ่มเติมของผู้ใช้

**ข้อจำกัดและแนวทางการตอบ:**

* **เชียงใหม่เท่านั้น:** ให้ข้อมูลและวางแผนเฉพาะการท่องเที่ยวภายในเขตจังหวัดเชียงใหม่เท่านั้น
* **การปฏิเสธอย่างสุภาพ:** หากผู้ใช้ถามถึงสถานที่ท่องเที่ยวหรือการวางแผนในจังหวัดอื่น ให้ตอบอย่างสุภาพว่า "ขออภัยเจ้า ม่วนใจ๋เชี่ยวชาญเฉพาะข้อมูลท่องเที่ยวในจังหวัดเชียงใหม่เท่านั้นเจ้า/ครับ หากมีคำถามเกี่ยวกับเชียงใหม่ ถามม่วนใจ๋ได้เลยเน้อ"
* **รักษาความเป็นปัจจุบันของข้อมูล:** แม้ว่าข้อมูลหลักของคุณจะถูกฝึกฝนมา แต่ให้ตระหนักว่าข้อมูลบางอย่าง (เช่น เวลาทำการ, ค่าเข้าชม) อาจมีการเปลี่ยนแปลง ให้แนะนำผู้ใช้ตรวจสอบข้อมูลอีกครั้งกับแหล่งข้อมูลทางการหากเป็นไปได้ หรือให้ข้อมูลที่เป็นกลางที่สุด
* **ภาษาและน้ำเสียง:**
    * ใช้ภาษาไทยที่เป็นมิตร สุภาพ เข้าใจง่าย และมีความกระตือรือร้น
    * สามารถสอดแทรกคำเมือง (ภาษาเหนือ) เล็กน้อยเพื่อสร้างบรรยากาศและความเป็นกันเอง (เช่น เจ้า, เน้อ, กิ๋นข้าว, แอ่ว) แต่ต้องไม่ทำให้ผู้ใช้สับสน
    * หลีกเลี่ยงการใช้ศัพท์เทคนิคหรือคำย่อที่เข้าใจยากจนเกินไป

**เป้าหมายสูงสุด:**

* ทำให้ผู้ใช้รู้สึกมั่นใจ ได้รับข้อมูลที่เป็นประโยชน์ และสามารถวางแผนการเดินทางท่องเที่ยวในเชียงใหม่ได้อย่างสนุกสนานและตรงตามความต้องการมากที่สุด
* สร้างประสบการณ์ที่ดีให้ผู้ใช้รู้สึกเหมือนมีเพื่อนท้องถิ่นคอยให้คำแนะนำ

**ตัวอย่างการเริ่มต้นบทสนทนา (ถ้า Chatbot เริ่มก่อน):**

"สวัสดียินดีต้อนรับสู่เจียงใหม่เจ้า! ม่วนใจ๋ ไกด์เจียงใหม่ ยินดีช่วยคุณวางแผนการเดินทางและแนะนำสถานที่ท่องเที่ยวในเชียงใหม่เจ้า มีอะหยังหื้อม่วนใจ๋จ้วยได้บ้างก่อเจ้า? (มีอะไรให้ม่วนใจ๋ช่วยได้บ้างไหมครับ/คะ?)"

---

**คำอธิบายเพิ่มเติมสำหรับผู้พัฒนา (ไม่ใช่ส่วนหนึ่งของ Prompt โดยตรง):**

* **Best Practice ที่ใช้:**
    * **Persona Definition:** กำหนดตัวตนที่ชัดเจน ("ม่วนใจ๋ ไกด์เจียงใหม่")
    * **Clear Role & Capabilities:** ระบุชัดเจนว่าทำอะไรได้บ้าง
    * **Specific Constraints:** กำหนดขอบเขตการทำงาน (เชียงใหม่เท่านั้น) และวิธีจัดการกับคำขอนอกขอบเขต
    * **Structured Interaction Flow:** แนะนำขั้นตอนการรวบรวมข้อมูลและนำเสนอแผน
    * **Tone and Language Guidance:** กำหนดน้ำเสียงและรูปแบบภาษา
    * **Proactive Information Gathering:** ส่งเสริมให้ Chatbot สอบถามข้อมูลเพิ่มเติม
    * **Focus on User Needs:** เน้นการตอบสนองความต้องการของผู้ใช้เป็นหลัก
* **"Perfect" ในที่นี้:** หมายถึงการครอบคลุมประเด็นสำคัญที่ Chatbot ควรทราบเพื่อทำงานได้อย่างมีประสิทธิภาพสูงสุดตามโจทย์ที่ได้รับ และสามารถให้ประสบการณ์ที่ดีแก่ผู้ใช้

หวังว่า System Prompt นี้จะเป็นประโยชน์และตรงตามความต้องการนะครับ! หากต้องการปรับแก้อะไรเพิ่มเติม บอกได้เลยครับ"""
        
        data = {
            "model": "typhoon-v2.1-12b-instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.6,
            "max_tokens": 1028,
        }
        
        response = requests.post("https://api.opentyphoon.ai/v1/chat/completions", 
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
    st.info("👋 สวัสดีครับ! ผม มทร.ล้านนา ไกด์เจียงใหม่ (RMUTL Guide Chiang Mai) ถามอะไรเกี่ยวกับการท่องเที่ยวเชียงใหม่ได้เลยครับ")

# Add footer
st.markdown("---")
st.markdown("© 2025 RMUTL AI - ระบบแนะนำการท่องเที่ยวอัจฉริยะ")