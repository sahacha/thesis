import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# โหลด Environment Variables (สำหรับ OpenAI API Key)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("กรุณาตั้งค่า OPENAI_API_KEY ใน .env file")

# ขั้นตอนที่ 1: Scrape ข้อมูลจากเว็บ
def scrape_chiangmai_data(url):
    try:
        # ส่ง HTTP Request ไปยัง URL
        response = requests.get(url)
        response.raise_for_status()  # ตรวจสอบว่า Request สำเร็จ
        soup = BeautifulSoup(response.content, 'html.parser')

        # ดึงข้อมูลจาก HTML (ตัวอย่าง: ดึงข้อความจากแท็ก <p> และ <h2>)
        data = []
        for element in soup.find_all(['div', 'p']):
            text = element.get_text(strip=True)
            if text:  # ตรวจสอบว่าไม่ใช่ข้อความว่าง
                data.append(text)

        return "\n".join(data)
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการ Scrape: {e}")
        return ""

# ขั้นตอนที่ 2: บันทึกข้อมูลที่ Scrape ได้
def save_scraped_data(data, filename="chiangmai_data.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(data)
    return filename

# ขั้นตอนที่ 3: สร้าง RAG Pipeline
def create_rag_pipeline(data_file):
    # โหลดข้อมูลจากไฟล์
    loader = TextLoader(data_file, encoding="utf-8")
    documents = loader.load()

    # แบ่งเอกสารเป็น Chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # สร้าง Embeddings และ Vector Store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    db = Chroma.from_documents(docs, embeddings, persist_directory="./chiangmai_db")
    db.persist()

    # สร้าง Retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # สร้าง LLM และ RetrievalQA Chain
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain

# ขั้นตอนที่ 4: ถามคำถามและรับคำตอบ
def ask_question(qa_chain, query):
    result = qa_chain({"query": query})
    return result["result"], result["source_documents"]

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # URL ตัวอย่าง (ควรเปลี่ยนเป็น URL จริงที่เกี่ยวกับการท่องเที่ยวเชียงใหม่)
    url = "https://www.wongnai.com/travel/trips/7caf29b0b05948158ec61dcd3743ae8d"

    # Scrape ข้อมูล
    scraped_data = scrape_chiangmai_data(url)
    if not scraped_data:
        print("ไม่สามารถดึงข้อมูลได้")
        exit()

    # บันทึกข้อมูล
    data_file = save_scraped_data(scraped_data)

    # สร้าง RAG Pipeline
    qa_chain = create_rag_pipeline(data_file)

    # ถามคำถาม
    query = "สถานที่ท่องเที่ยวในเชียงใหม่ที่เหมาะกับครอบครัวมีอะไรบ้าง?"
    answer, sources = ask_question(qa_chain, query)

    # แสดงผลลัพธ์
    print("\nคำตอบ:", answer)
    print("\nแหล่งข้อมูล:")
    for i, doc in enumerate(sources, 1):
        print(f"เอกสาร {i}: {doc.page_content[:200]}...")