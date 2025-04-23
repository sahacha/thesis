import asyncio
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import sqlite3
from transformers import pipeline
import streamlit as st
from datetime import datetime
import json

# การตั้งค่า
WONGNAI_URL = "https://www.wongnai.com/reviews"
X_URL = "https://x.com/explore"
MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # รองรับภาษาไทย
)
FAISS_INDEX_PATH = "faiss_index.bin"
DB_PATH = "scraped_data.db"


# สร้างฐานข้อมูล SQLite
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS scraped_data
                 (id TEXT PRIMARY KEY, source TEXT, content TEXT, timestamp TEXT)"""
    )
    conn.commit()
    conn.close()


# Scraping จาก Wongnai
async def scrape_wongnai():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(WONGNAI_URL)
        await page.wait_for_load_state("networkidle")

        # ดึงรีวิว (สมมติว่าเป็น div.review-item)
        content = await page.content()
        soup = BeautifulSoup(content, "html.parser")
        reviews = soup.find_all("div", class_="review-item")
        new_data = []

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        for review in reviews:
            review_id = review.get("data-id", "")
            text = review.get_text(strip=True)
            if text and review_id:
                c.execute("SELECT id FROM scraped_data WHERE id=?", (review_id,))
                if not c.fetchone():
                    new_data.append(
                        {
                            "id": review_id,
                            "source": "wongnai",
                            "content": text,
                            "timestamp": str(datetime.now()),
                        }
                    )
                    c.execute(
                        "INSERT INTO scraped_data (id, source, content, timestamp) VALUES (?, ?, ?, ?)",
                        (review_id, "wongnai", text, str(datetime.now())),
                    )

        conn.commit()
        conn.close()
        await browser.close()
        return new_data


# Scraping จาก X.com
async def scrape_x():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(X_URL)
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(2000)  # รอให้โหลดโพสต์

        content = await page.content()
        soup = BeautifulSoup(content, "html.parser")
        posts = soup.find_all("article", {"data-testid": "tweet"})
        new_data = []

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        for post in posts:
            post_id = post.get("data-tweet-id", "")
            text = post.get_text(strip=True)
            if text and post_id:
                c.execute("SELECT id FROM scraped_data WHERE id=?", (post_id,))
                if not c.fetchone():
                    new_data.append(
                        {
                            "id": post_id,
                            "source": "x",
                            "content": text,
                            "timestamp": str(datetime.now()),
                        }
                    )
                    c.execute(
                        "INSERT INTO scraped_data (id, source, content, timestamp) VALUES (?, ?, ?, ?)",
                        (post_id, "x", text, str(datetime.now())),
                    )

        conn.commit()
        conn.close()
        await browser.close()
        return new_data


# อัปเดต Vector Database
def update_vector_db(new_data):
    model = SentenceTransformer(MODEL_NAME)
    chunks = [item["content"] for item in new_data]
    if not chunks:
        return

    embeddings = model.encode(chunks)

    # โหลดหรือสร้าง FAISS index
    dimension = embeddings.shape[1]
    if not hasattr(update_vector_db, "index"):
        update_vector_db.index = faiss.IndexFlatL2(dimension)
        try:
            update_vector_db.index = faiss.read_index(FAISS_INDEX_PATH)
        except:
            pass

    update_vector_db.index.add(embeddings)
    faiss.write_index(update_vector_db.index, FAISS_INDEX_PATH)

    # บันทึก chunks
    with open("chunks.json", "a", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)


# RAG Pipeline
def answer_query(query):
    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode([query])

    index = faiss.read_index(FAISS_INDEX_PATH)
    D, I = index.search(query_embedding, k=3)

    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    retrieved_texts = [chunks[i]["content"] for i in I[0] if i < len(chunks)]

    generator = pipeline("text-generation", model="gpt2")
    prompt = f"คำถาม: {query}\nข้อมูล: {' '.join(retrieved_texts)}\nคำตอบ: "
    response = generator(prompt, max_length=150, num_return_sequences=1)[0][
        "generated_text"
    ]
    return response


# Scheduling Scraping
async def run_scrapers():
    print("Running scrapers...")
    wongnai_data = await scrape_wongnai()
    x_data = await scrape_x()
    new_data = wongnai_data + x_data
    if new_data:
        update_vector_db(new_data)
        print(f"Updated {len(new_data)} new items")


# Streamlit Interface
def main():
    init_db()
    st.title("Real-time Chatbot")
    query = st.text_input("ถามอะไรก็ได้:")
    if query:
        response = answer_query(query)
        st.write(response)

    # ตั้งเวลา scraping ทุก 10 นาที
    scheduler = AsyncIOScheduler()
    scheduler.add_job(run_scrapers, "interval", minutes=10)
    scheduler.start()


if __name__ == "__main__":
    asyncio.run(main())
