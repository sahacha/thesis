from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.indexes import load_index
import json
import os
from pathlib import Path
from evaluate import load

# 1. โหลด Index จาก JSON files
index_dir = Path("./docs/index/index_province77")  # Directory ที่มีไฟล์ index ต่างๆ
print(f"Loading indexes from {index_dir.absolute()}")

# ตรวจสอบว่ามีไฟล์ index ครบทั้ง 5 ไฟล์หรือไม่
expected_files = ["defaultvector_store.json", "docstore.json", "graph_store.json", 
                  "imagevector_store.json", "index_store.json"]
missing_files = [f for f in expected_files if not (index_dir / f).exists()]

if missing_files:
    print(f"Warning: Missing index files: {', '.join(missing_files)}")
else:
    print("All index files found.")

# โหลด index จาก LangChain
index = load_index(str(index_dir))
print(f"Successfully loaded index with {len(index.docstore.docs)} documents")

# 2. สร้าง Vector Store สำหรับ Retrieval
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. สร้าง RAG Pipeline โดยใช้ข้อมูลจาก index
llm = OpenAI(model_name="gpt-4")
retriever = index.vectorstore.as_retriever(search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# 4. ทดสอบและประเมิน
eval_questions = [
    "What is the primary cause of diabetes?",
    "What are the symptoms of hypertension?",
    "How is pneumonia diagnosed?"
]

results = {}
for i, question in enumerate(eval_questions):
    print(f"\nQuestion {i+1}: {question}")
    response = qa_chain.run(question)
    print(f"Response: {response}")
    results[question] = response

# 5. ประเมินด้วยเมตริก BLEU
# สมมติว่ามีคำตอบมาตรฐานใน ground_truth.json
try:
    with open(index_dir / "ground_truth.json", "r") as f:
        ground_truth = json.load(f)
    
    bleu = load("bleu")
    bleu_scores = {}
    
    for question in eval_questions:
        if question in ground_truth:
            score = bleu.compute(predictions=[results[question]], references=[ground_truth[question]])
            bleu_scores[question] = score["bleu"]
            print(f"BLEU Score for '{question}': {score['bleu']}")
        else:
            print(f"No ground truth available for '{question}'")
    
    # บันทึกผลลัพธ์
    with open(index_dir / "eval_results.json", "w") as f:
        json.dump({
            "responses": results,
            "bleu_scores": bleu_scores
        }, f, indent=2)
    
    print(f"\nResults saved to {index_dir / 'eval_results.json'}")
except FileNotFoundError:
    print("\nNo ground_truth.json file found. Saving only responses.")
    with open(index_dir / "eval_results.json", "w") as f:
        json.dump({
            "responses": results
        }, f, indent=2)
    print(f"Responses saved to {index_dir / 'eval_results.json'}")