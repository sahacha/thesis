# MY BACHELOR THESIS
# 🚀 RAG System Full Pipeline Q&A Chiang Mai Travel

## ✨ Features Implemented

### 🔧 Core Components
1. **ChromaDB Cloud Integration** - เชื่อมต่อกับ ChromaDB Cloud
2. **Langchain Integration** - ใช้ Langchain สำหรับ RAG Pipeline
3. **Gemini LLM** - Google's Gemini model สำหรับ Generation
4. **Hybrid Retrieval** - รวม Vector Search และ BM25 Keyword Search
5. **Document Reranking** - ปรับปรุงความแม่นยำด้วย Cross-Encoder
6. **Langsmith Monitoring** - ติดตามและวิเคราะห์ performance

### 📊 Collections Support
- **Attractions** - สถานที่ท่องเที่ยวในเชียงใหม่
- **Hotels** - ที่พักและโรงแรม
- **Restaurants** - ร้านอาหารและคาเฟ่

### 🎯 Key Capabilities
- **Multi-language Support** - รองรับภาษาไทยและอังกฤษ
- **Real-time Retrieval** - ค้นหาและตอบคำถามแบบ real-time
- **Performance Tracking** - ติดตาม response time และ accuracy
- **Conversation History** - บันทึกประวัติการสนทนา
- **Batch Processing** - ประมวลผลคำถามหลายข้อพร้อมกัน

## 🛠️ Usage Instructions

### Basic Usage
```python
# Ask a single question
result = rag_pipeline.query("แนะนำสถานที่ท่องเที่ยวในเชียงใหม่")
print(result['answer'])

# Use interactive demo
demo.ask("What are the best temples in Chiang Mai?")

# Run quick demonstration
demo.quick_demo()
```

### Batch Processing
```python
questions = [
    "Popular restaurants in Chiang Mai",
    "Best hotels in old city",
    "Temple recommendations"
]
results = rag_pipeline.batch_query(questions)
```

### Performance Analysis
```python
# Analyze system performance
analysis = analyze_rag_performance(results)
print(f"Success rate: {analysis['success_rate']:.1f}%")
print(f"Avg response time: {analysis['avg_response_time']:.2f}s")
```

## 🔧 Configuration

### Required Environment Variables
```bash
export CHROMA_API_KEY="your-chroma-api-key"
export GEMINI_API_KEY="your-gemini-api-key" 
export LANGSMITH_API_KEY="your-langsmith-api-key"  # Optional
export OPENAI_API_KEY="your-openai-api-key"       # Fallback
```

### Pipeline Parameters
- **top_k**: 10 (initial retrieval)
- **final_k**: 5 (after reranking)  
- **reranker_model**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **gemini_model**: gemini-1.5-pro

## 📈 Monitoring with Langsmith

The system automatically tracks:
- Query performance
- Response quality
- Token usage
- Error rates
- User sessions

## 🚦 System Status

✅ **Ready Components:**
- ChromaDB Connection
- Hybrid Retriever  
- Document Reranker
- RAG Pipeline
- Interactive Demo