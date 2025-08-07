# RAG Chat Application with Gemini

This is a Streamlit-based RAG (Retrieval-Augmented Generation) chat application that uses Google's Gemini AI model to answer questions about uploaded PDF documents.

## Features

- Upload PDF documents for analysis
- Chat interface with conversation memory
- Source document citations
- PDF preview in sidebar
- Progress indicators during document processing

## Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Google API Key:**
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Copy `.env.example` to `.env`
   - Replace the placeholder with your actual API key

3. **Run the Application:**
   ```bash
   streamlit run chat_rag.py
   ```

## How to Use

1. **Upload a PDF:** Use the file uploader in the sidebar to upload a PDF document
2. **Wait for Processing:** The application will process the document and create embeddings
3. **Start Chatting:** Once processing is complete, you can ask questions about the document
4. **View Sources:** Expand the "Source Documents" section to see relevant excerpts

## Technical Details

- **Embeddings:** BAAI/bge-large-en-v1.5 model for document embeddings
- **Vector Store:** FAISS for efficient similarity search
- **LLM:** Google Gemini 1.5 Flash for response generation
- **Framework:** LangChain for RAG pipeline orchestration

## Troubleshooting

- **API Key Error:** Make sure your Google API key is properly set in the `.env` file
- **Memory Issues:** Large PDFs might require more system memory
- **Slow Processing:** First-time model downloads may take a while
