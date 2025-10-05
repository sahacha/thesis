from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

import chromadb
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config

try:
    from flashrank import Ranker, RerankRequest

    FLASHRANK_AVAILABLE = True
except ImportError:
    Ranker = None  # type: ignore
    RerankRequest = None  # type: ignore
    FLASHRANK_AVAILABLE = False


def _ensure_session_id() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())


def _get_session_file() -> Path:
    return Config.SESSIONS_DIR / f"session_{st.session_state.session_id}.json"


def _load_session_history() -> None:
    session_file = _get_session_file()
    if session_file.exists():
        try:
            data = json.loads(session_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}

        messages = data.get("messages")
        if isinstance(messages, list):
            st.session_state.messages = messages

        created_at = data.get("created_at")
        if created_at:
            st.session_state.session_created_at = created_at

        updated_at = data.get("updated_at")
        if updated_at:
            st.session_state.last_saved_at = updated_at


def _save_session_history() -> None:
    session_file = _get_session_file()
    created_at = st.session_state.get("session_created_at")
    if not created_at:
        created_at = datetime.now().isoformat()
        st.session_state.session_created_at = created_at

    payload = {
        "session_id": st.session_state.session_id,
        "created_at": created_at,
        "updated_at": datetime.now().isoformat(),
        "message_count": len(st.session_state.messages),
        "messages": st.session_state.messages,
    }

    session_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    st.session_state.last_saved_at = payload["updated_at"]


def _delete_session_history() -> None:
    session_file = _get_session_file()
    if session_file.exists():
        session_file.unlink(missing_ok=True)


def _format_display_time(timestamp: str | None) -> str:
    if not timestamp:
        return "-"
    try:
        return datetime.fromisoformat(timestamp).strftime("%d %b %Y • %H:%M")
    except ValueError:
        return timestamp


st.title("ผู้ช่วยท่องเที่ยวเชียงใหม่")
st.caption("ระบบถามตอบด้วย RAG + FlashRank reranking")

class TravelAssistantRAG:
    """RAG pipeline that retrieves from Chroma, reranks, and answers with Groq."""

    def __init__(self):
        Config.validate_config()
        Config.setup_directories()

        self.client = chromadb.CloudClient(
            api_key=Config.CHROMADB_API_KEY,
            tenant=Config.CHROMADB_TENANT,
            database=Config.CHROMADB_DATABASE,
        )

        self.embedding_function = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )

        self.vector_store = Chroma(
            client=self.client,
            collection_name=Config.COLLECTIONS[0],
            embedding_function=self.embedding_function,
        )

        self.reranker = self._initialize_reranker()

        self.llm = ChatGroq(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.LLM_MAX_TOKENS,
            timeout=Config.LLM_TIMEOUT,
            max_retries=1,
            api_key=Config.GROQ_API_KEY,
        )

        self.prompt = ChatPromptTemplate.from_template(
            """คุณเป็นผู้ช่วยด้านการท่องเที่ยวเชียงใหม่ที่เชี่ยวชาญ ให้คำตอบอย่างเป็นมิตร
โดยใช้ข้อมูลจากเอกสารอ้างอิงด้านล่างเท่านั้น หากข้อมูลไม่เพียงพอให้แจ้งผู้ใช้

หลักเกณฑ์:
- ตอบเป็นภาษาไทย
- สรุปข้อมูลสำคัญ เช่น สถานที่ กิจกรรม เวลาเปิด-ปิด ค่าใช้จ่าย การเดินทาง
- แนะนำตัวเลือกที่หลากหลายเมื่อเหมาะสม
- หากข้อมูลไม่ครบถ้วน ให้ระบุข้อจำกัด

เอกสารอ้างอิง:
{context}

คำถาม: {question}

คำตอบที่เป็นประโยชน์:"""
        )

        self.output_parser = StrOutputParser()

    def _initialize_reranker(self):
        if not FLASHRANK_AVAILABLE:
            return None

        try:
            return Ranker(model_name=Config.FLASHRANK_MODEL, cache_dir="/tmp")
        except Exception:
            return None

    def _rerank_documents(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return []

        if not self.reranker:
            return docs[: Config.RETRIEVAL_K]

        passages = [
            {"id": idx, "text": doc.page_content, "meta": doc.metadata or {}}
            for idx, doc in enumerate(docs)
        ]

        try:
            response = self.reranker.rerank(
                RerankRequest(query=query, passages=passages)
            )
            ranked = sorted(response, key=lambda item: item["score"], reverse=True)
            selected_docs: List[Document] = []
            for item in ranked:
                original_doc = docs[item["id"]]
                selected_docs.append(original_doc)
                if len(selected_docs) >= Config.RETRIEVAL_K:
                    break
            return selected_docs
        except Exception:
            return docs[: Config.RETRIEVAL_K]

    def answer(self, question: str) -> tuple[str, float, List[dict]]:
        start_time = time.time()

        retrieved_docs = self.vector_store.similarity_search(
            question, k=Config.INITIAL_RETRIEVAL_K
        )
        reranked_docs = self._rerank_documents(question, retrieved_docs)

        context_text = (
            "\n\n".join(doc.page_content for doc in reranked_docs)
            or "ไม่พบข้อมูลอ้างอิง"
        )

        try:
            response = (
                self.prompt
                | self.llm
                | self.output_parser
            ).invoke({
                "context": context_text,
                "question": question,
            })
        except Exception:
            response = (
                "ขออภัย ระบบไม่สามารถสร้างคำตอบได้ในขณะนี้ "
                "กรุณาลองใหม่อีกครั้ง หรือระบุคำถามเพิ่มเติม"
            )

        elapsed = time.time() - start_time

        sources = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in reranked_docs
        ]

        return response, elapsed, sources

@st.cache_resource(show_spinner=True)
def load_rag_pipeline() -> TravelAssistantRAG:
    return TravelAssistantRAG()


def init_session_state() -> None:
    _ensure_session_id()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "session_created_at" not in st.session_state:
        st.session_state.session_created_at = None

    if "last_saved_at" not in st.session_state:
        st.session_state.last_saved_at = None

    if "history_loaded" not in st.session_state:
        _load_session_history()
        st.session_state.history_loaded = True

    if not st.session_state.session_created_at:
        st.session_state.session_created_at = datetime.now().isoformat()


init_session_state()

try:
    rag_pipeline = load_rag_pipeline()
except Exception as exc:
    st.error(
        "ไม่สามารถเริ่มต้นระบบ RAG ได้ กรุณาตรวจสอบการตั้งค่า หรือดูรายละเอียดในไฟล์ config"
    )
    st.exception(exc)
    st.stop()


with st.sidebar:
    st.subheader("ข้อมูลเซสชัน")
    st.write(f"🆔 Session: `{st.session_state.session_id[:8]}`…")
    st.write(f"� จำนวนข้อความ: {len(st.session_state.messages)}")
    st.write(
        f"🕒 สร้างเมื่อ: {_format_display_time(st.session_state.session_created_at)}"
    )
    st.write(f"💾 บันทึกล่าสุด: {_format_display_time(st.session_state.last_saved_at)}")

    st.markdown("---")
    st.subheader("ตั้งค่า")

    if st.button("ล้างประวัติการสนทนา"):
        st.session_state.messages = []
        st.session_state.session_created_at = None
        _delete_session_history()
        st.rerun()


for message in st.session_state.messages:
    role = message.get("role", "assistant")
    with st.chat_message(role):
        st.markdown(message.get("content", ""))
        if role == "assistant" and message.get("generation_time") is not None:
            st.caption(
                f"⏱️ เวลาในการสร้างคำตอบ: {message['generation_time']:.2f} วินาที"
            )
        if role == "assistant" and message.get("sources"):
            with st.expander("ดูแหล่งข้อมูลอ้างอิง"):
                for idx, source in enumerate(message["sources"], start=1):
                    st.markdown(f"**เอกสาร {idx}**")
                    metadata = source.get("metadata") or {}
                    for key, value in metadata.items():
                        st.markdown(f"- {key}: {value}")
                    st.markdown(f"> {source.get('content', '')}")


if prompt := st.chat_input("พิมพ์คำถามเกี่ยวกับการท่องเที่ยวเชียงใหม่ได้เลย..."):
    user_message = {
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().isoformat(),
    }
    st.session_state.messages.append(user_message)
    _save_session_history()

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("กำลังสืบค้นข้อมูลและสร้างคำตอบ..."):
            answer, elapsed, sources = rag_pipeline.answer(prompt)
        placeholder.markdown(answer)
        st.caption(f"⏱️ เวลาในการสร้างคำตอบ: {elapsed:.2f} วินาที")
        if sources:
            with st.expander("ดูแหล่งข้อมูลอ้างอิง"):
                for idx, source in enumerate(sources, start=1):
                    st.markdown(f"**เอกสาร {idx}**")
                    metadata = source.get("metadata") or {}
                    for key, value in metadata.items():
                        st.markdown(f"- {key}: {value}")
                    st.markdown(f"> {source.get('content', '')}")

    assistant_message = {
        "role": "assistant",
        "content": answer,
        "generation_time": elapsed,
        "sources": sources,
        "timestamp": datetime.now().isoformat(),
    }
    st.session_state.messages.append(assistant_message)
    _save_session_history()

    st.rerun()
