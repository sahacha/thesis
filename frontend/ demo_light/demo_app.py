import streamlit as st
from datetime import datetime

st.set_page_config(page_title="STORM", layout="wide")

# ---------- Helper: mock function ----------
def create_article(topic: str):
    st.session_state["sessions"].append(
        {
            "title": f"Draft: {topic}",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    )

# ---------- Session state ----------
if "sessions" not in st.session_state:
    st.session_state["sessions"] = []

# ---------- Layout ----------
sidebar_left, main, sidebar_right = st.columns([1, 3, 1])

# ---------- Left Sidebar ----------
with sidebar_left:
    st.title("STORM")
    st.markdown("---")
    topic = st.text_input("Enter the topic (English only)", placeholder="e.g. Behavioral Finance")
    if st.button("Create an Article"):
        if topic:
            create_article(topic)
            st.success(f"Session “{topic}” added!")
    st.markdown("---")
    st.markdown("### @ GitHub")
    st.markdown("### @ arXiv")

# ---------- Main Panel ----------
with main:
    st.markdown("## My Sessions")
    if st.button("➕ New Session", use_container_width=True):
        create_article("Untitled Topic")

    st.markdown("---")
    for idx, session in enumerate(reversed(st.session_state["sessions"])):
        with st.expander(f"{session['title']} — {session['created_at']}", expanded=False):
            st.write("Content goes here …")
            if st.button(f"Delete #{idx}", key=f"del_{idx}"):
                st.session_state["sessions"].pop(len(st.session_state["sessions"]) - 1 - idx)
                st.rerun()

# ---------- Right Sidebar ----------
with sidebar_right:
    st.markdown("### Discover")
    st.markdown("#### Behavioral Finance")
    st.markdown("An interdisciplinary field …")
    st.markdown("#### Generative AI applied to HR")
    st.markdown("A revolutionary shift …")
    st.markdown("---")
    st.markdown("#### Contact Us")
    st.markdown("#### Bug Report")