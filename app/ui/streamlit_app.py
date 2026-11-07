"""Streamlit user interface for the SmartDocs API."""

import os

import requests
import streamlit as st


API_URL = os.getenv("SMARTDOCS_API_URL", "http://127.0.0.1:8000").rstrip("/")
SUPPORTED_TYPES = ["pdf", "txt", "md"]

st.set_page_config(page_title="SmartDocs", page_icon="📚", layout="wide")
st.title("SmartDocs")
st.caption("Ask questions and get answers grounded in your documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def api_error(response: requests.Response) -> str:
    """Extract FastAPI's detail text when possible."""
    try:
        payload = response.json()
        return str(payload.get("detail", payload))
    except ValueError:
        return response.text or f"Request failed ({response.status_code})."


with st.sidebar:
    st.header("Documents")
    with st.form("upload_form", clear_on_submit=True):
        upload = st.file_uploader("Upload a document", type=SUPPORTED_TYPES)
        upload_clicked = st.form_submit_button("Upload and index")

    if upload_clicked and upload is not None:
        try:
            response = requests.post(
                f"{API_URL}/api/documents/upload",
                files={
                    "file": (
                        upload.name,
                        upload.getvalue(),
                        upload.type or "application/octet-stream",
                    )
                },
                timeout=300,
            )
            if response.ok:
                result = response.json()
                if result.get("duplicate"):
                    st.info(f"{result['filename']} is already indexed.")
                else:
                    st.success(
                        f"Indexed {result['filename']} ({result['chunks_stored']} chunks)."
                    )
            else:
                st.error(api_error(response))
        except requests.RequestException as error:
            st.error(f"Could not reach the SmartDocs API: {error}")

    st.subheader("Indexed documents")
    try:
        response = requests.get(f"{API_URL}/api/documents", timeout=30)
        response.raise_for_status()
        documents = response.json()
        if not documents:
            st.caption("No documents indexed yet.")
        for document in documents:
            left, right = st.columns([4, 1])
            with left:
                st.write(document["filename"])
                st.caption(f"{document['chunks_stored']} chunks")
            with right:
                if st.button("Delete", key=f"delete-{document['document_id']}"):
                    deleted = requests.delete(
                        f"{API_URL}/api/documents/{document['document_id']}",
                        timeout=30,
                    )
                    if deleted.ok:
                        st.success(f"Deleted {document['filename']}.")
                        st.rerun()
                    else:
                        st.error(api_error(deleted))
    except requests.RequestException as error:
        st.error(f"Could not load documents: {error}")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    page = source.get("page_number")
                    page_text = f", page {page}" if page is not None else ""
                    st.markdown(
                        f"**[{source['source_id']}] {source['filename']}**"
                        f"{page_text} · similarity {source['similarity']:.3f}"
                    )

question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents and preparing an answer..."):
            try:
                response = requests.post(
                    f"{API_URL}/api/chat",
                    json={"question": question},
                    timeout=180,
                )
                if response.ok:
                    result = response.json()
                    answer = result["answer"]
                    sources = result.get("sources", [])
                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            for source in sources:
                                page = source.get("page_number")
                                page_text = f", page {page}" if page is not None else ""
                                st.markdown(
                                    f"**[{source['source_id']}] {source['filename']}**"
                                    f"{page_text} · similarity {source['similarity']:.3f}"
                                )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                else:
                    detail = api_error(response)
                    st.error(detail)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Request failed: {detail}"}
                    )
            except requests.RequestException as error:
                detail = f"Could not reach the SmartDocs API: {error}"
                st.error(detail)
                st.session_state.messages.append(
                    {"role": "assistant", "content": detail}
                )
