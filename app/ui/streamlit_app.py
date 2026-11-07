"""Streamlit interface for conversation-scoped SmartDocs workspaces."""

import os

import requests
import streamlit as st


API_URL = os.getenv("SMARTDOCS_API_URL", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_CONVERSATION_ID = "default"
SUPPORTED_TYPES = ["pdf", "txt", "md"]

st.set_page_config(page_title="SmartDocs", page_icon="📚", layout="wide")
st.title("SmartDocs")
st.caption("Each conversation has its own documents and saved chat history.")


def api_error(response: requests.Response) -> str:
    """Extract FastAPI's detail text when possible."""
    try:
        payload = response.json()
        return str(payload.get("detail", payload))
    except ValueError:
        return response.text or f"Request failed ({response.status_code})."


def show_sources(sources: list[dict[str, object]]) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        for source in sources:
            page = source.get("page_number")
            page_text = f", page {page}" if page is not None else ""
            similarity = float(source.get("similarity", 0.0))
            st.markdown(
                f"**[{source['source_id']}] {source['filename']}**"
                f"{page_text} · similarity {similarity:.3f}"
            )


with st.sidebar:
    st.header("Conversations")
    try:
        conversations_response = requests.get(
            f"{API_URL}/api/conversations", timeout=30
        )
        conversations_response.raise_for_status()
        conversations = conversations_response.json()
    except requests.RequestException as error:
        st.error(f"Could not load conversations: {error}")
        conversations = []

    with st.form("new_conversation_form", clear_on_submit=True):
        new_title = st.text_input("New conversation name", value="New conversation")
        create_clicked = st.form_submit_button("＋ New conversation")
    if create_clicked:
        try:
            created = requests.post(
                f"{API_URL}/api/conversations",
                json={"title": new_title},
                timeout=30,
            )
            if created.ok:
                st.session_state["conversation_id"] = created.json()["id"]
                st.rerun()
            st.error(api_error(created))
        except requests.RequestException as error:
            st.error(f"Could not create conversation: {error}")

    conversation_ids = [item["id"] for item in conversations]
    if not conversation_ids:
        st.info("Create a conversation to get started.")
        st.stop()

    current_id = st.session_state.get("conversation_id")
    if current_id not in conversation_ids:
        current_id = conversation_ids[0]
    current_index = conversation_ids.index(current_id)
    selector_key = "conversation_selector_" + "_".join(conversation_ids)
    selected_id = st.selectbox(
        "Active conversation",
        conversation_ids,
        index=current_index,
        format_func=lambda value: next(
            item["title"] for item in conversations if item["id"] == value
        ),
        key=selector_key,
    )
    st.session_state["conversation_id"] = selected_id

    confirm_conversation_delete = False
    if selected_id != DEFAULT_CONVERSATION_ID:
        confirm_conversation_delete = st.checkbox(
            "Confirm deleting this conversation and its files",
            key=f"confirm-delete-{selected_id}",
        )
    if confirm_conversation_delete and st.button(
        "Delete conversation", type="secondary", key=f"delete-conversation-{selected_id}"
    ):
        try:
            deleted = requests.delete(
                f"{API_URL}/api/conversations/{selected_id}", timeout=30
            )
            if deleted.ok:
                st.session_state.pop("conversation_id", None)
                st.rerun()
            st.error(api_error(deleted))
        except requests.RequestException as error:
            st.error(f"Could not delete conversation: {error}")

    st.divider()
    st.header("Documents")
    with st.form("upload_form", clear_on_submit=True):
        upload = st.file_uploader("Upload a document", type=SUPPORTED_TYPES)
        upload_clicked = st.form_submit_button("Upload and index")

    if upload_clicked and upload is not None:
        try:
            response = requests.post(
                f"{API_URL}/api/documents/upload",
                data={"conversation_id": selected_id},
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
                    st.info(f"{result['filename']} is already in this conversation.")
                else:
                    st.success(
                        f"Indexed {result['filename']} ({result['chunks_stored']} chunks)."
                    )
            else:
                st.error(api_error(response))
        except requests.RequestException as error:
            st.error(f"Could not upload the document: {error}")

    st.subheader("Files in this conversation")
    try:
        response = requests.get(
            f"{API_URL}/api/documents",
            params={"conversation_id": selected_id},
            timeout=30,
        )
        response.raise_for_status()
        documents = response.json()
        if not documents:
            st.caption("No documents indexed in this conversation yet.")
        for document in documents:
            left, right = st.columns([4, 1])
            with left:
                st.write(document["filename"])
                st.caption(f"{document['chunks_stored']} chunks")
            with right:
                if st.button("Delete", key=f"delete-{document['document_id']}"):
                    deleted = requests.delete(
                        f"{API_URL}/api/documents/{document['document_id']}",
                        params={"conversation_id": selected_id},
                        timeout=30,
                    )
                    if deleted.ok:
                        st.rerun()
                    st.error(api_error(deleted))
    except requests.RequestException as error:
        st.error(f"Could not load documents: {error}")

try:
    history_response = requests.get(
        f"{API_URL}/api/conversations/{selected_id}/messages", timeout=30
    )
    history_response.raise_for_status()
    messages = history_response.json()
except requests.RequestException as error:
    st.error(f"Could not load conversation history: {error}")
    messages = []

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("sources", []))

question = st.chat_input("Ask a question about this conversation's documents")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching this conversation's documents..."):
            try:
                response = requests.post(
                    f"{API_URL}/api/chat",
                    json={"question": question, "conversation_id": selected_id},
                    timeout=180,
                )
                if response.ok:
                    st.rerun()
                st.error(api_error(response))
            except requests.RequestException as error:
                st.error(f"Could not reach the SmartDocs API: {error}")
