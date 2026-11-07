"""Conversation storage, API, and document-isolation tests."""

from pathlib import Path
import tempfile
import unittest
import os
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app.api.main import app, authenticated_user
from app.core.auth import require_streamlit_user
from app.services.conversation_service import (
    DEFAULT_CONVERSATION_ID,
    ConversationService,
)
from app.vectorstore.chroma_store import ChromaVectorStore


app.dependency_overrides[authenticated_user] = lambda: "user_test"


class FixedEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class ConversationTests(unittest.TestCase):
    def test_conversations_are_owned_by_their_creator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            conversations = ConversationService(Path(directory) / "sessions.sqlite3")
            private = conversations.create("user-a", "Private notes")

            self.assertIsNotNone(conversations.get(private["id"], "user-a"))
            self.assertIsNone(conversations.get(private["id"], "user-b"))
            self.assertEqual(
                [item["id"] for item in conversations.list_conversations("user-a")],
                [private["id"]],
            )
            self.assertEqual(conversations.list_conversations("user-b"), [])

    def test_internal_api_auth_checks_the_shared_secret(self) -> None:
        with patch.dict(os.environ, {"SMARTDOCS_INTERNAL_AUTH_TOKEN": "test-secret"}):
            self.assertEqual(
                require_streamlit_user("test-secret", "clerk-user-1"),
                "clerk-user-1",
            )
            with self.assertRaises(Exception) as error:
                require_streamlit_user("wrong-secret", "clerk-user-1")
            self.assertEqual(error.exception.status_code, 401)

    def test_history_persists_and_is_scoped_by_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "conversations.sqlite3"
            conversations = ConversationService(database)
            first = conversations.create("user-a", "  Policy questions  ")
            second = conversations.create("user-a", "Engineering notes")
            conversations.add_message(first["id"], "user", "Question one")
            conversations.add_message(
                first["id"], "assistant", "Answer one", [{"source_id": "S1"}]
            )
            reopened = ConversationService(database)

            self.assertEqual(first["title"], "Policy questions")
            self.assertEqual(len(reopened.messages(first["id"])), 2)
            self.assertEqual(reopened.messages(second["id"]), [])
            self.assertEqual(reopened.messages(first["id"])[1]["sources"], [
                {"source_id": "S1"}
            ])

    def test_chroma_search_and_document_list_are_conversation_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ChromaVectorStore(directory, embedder=FixedEmbedder())  # type: ignore[arg-type]
            chunk = {
                "text": "Annual leave is 25 days.",
                "filename": "policy.txt",
                "page_number": None,
                "chunk_index": 1,
            }
            store.add_document("doc-a", [chunk], conversation_id="conversation-a")
            store.add_document("doc-b", [chunk], conversation_id="conversation-b")

            results = store.search(
                "How much leave?", conversation_id="conversation-a"
            )
            scoped_documents = store.list_documents("conversation-b")
            wrong_owner_delete = store.delete_document("doc-b", "conversation-a")
            store.close()

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["metadata"]["document_id"], "doc-a")
            self.assertEqual(
                [doc["document_id"] for doc in scoped_documents],
                ["doc-b"],
            )
            self.assertEqual(wrong_owner_delete, 0)

    def test_legacy_chunks_are_assigned_to_default_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ChromaVectorStore(directory, embedder=FixedEmbedder())  # type: ignore[arg-type]
            store.add_document(
                "legacy",
                [{
                    "text": "legacy content",
                    "filename": "old.txt",
                    "page_number": None,
                    "chunk_index": 1,
                }],
                conversation_id="temporary-owner",
            )
            record = store.collection.get(
                where={"document_id": "legacy"},
                include=["metadatas", "documents", "embeddings"],
            )
            metadata = dict(record["metadatas"][0])
            metadata.pop("conversation_id")
            store.collection.delete(ids=record["ids"])
            store.collection.add(
                ids=record["ids"],
                documents=record["documents"],
                embeddings=record["embeddings"],
                metadatas=[metadata],
            )

            migrated = store.assign_legacy_chunks(DEFAULT_CONVERSATION_ID)
            listed = store.list_documents(DEFAULT_CONVERSATION_ID)
            store.close()
            self.assertEqual(migrated, 1)
            self.assertEqual(listed[0]["document_id"], "legacy")

    def test_conversation_api_creates_reads_and_deletes_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            conversations = ConversationService(Path(directory) / "sessions.sqlite3")
            conversations.claim_unowned_conversations("user_test")
            documents = Mock()
            documents.delete_conversation.return_value = 2
            client = TestClient(app)
            with (
                patch("app.api.main.get_conversation_service", return_value=conversations),
                patch("app.api.main.get_document_service", return_value=documents),
            ):
                created = client.post(
                    "/api/conversations", json={"title": "Project notes"}
                )
                conversation_id = created.json()["id"]
                conversations.add_message(conversation_id, "user", "Remember this")
                history = client.get(
                    f"/api/conversations/{conversation_id}/messages"
                )
                deleted = client.delete(f"/api/conversations/{conversation_id}")

            self.assertEqual(created.status_code, 201)
            self.assertEqual(history.json()[0]["content"], "Remember this")
            self.assertEqual(deleted.json()["chunks_deleted"], 2)
            self.assertIsNone(conversations.get(conversation_id))

    def test_chat_uses_and_persists_the_selected_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            conversations = ConversationService(Path(directory) / "sessions.sqlite3")
            selected = conversations.create("user_test", "Selected workspace")
            rag = Mock()
            rag.answer.return_value = {
                "answer": "Grounded response",
                "sources": [{"source_id": "S1", "filename": "guide.txt"}],
                "retrieved_count": 1,
                "used_llm": True,
            }
            client = TestClient(app)
            with (
                patch("app.api.main.get_conversation_service", return_value=conversations),
                patch("app.api.main.get_rag_service", return_value=rag),
            ):
                response = client.post(
                    "/api/chat",
                    json={
                        "question": "What does the guide say?",
                        "conversation_id": selected["id"],
                    },
                )
                history = conversations.messages(selected["id"])

            self.assertEqual(response.status_code, 200)
            rag.answer.assert_called_once_with(
                "What does the guide say?",
                top_k=4,
                conversation_id=selected["id"],
                owner_user_id="user_test",
            )
            self.assertEqual([message["role"] for message in history], ["user", "assistant"])
            self.assertEqual(history[1]["sources"][0]["source_id"], "S1")


if __name__ == "__main__":
    unittest.main()
