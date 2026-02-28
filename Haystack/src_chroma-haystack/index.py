# index.py
from haystack import Pipeline, Document
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.utils import Secret

from haystack_integrations.document_stores.chroma import ChromaDocumentStore

def main():
    # 1) “ローカルDB” を用意（persist_path に永続化される）
    document_store = ChromaDocumentStore(
        persist_path="chroma_db",
        collection_name="demo_collection",  # わかりやすく固定
    )

    # 2) DBに入れる文書（適当に“知識ベース”を作る）
    docs = [
        Document(content="Taro is a high school student. He is 15 years old.", meta={"source": "kb", "topic": "profile"}),
        Document(content="Hanako is a graduate student. She is 23 years old.", meta={"source": "kb", "topic": "profile"}),
        Document(content="Project Alpha uses Graph RAG for document QA.", meta={"source": "kb", "topic": "project"}),
        Document(content="Chroma can persist data on disk by setting persist_path.", meta={"source": "kb", "topic": "chroma"}),
        Document(content="In RAG, retrieval quality strongly affects answer quality.", meta={"source": "kb", "topic": "rag"}),
    ]

    # 3) インデックス作成パイプライン：documents → embed → write
    doc_embedder = OpenAIDocumentEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model="text-embedding-3-small",
    )
    writer = DocumentWriter(document_store=document_store)

    indexing = Pipeline()
    indexing.add_component("embedder", doc_embedder)
    indexing.add_component("writer", writer)
    indexing.connect("embedder.documents", "writer.documents")

    result = indexing.run({"embedder": {"documents": docs}})

    print("Indexed documents:", document_store.count_documents())
    print("Done. Persisted to ./chroma_db")

if __name__ == "__main__":
    main()
