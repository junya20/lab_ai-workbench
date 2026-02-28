from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret


def debug_embedding_sanity_check(docs, doc_embedder, query_embedder, retriever):
    """
    Embeddingが正しくできているかの簡易テスト。
    1) embeddingベクトルの次元・中身
    2) docsにembeddingが入っているか
    3) 検索結果が期待通りか（正例/負例）
    """
    print("\n=== [TEST] Embedding sanity check ===")

    # 1) ベクトル生成の形チェック
    e = doc_embedder.run(text="hello")["embedding"]
    print(f"[1] embed type={type(e).__name__}, dim={len(e)}, head={e[:5]}")

    # 2) docsにembeddingが入っているか
    for i, d in enumerate(docs):
        assert d.embedding is not None, f"[2] docs[{i}] embedding is None"
        assert len(d.embedding) > 0, f"[2] docs[{i}] embedding is empty"
    print("[2] OK: all docs have embeddings")

    # 3) Retrieverが意味的にそれっぽいものを返すか
    def show_retrieve(query: str):
        q_emb = query_embedder.run(text=query)["embedding"]
        res = retriever.run(query_embedding=q_emb)
        print(f"\n[3] Query: {query}")
        for rank, doc in enumerate(res["documents"], 1):
            # scoreはDocumentに付与されます
            print(f"  {rank}. score={getattr(doc, 'score', None):.4f} | {doc.content}")

    # 正例：Taroの年齢 → "Taro is 15 years old." が上に来てほしい
    show_retrieve("How old is Taro?")

    # 正例：Hanakoの年齢
    show_retrieve("How old is Hanako?")

    # 負例：関係ない質問（トップが変わる/スコアが下がるのを期待）
    show_retrieve("What is the capital of France?")

    print("\n=== [TEST] Done ===\n")


def main():
    # 0) DocumentStore（embedding検索をするので embedding_similarity_function を指定）
    docstore = InMemoryDocumentStore(embedding_similarity_function="cosine")

    # 1) 文書（Document）を用意
    docs = [
        Document(content="The boy's name is Taro"),
        Document(content="Taro is 15 years old."),
        Document(content="The girl's name is Hanako"),
        Document(content="Hanako is 23 years old.")
    ]

    # 2) 文書 embedding を作って保存（最小なのでループでOK）
    doc_embedder = OpenAITextEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model="text-embedding-3-small",
    )
    for d in docs:
        d.embedding = doc_embedder.run(text=d.content)["embedding"]
    docstore.write_documents(docs)

    # 3) クエリ embedding → embedding検索
    query_embedder = OpenAITextEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model="text-embedding-3-small",
    )
    retriever = InMemoryEmbeddingRetriever(document_store=docstore, top_k=2)

    # ✅ 追加：Embeddingテスト（生成前にここで止めても確認できます）
    debug_embedding_sanity_check(docs, doc_embedder, query_embedder, retriever)

    # 4) 取得文書をプロンプトに詰めて回答生成
    template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
- {{ document.content }}
{% endfor %}

Question: {{ query }}
Answer:
""".strip()

    prompt_builder = PromptBuilder(
        template=template,
        required_variables=["documents", "query"],
    )

    generator = OpenAIGenerator(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )

    # 5) パイプライン構築
    pipe = Pipeline()
    pipe.add_component("query_embedder", query_embedder)
    pipe.add_component("retriever", retriever)
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("generator", generator)

    pipe.connect("query_embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "generator.prompt")

    # 6) 実行
    question = "How old is boy?"
    result = pipe.run(
        {
            "query_embedder": {"text": question},
            "retriever": {},
            "prompt_builder": {"query": question},
        }
    )
    print(result["generator"]["replies"][0])


if __name__ == "__main__":
    main()
