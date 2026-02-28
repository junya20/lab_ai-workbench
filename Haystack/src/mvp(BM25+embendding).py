from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever, InMemoryBM25Retriever
from haystack.components.joiners import DocumentJoiner
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret


def main():
    # 0) DocumentStore
    docstore = InMemoryDocumentStore(embedding_similarity_function="cosine")

    # 1) Documents
    docs = [
        Document(content="The boy's name is Taro"),
        Document(content="Taro is 15 years old."),
        Document(content="The girl's name is Hanako"),
        Document(content="Hanako is 23 years old.")
    ]

    # 2) Document embeddings を作って保存
    doc_embedder = OpenAITextEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model="text-embedding-3-small",
    )
    for d in docs:
        d.embedding = doc_embedder.run(text=d.content)["embedding"]
    docstore.write_documents(docs)

    # 3) Query embedder
    query_embedder = OpenAITextEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model="text-embedding-3-small",
    )

    # 4) Two retrievers (Sparse + Dense)
    bm25_retriever = InMemoryBM25Retriever(document_store=docstore, top_k=3)
    embedding_retriever = InMemoryEmbeddingRetriever(document_store=docstore, top_k=3)

    # 5) Join results (Hybrid)
    # - concatenate: 単純結合（重複排除）
    # - merge / reciprocal_rank_fusion なども選べる
    joiner = DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=3)

    # 6) Prompt + LLM
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

    # 7) Build pipeline (two-branch retrieval → join → prompt → LLM)
    pipe = Pipeline()
    pipe.add_component("query_embedder", query_embedder)

    pipe.add_component("bm25_retriever", bm25_retriever)
    pipe.add_component("embedding_retriever", embedding_retriever)

    pipe.add_component("joiner", joiner)
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("generator", generator)

    # Branch A: BM25 needs raw query
    pipe.connect("bm25_retriever.documents", "joiner.documents")

    # Branch B: Embedding retriever needs query_embedding
    pipe.connect("query_embedder.embedding", "embedding_retriever.query_embedding")
    pipe.connect("embedding_retriever.documents", "joiner.documents")

    # After join
    pipe.connect("joiner.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "generator.prompt")

    # 8) Run
    question = "How old is Taro?"
    result = pipe.run(
        {
            "query_embedder": {"text": question},
            "bm25_retriever": {"query": question},
            "embedding_retriever": {},   # query_embedding は接続済み
            "joiner": {},               # 追加入力なし
            "prompt_builder": {"query": question},
        }
    )
    print(result["generator"]["replies"][0])


if __name__ == "__main__":
    main()
