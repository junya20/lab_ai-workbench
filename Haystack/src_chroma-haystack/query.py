# query.py
from haystack import Pipeline
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever

def main():
    # 1) index.py と同じ設定で “同じDB” を開く
    document_store = ChromaDocumentStore(
        persist_path="chroma_db",
        collection_name="demo_collection",
    )

    # 2) Query → embedding → retrieve
    query_embedder = OpenAITextEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model="text-embedding-3-small",
    )
    retriever = ChromaEmbeddingRetriever(document_store=document_store, top_k=3)

    # 3) Prompt → LLM
    template = """
You are a QA assistant.
Answer ONLY using the provided Context.
If the answer is not in the Context, say: "I don't know based on the context."

Context:
{% for document in documents %}
- {{ document.content }} (meta={{ document.meta }})
{% endfor %}

Question: {{ query }}
Answer:
""".strip()

    prompt_builder = PromptBuilder(template=template, required_variables=["documents", "query"])

    generator = OpenAIGenerator(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )

    pipe = Pipeline()
    pipe.add_component("query_embedder", query_embedder)
    pipe.add_component("retriever", retriever)
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("generator", generator)

    pipe.connect("query_embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "generator.prompt")

    question = "How old is Taro?"
    # --- ここで “DBから取れた文書” を確実に表示する（Pipelineの戻り値に頼らない） ---
    q_emb = query_embedder.run(text=question)["embedding"]
    retrieved = retriever.run(query_embedding=q_emb)
    docs = retrieved["documents"]

    print("\n=== Retrieved docs (direct call) ===")
    for i, d in enumerate(docs, 1):
        score = getattr(d, "score", None)
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
        print(f"{i}. score={score_str} | {d.content} | meta={d.meta}")

    # --- ここからは今まで通り Pipeline を実行して回答生成 ---
    result = pipe.run(
        {
            "query_embedder": {"text": question},
            "prompt_builder": {"query": question},
        }
    )

    print("\n=== Answer ===")
    print(result["generator"]["replies"][0])
if __name__ == "__main__":
    main()
