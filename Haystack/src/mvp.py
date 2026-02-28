from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

def main():
    # 1) 文書を用意（最小のインメモリDocumentStore）
    docstore = InMemoryDocumentStore()
    docstore.write_documents([
        Document(content="The boy's name is taro"),
        Document(content="taro is 20 years old"),
        Document(content="The girl's name is hana"),
        Document(content="hana is 20 years old")
    ])

    # 2) キーワード検索（BM25）
    retriever = InMemoryBM25Retriever(document_store=docstore, top_k=2)

    # 3) 取得した文書をプロンプトに埋め込むテンプレート
    template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
- {{ document.content }}
{% endfor %}

Question: {{ query }}
Answer:
""".strip()
    prompt_builder = PromptBuilder(template=template, required_variables=["documents", "query"])

    # 4) LLM（OpenAI）
    generator = OpenAIGenerator(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model="gpt-4o-mini"  # ここは使いたいモデル名に変えてOK
    )

    # 5) パイプラインを接続
    pipe = Pipeline()
    pipe.add_component("retriever", retriever)
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("generator", generator)

    pipe.connect("retriever.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "generator.prompt")

    # 6) 実行
    question = "How old is taro?"
    result = pipe.run({
        "retriever": {"query": question},
        "prompt_builder": {"query": question},
    })

    # OpenAIGeneratorの出力（返り値の形式はバージョンで多少変わることがあります）
    print(result["generator"]["replies"][0])

if __name__ == "__main__":
    main()
