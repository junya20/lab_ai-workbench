from langfuse import Langfuse
from openai import OpenAI

langfuse = Langfuse()
client = OpenAI()

with langfuse.start_as_current_span(name="gpt-test") as span:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "日本の首都は？"}],
    )

    span.update(
        input="日本の首都は？",
        output=response.choices[0].message.content
    )

print(response.choices[0].message.content)