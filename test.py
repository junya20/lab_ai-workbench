import os
from litellm import completion

# どのみち環境変数でOK。念のためコードでも設定したいなら:
# os.environ["OPENAI_API_KEY"] = "sk-..."

resp = completion(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "LiteLLMって何？1文で。"}],
)

print(resp.choices[0].message["content"])
