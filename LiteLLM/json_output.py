import json
from litellm import completion

response = completion(
    model="gpt-5-mini",
    messages=[
        {
            "role": "system",
            "content": "Output ONLY valid JSON. No text, no markdown."
        },
        {
            "role": "user",
            "content": "Return JSON with fields name (string) and age (number). Name=Junya, Age=25."
        }
    ],
    temperature=0,
)

# LLMの生出力
content = response["choices"][0]["message"]["content"]
print("RAW:", content)

# JSONパース確認
try:
    data = json.loads(content)
    print("✅ JSON parse success:", data)
except json.JSONDecodeError as e:
    print("❌ JSON parse failed:", e)
