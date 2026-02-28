import json
from datetime import datetime
from litellm import completion
from pydantic import BaseModel, ValidationError

# ======================
# Pydantic Schema
# ======================
class Person(BaseModel):
    name: str
    age: int

# ======================
# LLM 呼び出し
# ======================
response = completion(
    model="gpt-5-mini",
    messages=[
        {
            "role": "system",
            "content": "Output ONLY valid JSON. No text, no markdown."
        },
        {
            "role": "user",
            "content": "Return JSON with fields name (string) and age (number). Name=ken, Age=100."
        }
    ],
)

raw = response["choices"][0]["message"]["content"]
print("=== RAW OUTPUT ===")
print(raw)

# ======================
# JSON + Schema 検証
# ======================
print("\n=== VALIDATION RESULT ===")
try:
    data = json.loads(raw)          # JSONとしてパースできるか
    person = Person(**data)         # Schemaに合うか（ここで検証）

    # 保存するデータ（Pydanticで正規化された値を使う）
    out_dict = person.model_dump()  # pydantic v2
    # pydantic v1 の場合は: out_dict = person.dict()

    # ファイル名（タイムスタンプ付き）
    out_path = f"person_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=2)

    print("✅ VALID JSON")
    print(person)
    print(f"💾 Saved to: {out_path}")

except json.JSONDecodeError as e:
    print("❌ JSON decode failed")
    print(e)
except ValidationError as e:
    print("❌ Schema validation failed")
    print(e)