from langfuse import Langfuse
from openai import OpenAI
import json
import time

langfuse = Langfuse()
client = OpenAI()

document_id = "001"
ground_truth = {"name": "山田太郎", "age": 35}

print("START")  # 追加：開始ログ

with langfuse.start_as_current_span(name="document") as trace:

    with langfuse.start_as_current_span(name="preprocess") as span1:
        time.sleep(0.2)
        extracted_text = "氏名: 山田太郎 年齢: 35"
        span1.update(output=extracted_text)

    with langfuse.start_as_current_span(name="llm-extraction") as span2:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "次のテキストから name と age をJSONで抽出して。JSONのみ返して。"},
                {"role": "user", "content": extracted_text},
            ],
        )
        output_text = response.choices[0].message.content
        span2.update(input=extracted_text, output=output_text)

    with langfuse.start_as_current_span(name="validation") as span3:
        try:
            parsed = json.loads(output_text)
            is_valid = parsed == ground_truth
        except Exception as e:
            is_valid = False
            parsed = {"_parse_error": str(e), "_raw": output_text}

        span3.update(output=json.dumps({"is_valid": is_valid, "parsed": parsed, "ground_truth": ground_truth}, ensure_ascii=False))

print("DONE")  # 追加：終了ログ
print("LLM output:", output_text)  # 追加：LLM出力を表示