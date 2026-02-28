from langfuse import Langfuse
from openai import OpenAI
import json
import time

langfuse = Langfuse()
client = OpenAI()

document_id = "001"
ground_truth = {"name": "山田太郎", "age": 35}
prompt_versions = {
    "v1": "次のテキストから name と age をJSONで抽出して。JSONのみ返して。",
    "v2": "以下の情報から name(文字列) と age(整数) を必ずJSON形式で返してください。余計な文章は禁止。"
}

print("START")

for version, system_prompt in prompt_versions.items():

    print(f"Running {version}")

    with langfuse.start_as_current_span(name="document") as trace:

        trace.update(metadata={
            "document_id": document_id,
            "prompt_version": version
        })

        with langfuse.start_as_current_span(name="preprocess") as span1:
            time.sleep(0.2)
            extracted_text = "氏名: 山田太郎 年齢: 35"
            span1.update(output=extracted_text)

        with langfuse.start_as_current_span(name="llm-extraction") as span2:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": extracted_text},
                ],
            )

            output_text = response.choices[0].message.content

            span2.update(
                input=extracted_text,
                output=output_text,
                metadata={
                    "prompt_version": version,
                    "model": "gpt-4o-mini"
                }
            )

        with langfuse.start_as_current_span(name="validation") as span3:
            try:
                parsed = json.loads(output_text)
                is_valid = parsed == ground_truth
            except Exception as e:
                is_valid = False
                parsed = {"_parse_error": str(e), "_raw": output_text}

            span3.update(
                output=json.dumps({
                    "is_valid": is_valid,
                    "parsed": parsed,
                    "ground_truth": ground_truth
                }, ensure_ascii=False)
            )

            print(f"{version} valid?: {is_valid}")

print("DONE")