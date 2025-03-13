import os
import json

if __name__ == '__main__':
    json_path = "./saves/LLaMA3.1-8B_v1-1/lora/predict_2025-03-07-14-55-12"
    with open(os.path.join(json_path, "generated_predictions.jsonl"), "r", encoding="utf-8") as f:
        generated_predictions = [json.loads(line) for line in f]
        print(1)

    generated_predictions_path = "generated_predictions"
    if not os.path.exists(os.path.join(json_path, generated_predictions_path)):
        os.makedirs(os.path.join(json_path, generated_predictions_path))
    for idx, prediction in enumerate(generated_predictions):
        for key in prediction.keys():
            md_content = prediction[key]
            md_filename = os.path.join(json_path, generated_predictions_path, f"{idx:05d}_{key}.md")
            with open(md_filename, "w", encoding="utf-8") as md_file:
                md_file.write(md_content)
            print(f"Saved: {md_filename}")