import json
import yaml
from pathlib import Path

INPUT_JSON = "data/intent.json"
OUTPUT_NLU = "data/preprocessed_nlu.yml"
OUTPUT_DOMAIN = "rasa_project/domain.yml"
OUTPUT_STORIES = "data/stories.yml"

def preprocess_text(text):
    return text.strip().lower()

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    nlu_data = {"version": "3.1", "nlu": []}
    domain_data = {"version": "3.1", "intents": [], "responses": {}}
    stories_data = {"version": "3.1", "stories": []}

    for intent_obj in data.get("intents", []):
        intent = intent_obj.get("intent")
        texts = intent_obj.get("text", [])
        responses = intent_obj.get("responses", [])

        # Add to NLU
        if texts:
            examples = "\n".join(f"- {preprocess_text(t)}" for t in texts)
            nlu_data["nlu"].append({"intent": intent, "examples": examples})

        # Add to domain
        domain_data["intents"].append(intent)
        if responses:
            domain_data["responses"][f"utter_{intent}"] = [{"text": r} for r in responses]

        # Add to stories
        if responses:
            stories_data["stories"].append({
                "story": f"{intent} path",
                "steps": [
                    {"intent": intent},
                    {"action": f"utter_{intent}"}
                ]
            })

    # Write NLU
    Path(OUTPUT_NLU).write_text(
        yaml.dump(nlu_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8"
    )

    # Write Domain
    Path(OUTPUT_DOMAIN).write_text(
        yaml.dump(domain_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8"
    )

    # Write Stories
    Path(OUTPUT_STORIES).write_text(
        yaml.dump(stories_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8"
    )

    print(f"✅ Generated {len(nlu_data['nlu'])} intents")
    print(f"➡️ {OUTPUT_NLU}")
    print(f"➡️ {OUTPUT_DOMAIN}")
    print(f"➡️ {OUTPUT_STORIES}")

if __name__ == "__main__":
    main()
