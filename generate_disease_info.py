import json
import requests
import time
import pandas as pd

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3"

df = pd.read_csv("Symptom2Disease.csv")
diseases = sorted(df["label"].unique().tolist())

def query_gemma(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        print(f"Error querying Gemma: {e}")
        return ""

disease_info = {}

for i, disease in enumerate(diseases):
    print(f"[{i+1}/{len(diseases)}] Generating info for: {disease}")

    sample_texts = df[df["label"] == disease]["text"].head(5).tolist()
    sample_symptoms = " | ".join(sample_texts[:3])

    prompt = f"""You are a medical AI assistant. Provide structured information about the disease "{disease}".

Here are some real patient symptom descriptions for reference:
{sample_symptoms}

Provide your response in EXACTLY this JSON format (no markdown, no extra text):
{{
  "description": "A clear 2-3 sentence medical description of {disease}.",
  "common_symptoms": ["symptom1", "symptom2", "symptom3", "symptom4", "symptom5"],
  "precautions": ["precaution1", "precaution2", "precaution3", "precaution4"],
  "medications": ["medication category 1", "medication category 2", "medication category 3"],
  "diet_recommendations": ["diet tip 1", "diet tip 2", "diet tip 3"],
  "when_to_see_doctor": "When should the patient urgently see a doctor for this condition."
}}"""

    raw = query_gemma(prompt)

    try:
        json_start = raw.index("{")
        json_end = raw.rindex("}") + 1
        parsed = json.loads(raw[json_start:json_end])
        disease_info[disease] = parsed
        print(f"  -> Success!")
    except (ValueError, json.JSONDecodeError) as e:
        print(f"  -> JSON parse failed, retrying with simpler prompt...")
        simple_prompt = f"""Give me a JSON object about the disease "{disease}" with these keys: description (string), common_symptoms (list of 5), precautions (list of 4), medications (list of 3), diet_recommendations (list of 3), when_to_see_doctor (string). Only output valid JSON, nothing else."""

        raw = query_gemma(simple_prompt)
        try:
            json_start = raw.index("{")
            json_end = raw.rindex("}") + 1
            parsed = json.loads(raw[json_start:json_end])
            disease_info[disease] = parsed
            print(f"  -> Success on retry!")
        except Exception as e2:
            print(f"  -> Failed again: {e2}")
            disease_info[disease] = {
                "description": f"Information about {disease}.",
                "common_symptoms": [],
                "precautions": [],
                "medications": [],
                "diet_recommendations": [],
                "when_to_see_doctor": "Please consult a doctor if symptoms persist."
            }

    time.sleep(2)

OUTPUT_FILE = "disease_info.json"
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(disease_info, f, indent=2, ensure_ascii=False)

print(f"\nSaved disease info for {len(disease_info)} diseases to {OUTPUT_FILE}")
