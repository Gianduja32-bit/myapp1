from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

app = Flask(__name__)

model_id = "google/gemma-3-1b-it"

# Ta clé ici :
access_token = "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"

# Chargement du modèle
tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="mps",  # Ou "auto"
    token=access_token
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="mps"
)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form["question"]
    prompt = f"<bos><|startofinst|>{question}<|endofinst|>"

    output = generator(
    prompt,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.1,
    repetition_penalty=1.2,    
    top_k=5,            
    top_p=0.9,
)
    generated_text = output[0]["generated_text"]

    # Nettoyage de la réponse
    clean_response = (
        generated_text
        .replace("<bos>", "")
        .replace("<|startofinst|>", "")
        .replace("<|endofinst|>", "")
        .replace("*", "")
        .replace(">", "")
        .replace("<", "")
        .replace("|endofinst|", "")
        .replace("|writeback", "")
        .strip()
    )

    return render_template("index.html", response=clean_response)

if __name__ == "__main__":
    app.run(debug=True)