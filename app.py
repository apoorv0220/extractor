from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from pdf2image import convert_from_path
from pytesseract import image_to_string
from docx import Document

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

logging.info("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained("numind/NuExtract-1.5-tiny", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("numind/NuExtract-1.5-tiny", trust_remote_code=True)

model.to("cpu")
model.eval()
logging.info("Model and tokenizer loaded.")

def pdf_to_text(file_path):
    logging.debug(f"Converting PDF to text: {file_path}")
    pages = convert_from_path(file_path, 500)
    text = ""
    for page in pages:
        text += image_to_string(page)
    logging.debug("PDF converted to text.")
    return text

def docx_to_text(file_path):
    logging.debug(f"Converting DOCX to text: {file_path}")
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    logging.debug("DOCX converted to text.")
    return text

def predict_NuExtract(model, tokenizer, text, schema, examples=[""]):
    logging.debug("Starting prediction with NuExtract.")

    input_llm = "<|input|>\n### Template:\n" + schema + "\n"
    for example in examples:
        if example:
            input_llm += "### Example:\n" + json.dumps(json.loads(example), indent=4) + "\n"
    input_llm += "### Text:\n" + text + "\n<|output|>\n"

    logging.debug("Tokenizing input...")
    input_ids = tokenizer(input_llm, return_tensors="pt", truncation=True, max_length=2048).to('cpu')
    logging.debug("Generating model output...")
    output = tokenizer.decode(model.generate(**input_ids)[0], skip_special_tokens=True)

    start_index = output.find("<|output|>") + len("<|output|>")
    end_index = output.find("<|end-output|>")
    result = output[start_index:end_index].strip()
    logging.debug(f"Prediction completed: {result}")
    return result

@app.route('/extract', methods=['POST'])
def extract():
    logging.debug("Received request for extraction.")

    data = request.json
    input_text = data.get("text")
    schema = data.get("schema")
    examples = data.get("examples", [""])
    is_pdf = data.get("is_pdf", False)
    is_docx = data.get("is_docx", False)

    logging.debug("Extracting text from file if necessary.")
    if is_pdf:
        input_text = pdf_to_text(input_text)
    elif is_docx:
        input_text = docx_to_text(input_text)

    prediction = predict_NuExtract(model, tokenizer, input_text, schema, examples)
    logging.debug("Returning prediction.")
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    logging.info("Starting Flask application...")
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=5000, debug=True)
