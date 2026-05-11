"""
Diet Optima - Flask Backend
AI-powered nutrition analysis using Google Gemini 1.5 Flash
"""

import os
import json
import base64
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from PIL import Image
import io

app = Flask(__name__)

# ── CORS: allow ALL origins (fixes "client failed to fetch" in Chrome) ────────
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

NUTRITION_PROMPT = """
You are an expert nutritionist and food recognition AI. Analyze the food image provided and return ONLY a valid JSON object with no markdown, no code fences, and no extra text.

The JSON must follow this exact schema:
{
  "food_name": "string",
  "description": "string",
  "serving_size": "string",
  "calories": number,
  "protein_g": number,
  "carbs_g": number,
  "fat_g": number,
  "fiber_g": number,
  "sugar_g": number,
  "confidence": number,
  "ingredients": ["list", "of", "ingredients"],
  "health_tags": ["tag1", "tag2"],
  "notes": "string"
}

Rules:
- All numeric values must be plain numbers (no units).
- Estimate values for a typical single serving visible in the image.
- Never return null for numeric fields; use 0 if truly unknown.
- Return ONLY the JSON object. No preamble, no explanation.
"""

# ── Dummy fallback data (used when AI fails or for demo mode) ─────────────────
DUMMY_RESULTS = [
    {
        "food_name": "Grilled Chicken Bowl",
        "description": "A balanced bowl with grilled chicken breast, brown rice, and steamed vegetables.",
        "serving_size": "1 bowl (~450g)",
        "calories": 520.0,
        "protein_g": 42.0,
        "carbs_g": 48.0,
        "fat_g": 12.0,
        "fiber_g": 6.0,
        "sugar_g": 4.0,
        "confidence": 0.91,
        "ingredients": ["chicken breast", "brown rice", "broccoli", "carrots", "olive oil"],
        "health_tags": ["High Protein", "Balanced Meal", "Low Fat"],
        "notes": "Excellent post-workout meal. High in lean protein and complex carbs."
    },
    {
        "food_name": "Paneer Butter Masala",
        "description": "Creamy tomato-based curry with soft paneer cubes, served with rice.",
        "serving_size": "1 plate (~350g)",
        "calories": 480.0,
        "protein_g": 18.0,
        "carbs_g": 38.0,
        "fat_g": 28.0,
        "fiber_g": 3.0,
        "sugar_g": 6.0,
        "confidence": 0.88,
        "ingredients": ["paneer", "tomatoes", "butter", "cream", "spices", "rice"],
        "health_tags": ["Vegetarian", "High Calcium", "Rich"],
        "notes": "Good source of calcium and protein from paneer. Moderate in calories."
    },
    {
        "food_name": "Avocado Toast with Eggs",
        "description": "Whole grain toast topped with smashed avocado and two poached eggs.",
        "serving_size": "2 slices (~300g)",
        "calories": 390.0,
        "protein_g": 20.0,
        "carbs_g": 32.0,
        "fat_g": 22.0,
        "fiber_g": 8.0,
        "sugar_g": 2.0,
        "confidence": 0.95,
        "ingredients": ["whole grain bread", "avocado", "eggs", "lemon", "salt", "pepper"],
        "health_tags": ["High Fiber", "Healthy Fats", "High Protein"],
        "notes": "Rich in healthy monounsaturated fats and complete protein from eggs."
    },
]

import random

def get_dummy_result():
    return random.choice(DUMMY_RESULTS)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.after_request
def after_request(response):
    """Ensure CORS headers are always present, even on errors."""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    return jsonify({"status": "ok", "service": "Diet Optima API", "version": "1.0.0"})


@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze_meal():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    # ── Demo mode: if no Gemini key set, return dummy ──
    if GEMINI_API_KEY in ("YOUR_GEMINI_API_KEY_HERE", "", None):
        return jsonify({"success": True, "data": get_dummy_result(), "demo": True})

    try:
        image_bytes = None

        if "image" in request.files:
            image_bytes = request.files["image"].read()
        elif request.is_json and "image_base64" in request.json:
            b64 = request.json["image_base64"]
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            image_bytes = base64.b64decode(b64)
        else:
            return jsonify({"success": False, "error": "No image provided."}), 400

        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            pil_image.verify()
            pil_image = Image.open(io.BytesIO(image_bytes))
            if pil_image.mode not in ("RGB", "RGBA"):
                pil_image = pil_image.convert("RGB")
        except Exception as img_err:
            return jsonify({"success": False, "error": f"Invalid image: {str(img_err)}"}), 422

        response = model.generate_content(
            [NUTRITION_PROMPT, pil_image],
            generation_config=genai.types.GenerationConfig(
                temperature=0.2, max_output_tokens=1024)
        )

        raw_text = response.text.strip()
        raw_text = re.sub(r"^```[a-z]*\n?", "", raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(r"\n?```$", "", raw_text, flags=re.IGNORECASE)
        nutrition_data = json.loads(raw_text)

        for field in ["calories", "protein_g", "carbs_g", "fat_g", "fiber_g", "sugar_g", "confidence"]:
            if field in nutrition_data:
                try:
                    nutrition_data[field] = float(nutrition_data[field])
                except (TypeError, ValueError):
                    nutrition_data[field] = 0.0

        return jsonify({"success": True, "data": nutrition_data})

    except json.JSONDecodeError:
        # AI returned bad JSON — fall back to dummy
        return jsonify({"success": True, "data": get_dummy_result(), "demo": True})
    except Exception as e:
        # Any other failure — fall back to dummy instead of crashing
        return jsonify({"success": True, "data": get_dummy_result(), "demo": True, "error_info": str(e)})


@app.route("/quick-add", methods=["POST", "OPTIONS"])
def quick_add():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if GEMINI_API_KEY in ("AlzaSyAxb8KlzjvhjxvSdHgJgrY4QbRAKGQ_BDQ", "", None):
        return jsonify({"success": True, "data": get_dummy_result(), "demo": True})

    try:
        data = request.get_json()
        if not data or "food_query" not in data:
            return jsonify({"success": False, "error": "Missing 'food_query'"}), 400

        query = data["food_query"].strip()
        prompt = f"""
You are an expert nutritionist. For the food item described below, return ONLY a valid JSON object.
Food: "{query}"
Schema: {{"food_name":"string","description":"string","serving_size":"string","calories":number,"protein_g":number,"carbs_g":number,"fat_g":number,"fiber_g":number,"sugar_g":number,"confidence":number,"ingredients":[],"health_tags":[],"notes":"string"}}
Return ONLY the JSON. No markdown.
"""
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.1, max_output_tokens=512)
        )
        raw_text = re.sub(r"^```[a-z]*\n?", "", response.text.strip(), flags=re.IGNORECASE)
        raw_text = re.sub(r"\n?```$", "", raw_text, flags=re.IGNORECASE)
        return jsonify({"success": True, "data": json.loads(raw_text)})

    except Exception:
        return jsonify({"success": True, "data": get_dummy_result(), "demo": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # host=0.0.0.0 makes it reachable from Chrome on same machine
    app.run(host="0.0.0.0", port=port, debug=True)
