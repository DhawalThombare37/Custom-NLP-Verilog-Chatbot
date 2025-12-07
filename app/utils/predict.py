"""
Prediction utilities for the Custom-NLP-Verilog-Chatbot.
Handles:
- Loading model + resources
- Intent prediction
- Response selection
"""

import os
import json
import random
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from .preprocessing import bow

# -----------------------------
# File paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
WORDS_PATH = os.path.join(MODEL_DIR, "texts.pkl")
CLASSES_PATH = os.path.join(MODEL_DIR, "labels.pkl")
INTENTS_PATH = os.path.join(DATA_DIR, "data2.json")

# -----------------------------
# Cache loaded resources
# -----------------------------
_model = None
_words = None
_classes = None
_intents = None


def _load_resources():
    """Load model, vocabulary, labels, and intents JSON once."""
    global _model, _words, _classes, _intents

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
        _model = load_model(MODEL_PATH)

    if _words is None:
        with open(WORDS_PATH, "rb") as f:
            _words = pickle.load(f)

    if _classes is None:
        with open(CLASSES_PATH, "rb") as f:
            _classes = pickle.load(f)

    if _intents is None:
        with open(INTENTS_PATH, "r", encoding="utf-8") as f:
            _intents = json.load(f)


# -----------------------------
# Intent prediction
# -----------------------------
def predict_class(sentence: str, model, words, classes, threshold=0.25):
    """
    Predicts the intent class of a sentence using the trained neural network.
    """
    bow_vector = np.array([bow(sentence, words)])
    probabilities = model.predict(bow_vector, verbose=0)[0]

    # Filter predictions by threshold
    results = [[i, p] for i, p in enumerate(probabilities) if p > threshold]

    # Sort by probability
    results.sort(key=lambda x: x[1], reverse=True)

    # Convert to intent dicts
    return [{"intent": classes[i], "probability": str(p)} for i, p in results]


def get_response(predicted_intents, intents_json):
    """
    Returns a random response for the predicted intent.
    """
    tag = predicted_intents[0]["intent"]

    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "I'm not sure about that — can you rephrase?"


# -----------------------------
# Public function called by Flask route
# -----------------------------
def chatbot_response(message: str):
    """
    Main function used by Flask.
    Takes a user message and returns a chatbot answer.
    """
    if not message or not isinstance(message, str):
        return "Please enter a message."

    _load_resources()

    global _model, _words, _classes, _intents

    intents = predict_class(message, _model, _words, _classes)

    if not intents:
        # Fallback to "noanswer" tag
        for intent in _intents["intents"]:
            if intent["tag"] == "noanswer":
                return random.choice(intent["responses"])
        return "Sorry, I didn't understand that."

    return get_response(intents, _intents)

