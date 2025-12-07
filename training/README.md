# 🔧 Verilog Chatbot Model Training Guide

This folder contains the training pipeline used to generate the NLP model for  
**Custom-NLP-Verilog-Chatbot**.

The chatbot uses a simple neural network (Keras/TensorFlow) trained on a  
custom Verilog intents dataset to classify user messages and generate responses.

---

## 📌 What This Training Script Does

`training.py` performs the following operations:
```markdown
1. Loads the Verilog intents dataset from: app/data/data2.json
2. Tokenizes and lemmatizes words using NLTK.
3. Builds a vocabulary (`texts.pkl`) and label set (`labels.pkl`).
4. Generates bag-of-words training vectors.
5. Trains a neural network classifier using TensorFlow/Keras.
6. Saves the trained model and tokenizers to:
app/model/model.h5
app/model/texts.pkl
app/model/labels.pkl
```

These files are later used by the chatbot for real-time inference.

---

## 📦 Requirements

Before running the training script, install dependencies:

```bash
pip install -r ../requirements.txt
```
Also ensure NLTK packages are installed:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

---

## 🚀 Running the Training Script

From the project root, run:
```bash
python training/training.py
```

If you're inside the training/ folder already:
```bash
python training.py
```

After training, you should see:
```bash
app/model/model.h5
app/model/texts.pkl
app/model/labels.pkl
```

These artifacts are automatically used by the chatbot during inference.
---
## 🧠 Model Architecture

The training script uses a lightweight neural network:
- Input: Bag-of-words vector
-Hidden Layer 1: Dense(128) + Dropout(0.5)
- Hidden Layer 2: Dense(64) + Dropout(0.5)
- Output Layer: Softmax over intent classes
- Optimizer: SGD (learning rate=0.01, momentum=0.9)

This architecture is ideal for small domain-specific datasets.
---
## 🔄 When Should You Retrain?

Retrain the model if you:
- Add new intents to data2.json
- Modify existing patterns or responses
- Improve dataset size or quality
- Change vocabulary structure
- Want better accuracy or robustness

Just run training.py again — new model files will overwrite the older ones.
---
## 📁 Output Files Explained
File	Purpose
- model.h5	Trained TensorFlow model used for prediction
- texts.pkl	List of all unique words in vocabulary
- labels.pkl	List of all intent categories (tags)
- data2.json	Intents dataset used for training
---
## ⭐ Training Tip
- If you expand your dataset significantly:
- Increase epochs
- Increase hidden layer size
- Reduce dropout slightly
- Shuffle training data more frequently

This can improve classification performance.
---
## 📝 Final Note

Your chatbot will not work correctly without the generated model files.
Always ensure:

- app/model/model.h5
- app/model/texts.pkl
- app/model/labels.pkl

are present before running app/app.py.

---










