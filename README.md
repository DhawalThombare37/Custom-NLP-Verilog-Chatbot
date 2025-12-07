# Custom-NLP-Verilog-Chatbot
## 👋 Welcome to Custom-NLP-Verilog-Chatbot
<img width="986" height="834" alt="image" src="https://github.com/user-attachments/assets/17446257-cdae-4ebf-8628-b2d42a8d5189" />
This project is a custom-built NLP chatbot trained on a domain-specific Verilog dataset.  
Using **Deep learning** , **Intents-based NLP** , and a clean **Flask web interface** , the chatbot can answer questions related to Verilog HDL, digital logic, circuits, and RTL design.

It showcases end-to-end chatbot development—from dataset creation and model training to web deployment.

## Highlights
- Domain-specific chatbot trained on your own Verilog intents.
- Simple Flask frontend with a responsive chat UI.
- Training pipeline included to retrain the model.
- Clear modular structure for inference and retraining.

## Repo layout
See top-level structure in the repository. Key parts:
- `app/` — Flask app, model, templates, static assets, utils.
- `training/` — training script & notes.
- `requirements.txt` — Python dependencies.

**Structure Overview :**
```markdown
Custom-NLP-Verilog-Chatbot/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── app/
│   ├── __init__.py
│   ├── app.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   └── predict.py
│   │
│   ├── model/
│   │   ├── model.h5          
│   │   ├── texts.pkl
│   │   ├── labels.pkl
│   │   └── README.md
│   │
│   ├── data/
│   │   └── data2.json
│   │
│   ├── static/
│   │   └── styles/
│   │       └── style.css
│   │
│   └── templates/
│       └── index.html
│
└── training/
    ├── training.py
    └── README.md

```

## Quick start (in development environment)
1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows PowerShell
```
2.Install dependencies:
```bash
pip install -r requirements.txt
```
3.Run the app :
```bash
# from repo root
python -m app.app
```
Then open http://127.0.0.1:5000 in your browser.

---

## Model files
The app expects:
- app/model/model.h5
- app/model/texts.pkl
- app/model/labels.pkl

## Retraining
To retrain from data/data2.json, see training/training.py. After retraining, place the saved model.h5, texts.pkl, and labels.pkl into app/model/.

---
## License
This project is released under the MIT License. See LICENSE for details.

---
## ⭐ Support the Project
If you like this project or found it useful, please consider giving the repository a ⭐ on GitHub — it really motivates and supports further development!












