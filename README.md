# 🧠 AI Player Prop Edge Finder

This project uses a machine learning model to estimate the probability that a
player prop will hit, then compares that to the implied probability from the
betting odds to calculate an **edge**.

This tool is for **educational and analytical purposes only**.  
This is **not** betting advice.

---

## 🚀 Features

- Upload **historical prop data** and train a custom ML model
- Upload **today's props** and get:
  - Model-estimated hit probabilities
  - Estimated edge vs implied probability
- Multi-sport support (NBA, NFL, MLB, NHL, etc.)
- Streamlit-based web interface
- Clean, transparent feature-driven modeling (no "mystery model")

---

## 📄 File Overview

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit app (UI). |
| `prop_model_core.py` | ML model training + scoring logic. |
| `requirements.txt` | Python dependencies for Streamlit Cloud. |
| `README.md` | Project description + usage instructions. |

---

## 📦 Installation & Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
