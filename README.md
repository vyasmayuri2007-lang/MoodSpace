# 💜 MoodSpace: AI Emotional Wellness Companion

An AI/ML-powered emotional wellness web application that helps users express their emotions, receive empathetic responses, analyze emotional intensity, and track mental well-being over time.

The system classifies user-typed text into five core emotional states using a trained NLP model, then layers on confidence scoring, secondary emotion detection, intensity grading, and historical mood trend prediction — all surfaced through a clean, lavender-themed web dashboard.

This project includes:

- **Python/Flask backend** for ML orchestration, mood storage, and graph generation.
- **Jinja2-powered HTML frontend** with a lavender glassmorphism aesthetic.
- **TF-IDF + Multinomial Naive Bayes NLP engine** with keyword-lexicon hybrid boosting.
- **Multi-factor emotion analysis pipeline** covering negation handling, contrast clauses, and intensifier detection.
- **Automated mood trend prediction and weekly summary** powered by NumPy analytics.

---

## 📂 Project Structure

```
VITyarthi Project/
│
├── app.py                  (Flask server — routes, graph gen, AJAX endpoint)
├── emotion_classifier.py   (TF-IDF + Naive Bayes NLP engine & intensity scorer)
├── mood_analytics.py       (Trend prediction & weekly summary analytics)
├── mood_data.csv           (Persistent CSV mood history database)
│
├── static/
│   └── mood_graph.png      (Auto-generated Matplotlib mood trajectory graph)
│
├── templates/
│   └── index.html          (Main Jinja2 web UI template)
│
├── test_suite.py           (Core emotion classification test cases)
├── test_user_cases.py      (User-simulated scenario test cases)
├── test_cases_fix.py       (Regression & edge-case fix validation)
│
└── README.md
```

---

## ✨ Features

- **NLP/ML Emotion Detection** using TF-IDF (bigram) vectorization and a Multinomial Naive Bayes classifier trained on a curated benchmark dataset, classifying text into:
  - `Happy` · `Sad` · `Angry` · `Anxious` · `Neutral`
- **Keyword-Lexicon Hybrid Boosting** — A secondary keyword match layer re-weights ML probabilities using a domain-specific emotion lexicon, improving recall on emotionally explicit language.
- **Negation & Contrast Handling** — Regex-based preprocessing detects phrases like *"not sad anymore"* or *"was stressed, but now feel better"* and correctly inverts or redirects classification.
- **Confidence Scoring (54–92%)** — Primary emotion probability is normalized across all classes and scaled to a human-readable percentage.
- **Secondary Emotion Detection** — Identifies a meaningful second emotion (e.g., *"excited but nervous"*) when multiple keyword matches or a conjunction (`but`, `yet`, `although`) are present.
- **Emotion Intensity Grading** (`High` / `Medium` / `Low`) — Derived from the confidence score, presence of intensifier words (`"really"`, `"extremely"`, `"immense"`), and exclamation marks.
- **Empathetic Chat Responses** — Tailored, human-like multi-sentence responses for each emotion class, served via both full-page POST and a dedicated AJAX `/chat` endpoint.
- **Mood Timeline Graph** — Matplotlib chart of the user's emotional journey, rendered in lavender (`#8B5CF6`) and auto-saved to `static/mood_graph.png`.
- **Mood Trend Prediction** (`Improving` / `Declining` / `Stable`) — Compares rolling averages of the earliest vs. most recent half of up to 10 entries using numeric mood mappings.
- **Weekly Mood Summary** — Displays total entries, most common mood, positive/negative/neutral breakdowns, and a generated insight sentence.
- **Persistent Storage** — All entries are appended to `mood_data.csv` with date, raw text, detected sentiment, and confidence score.

---

## 🧠 How the System Works

1. **Input**: User types a free-text mood entry into the MoodSpace dashboard.
2. **Preprocessing**: `preprocess_text()` lowercases the input, detects contrast clauses (`"but now I feel…"`), and applies regex-based negation transformations (e.g., `"not angry"` → `not_angry_neutral` token).
3. **ML Prediction**: The TF-IDF vectorizer transforms the preprocessed text into a feature vector; the Naive Bayes model outputs a probability distribution over all five classes.
4. **Lexicon Boosting**: Each emotion's probability is boosted if matching keywords are found in the raw text (`boost = 0.35 + 0.10 × keyword_count`), then the full distribution is re-normalized.
5. **Emotion Selection**: The highest-scoring non-neutral emotion is selected as primary (unless Neutral probability > 0.60 with no keyword matches).
6. **Secondary & Intensity**: A secondary emotion is reported if a second distinct class is supported by keyword evidence or a conjunction; intensity is graded from the final confidence score.
7. **Response & Storage**: A tailored empathetic response is returned, and the entry is appended to `mood_data.csv`.
8. **Analytics Render**: The Flask route regenerates the mood graph, computes the trend prediction, and passes the weekly summary to the Jinja2 template for display.

---

## 📊 Emotion Analysis Logic

The system evaluates the input using five layered mechanisms:

| Mechanism | Description |
|---|---|
| **TF-IDF Vectorization** | Converts text into bigram frequency features weighted by inverse document frequency. |
| **Naive Bayes Classifier** | Computes $P(\text{Emotion} \mid \text{Text})$ using Bayes' theorem with Laplace smoothing (`α = 0.3`). |
| **Keyword Lexicon Boost** | Domain-specific word lists (e.g., `"anxious"`, `"exam"`, `"tomorrow"`) directly boost the probability of matched emotion classes. |
| **Negation Preprocessing** | Regex patterns detect negative constructions and inject special tokens before vectorization to prevent misclassification. |
| **Intensity Scoring** | `High` if confidence ≥ 74% OR (intensifier present AND confidence ≥ 58%) OR exclamation mark found. `Medium` if confidence ≥ 54%. Otherwise `Low`. |

---

## 🛠️ Technologies Used

**Backend:**

- **Python 3.x & Flask** — Web server, routing, and Jinja2 template rendering.
- **Scikit-Learn** — TF-IDF vectorization and Multinomial Naive Bayes classification.
- **Pandas & NumPy** — CSV mood history management and rolling trend analysis.
- **Matplotlib** — Dynamic mood timeline graph generation (`Agg` backend for server-side rendering).
- **Regular Expressions (`re`)** — Negation detection and contrast clause parsing.

**Frontend:**

- **HTML5 & Jinja2** — Templated web interface with dynamic data injection.
- **CSS3** — Lavender-themed design system with responsive layout.

---

## 💻 System Requirements

**Minimum:**

- Python 3.8+
- pip
- 4 GB RAM
- Windows / Linux / macOS
- A modern web browser

---

## 🚀 Setup & Installation

Follow these steps to run MoodSpace locally.

### 1. Prerequisites

Ensure you have **Python 3.8+** and **pip** installed.

### 2. Clone or Navigate to the Repository

```bash
cd "d:/2nd Year/SEM 3/AIML/VITyarthi Project"
```

### 3. (Recommended) Set Up a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows:
.\venv\Scripts\activate

# Activate on Mac/Linux:
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install flask pandas scikit-learn matplotlib nltk numpy
```

### 5. Run the Application

```bash
python app.py
```

Wait for the console to confirm:
```
Running on http://0.0.0.0:5000
```

### 6. Open in Browser

Navigate to:
```
http://127.0.0.1:5000
```

---

## 🧪 AIML Viva Quick Reference

| Concept | Detail |
|---|---|
| **Model Architecture** | TF-IDF (bigram, English stop-words removed) → Multinomial Naive Bayes (`α = 0.3`) |
| **Confidence Formula** | `min(max(top_class_probability × 100, 54), 92)` — clipped to a realistic display range |
| **Intensity Logic** | `High` if conf ≥ 74% or (intensifier + conf ≥ 58%) or `!`; `Medium` if ≥ 54%; `Low` otherwise |
| **Negation Handling** | Regex replaces constructs like `"not angry"` with synthetic token `not_angry_neutral` before vectorization |
| **Trend Algorithm** | Splits last 10 entries into two halves; computes `mean(recent) − mean(earlier)` — threshold ±0.3 |
| **Secondary Emotion** | Reported when: ≥ 2 keyword-matched classes OR (conjunction present AND 2nd class probability ≥ 0.12) |
| **Mood Mapping (numeric)** | `Happy=2, Neutral=1, Sad=0, Angry=-1, Anxious=-2` |
| **Storage** | Append-mode CSV (`mood_data.csv`) with columns: `date`, `text`, `sentiment`, `score` |

---

## 🔧 Troubleshooting

- **`ModuleNotFoundError`**: Ensure your virtual environment is active and all dependencies are installed via `pip install`.
- **Graph not displaying**: The `static/` directory is auto-created on first run; ensure Flask has write permissions to the project folder.
- **`mood_data.csv` errors**: If the file becomes corrupted, delete it — Flask will auto-recreate an empty one on the next startup.
- **Port conflict**: If port `5000` is in use, set the `PORT` environment variable: `set PORT=5001` (Windows) before running `python app.py`.

---

## 📝 License

This project is intended for educational and research purposes.

Developed as a **Capstone Project for the AIML Course** — VITyarthi Academic Program.
