# ===== IMPORTS =====
from flask import Flask, render_template, request
import pandas as pd
from datetime import datetime
import os

# Fix matplotlib GUI error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from emotion_classifier import emotion_classifier
from mood_analytics import predict_mood_trend, generate_weekly_summary

# Allow Flask to find index.html either in root or in templates directory
app = Flask(__name__, template_folder='templates' if os.path.exists('templates/index.html') else '.')

# ===== FILE SETUP =====
FILE = "mood_data.csv"

# Create file if not exists or empty
if not os.path.exists(FILE) or os.stat(FILE).st_size == 0:
    pd.DataFrame(columns=["date", "text", "sentiment", "score"]).to_csv(FILE, index=False)


# ===== SENTIMENT DETECTION (Preserved Legacy Fallback) =====
def analyze_sentiment(text):
    return emotion_classifier.analyze(text)["emotion"]


# ===== HUMAN-LIKE RESPONSE =====
def generate_response(mood, text):
    if mood == "Sad":
        return [
            "That sounds really heavy. I'm really glad you shared it here.",
            "You don’t have to deal with everything at once.",
            "Try doing one small thing for yourself right now—even drinking water helps."
        ]
    elif mood == "Angry":
        return [
            "I can sense your frustration.",
            "It's okay to feel angry—it usually means something matters.",
            "Pause for a moment and take a few slow breaths."
        ]
    elif mood == "Anxious":
        return [
            "It sounds like your mind is racing.",
            "You're not alone—this happens to many people.",
            "Try grounding: name 5 things you can see around you."
        ]
    elif mood == "Happy":
        return [
            "That’s really nice to hear 😊",
            "Moments like these matter more than we realize.",
            "Take a second to appreciate this feeling."
        ]
    else:
        return [
            "Thanks for sharing that with me.",
            "Even small check-ins matter.",
            "Take things one step at a time."
        ]


# ===== GRAPH FUNCTION =====
def generate_graph():
    try:
        if not os.path.exists(FILE):
            return None

        df = pd.read_csv(FILE)

        if df.empty:
            return None

        df = df.dropna()

        mapping = {
            "Happy": 2,
            "Neutral": 1,
            "Sad": 0,
            "Angry": -1,
            "Anxious": -2
        }

        df = df[df["sentiment"].isin(mapping.keys())].copy()
        if df.empty:
            return None

        df["date"] = pd.to_datetime(df["date"], errors='coerce')
        df = df.dropna(subset=["date"])

        df["value"] = df["sentiment"].map(mapping)
        df = df.dropna()

        if df.empty:
            return None

        df = df.sort_values("date")

        os.makedirs("static", exist_ok=True)

        plt.figure(figsize=(10, 5))
        plt.plot(df["date"], df["value"], marker='o', color='#8B5CF6', linewidth=2, markersize=6)

        plt.title("Your Mood Journey 💜", fontsize=14, fontweight='bold', color='#5A4FCF')
        plt.xlabel("Date", fontsize=11)
        plt.ylabel("Mood Level", fontsize=11)

        plt.yticks([-2, -1, 0, 1, 2], ["Anxious (-2)", "Angry (-1)", "Sad (0)", "Neutral (1)", "Happy (2)"])
        plt.xticks(rotation=30)
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        graph_path = "static/mood_graph.png"
        plt.savefig(graph_path)
        plt.close()

        return graph_path

    except Exception as e:
        print("Graph Error:", e)
        return None


# ===== MAIN ROUTE =====
@app.route("/", methods=["GET", "POST"])
def index():
    user_text = ""
    response = []
    emotion_result = None

    try:
        df = pd.read_csv(FILE)
    except:
        df = pd.DataFrame(columns=["date", "text", "sentiment", "score"])

    if request.method == "POST":
        text = request.form.get("mood", "")

        if text.strip() != "":
            # Run enhanced NLP/ML emotion classification
            emotion_result = emotion_classifier.analyze(text)
            mood = emotion_result["emotion"]
            response = generate_response(mood, text)

            file_exists = os.path.exists(FILE) and os.stat(FILE).st_size > 0

            # Save entry to CSV
            pd.DataFrame([{
                "date": datetime.now().strftime("%Y-%m-%d"),
                "text": text,
                "sentiment": mood,
                "score": emotion_result["confidence"]
            }]).to_csv(FILE, mode='a', header=not file_exists, index=False)

            user_text = text
            # Reload updated dataframe
            try:
                df = pd.read_csv(FILE)
            except:
                pass

    graph = generate_graph()
    trend_prediction = predict_mood_trend(df)
    weekly_summary = generate_weekly_summary(df)

    return render_template(
        "index.html",
        message=user_text,
        user_text=user_text,
        response=response,
        emotion_result=emotion_result,
        graph=graph,
        trend_prediction=trend_prediction,
        weekly_summary=weekly_summary,
        data=df.tail(5)
    )


# ===== AJAX CHAT ENDPOINT =====
@app.route("/chat", methods=["POST"])
def chat():
    from flask import jsonify
    text = request.form.get("mood", "")

    if not text.strip():
        return jsonify({"error": "Empty input"}), 400

    emotion_result = emotion_classifier.analyze(text)
    mood = emotion_result["emotion"]
    response = generate_response(mood, text)

    file_exists = os.path.exists(FILE) and os.stat(FILE).st_size > 0
    pd.DataFrame([{
        "date": datetime.now().strftime("%Y-%m-%d"),
        "text": text,
        "sentiment": mood,
        "score": emotion_result["confidence"]
    }]).to_csv(FILE, mode='a', header=not file_exists, index=False)

    return jsonify({
        "user_text": text,
        "response": response,
        "emotion": emotion_result["emotion"],
        "confidence": emotion_result["confidence"],
        "intensity": emotion_result["intensity"],
        "secondary_emotion": emotion_result["secondary_emotion"],
        "secondary_confidence": emotion_result["secondary_confidence"]
    })


# ===== RUN =====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)