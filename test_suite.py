import pandas as pd
from emotion_classifier import emotion_classifier
from mood_analytics import predict_mood_trend, generate_weekly_summary

test_cases = {
    "Clear Emotions": [
        "I am extremely happy today.",
        "I am very sad today.",
        "I am angry about what happened.",
        "I am worried about my exam.",
        "I had a normal day."
    ],
    "Mixed Emotions": [
        "I am excited about my presentation tomorrow, but I am also really nervous.",
        "I am happy that my project is finished, but I am tired.",
        "I am angry about the situation but also disappointed."
    ],
    "Negations": [
        "I am not angry.",
        "I am not sad anymore."
    ],
    "Context & Relief": [
        "I was stressed earlier, but now I feel much better.",
        "I thought I would be nervous, but I am actually excited.",
        "I finally completed my project and I feel so relieved."
    ],
    "Ambiguous Inputs": [
        "I don't really know how I feel."
    ]
}

print("=" * 60)
print("     MOODSPACE ENHANCED ACCURACY TEST SUITE")
print("=" * 60)

for category, queries in test_cases.items():
    print(f"\n--- {category} ---")
    for q in queries:
        res = emotion_classifier.analyze(q)
        sec_str = f"{res['secondary_emotion']} ({res['secondary_confidence']}%)" if res['secondary_emotion'] else "None"
        print(f"Query: \"{q}\"")
        print(f"  -> Primary  : {res['emotion']} (Conf: {res['confidence']}%, Intensity: {res['intensity']})")
        print(f"  -> Secondary: {sec_str}")
        print("-" * 50)

print("\n" + "=" * 60)
print("     HISTORICAL ANALYTICS VERIFICATION")
print("=" * 60)
try:
    df = pd.read_csv("mood_data.csv")
    print("Trend Prediction:", predict_mood_trend(df))
    print("Weekly Summary  :", generate_weekly_summary(df))
except Exception as e:
    print("Analytics error:", e)
