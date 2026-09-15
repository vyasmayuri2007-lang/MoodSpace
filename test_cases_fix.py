from emotion_classifier import emotion_classifier

test_sentences = [
    "I am worried today because I have exam tommorow",
    "I am excited about my presentation tomorrow, but I am also nervous.",
    "I am happy that I finished my project, but I am exhausted and stressed.",
    "I am angry about what happened and also disappointed.",
    "I am angry and disappointed with my team.",
    "I feel relieved after finishing the exam, but I am still tired.",
    "I am sad about leaving my friends, but excited for the new opportunity.",
    "I am not angry.",
    "I was stressed earlier, but now I feel much better.",
    "I don't know how I feel today.",
    "I am extremely happy today."
]

for s in test_sentences:
    res = emotion_classifier.analyze(s)
    sec_str = f"{res['secondary_emotion']} ({res['secondary_confidence']}%)" if res['secondary_emotion'] else "None"
    print(f"Input: \"{s}\"")
    print(f"   Primary: {res['emotion']} ({res['confidence']}%), Secondary: {sec_str}")
    print("-" * 50)
