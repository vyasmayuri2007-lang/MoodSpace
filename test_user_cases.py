from emotion_classifier import emotion_classifier

user_queries = [
    "I am extremely happy today.",
    "I am worried about my exam tomorrow.",
    "I am excited about my presentation tomorrow, but I am also nervous.",
    "I am angry about what happened and also disappointed.",
    "I am happy that I finished my project, but I am exhausted and stressed.",
    "I am angry and disappointed with my team.",
    "I feel relieved after finishing the exam, but I am still tired.",
    "I am sad about leaving my friends, but excited for the new opportunity.",
    "I am not angry.",
    "I was stressed earlier, but now I feel much better.",
    "I don't know how I feel today."
]

print("=== TESTING USER SPECIFIC TEST CASES ===")
for q in user_queries:
    res = emotion_classifier.analyze(q)
    sec_str = f"{res['secondary_emotion']} ({res['secondary_confidence']}%)" if res['secondary_emotion'] else "None"
    print(f"Query: \"{q}\"")
    print(f"   Primary  : {res['emotion']} ({res['confidence']}%, Intensity: {res['intensity']})")
    print(f"   Secondary: {sec_str}")
    print("-" * 60)
