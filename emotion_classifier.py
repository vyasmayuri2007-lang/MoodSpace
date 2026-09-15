import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Comprehensive benchmark dataset for AIML course project
TRAINING_DATA = [
    # Happy & Relief
    ("I am feeling really happy today", "Happy"),
    ("I got a great grade on my exam", "Happy"),
    ("I am excited about the event", "Happy"),
    ("I got married and I am feeling immense joy", "Happy"),
    ("This is the best day ever, I feel wonderful", "Happy"),
    ("I love spending time with my friends", "Happy"),
    ("I feel peaceful, glad, and content", "Happy"),
    ("I was laughing a lot today and had so much fun", "Happy"),
    ("I feel like dancing today, life is great", "Happy"),
    ("I achieved my goals and I am super proud", "Happy"),
    ("I am so cheerful and full of energy", "Happy"),
    ("I am happy and delighted", "Happy"),
    ("I am extremely happy today", "Happy"),
    ("I finally completed my project and I feel so relieved", "Happy"),
    ("I finished my work and feel so relieved and calm", "Happy"),
    ("I thought I would be nervous, but I am actually excited", "Happy"),
    ("I was stressed earlier, but now I feel much better", "Happy"),
    ("I am not sad anymore, I feel great", "Happy"),
    ("I am not angry at all, everything is good", "Happy"),
    ("I am feeling good and relieved", "Happy"),
    ("I am excited for the new opportunity", "Happy"),

    # Sad
    ("I am feeling very sad and down", "Sad"),
    ("I cried a lot today and feel lonely", "Sad"),
    ("I recently had a breakup with my boyfriend and feel lost", "Sad"),
    ("I failed my semester exam this year and feel hopeless", "Sad"),
    ("Someone bullied me today and I cried", "Sad"),
    ("I feel broken, depressed, and exhausted", "Sad"),
    ("Everything is going wrong, I feel miserable", "Sad"),
    ("I miss my family so much, feeling hurt", "Sad"),
    ("I feel unloved, alone, and empty inside", "Sad"),
    ("I am suffering and feeling disappointed", "Sad"),
    ("I am very sad today", "Sad"),
    ("I am feeling down and heartbroken", "Sad"),
    ("I am disappointed with my performance", "Sad"),
    ("I feel tired and exhausted", "Sad"),
    ("I am sad about leaving my friends", "Sad"),

    # Angry
    ("I am feeling very mad and furious right now", "Angry"),
    ("I am so angry at my classmate for betraying me", "Angry"),
    ("People are annoying me and making me frustrated", "Angry"),
    ("I hate when things fail, it makes me furious", "Angry"),
    ("I am irritated, mad, and full of rage", "Angry"),
    ("Why does this always happen? I am so pissed off", "Angry"),
    ("My friends bullied me and I am enraged", "Angry"),
    ("I can't stand this unfair treatment", "Angry"),
    ("I am angry about what happened", "Angry"),
    ("I am angry about the situation", "Angry"),
    ("I am angry and frustrated with my team", "Angry"),

    # Anxious
    ("I have an exam tomorrow and I am really worried about it", "Anxious"),
    ("I am feeling kinda nervous about my presentation", "Anxious"),
    ("I am so stressed out and overthinking everything", "Anxious"),
    ("I feel scared, panic-stricken, and uneasy", "Anxious"),
    ("My heart is racing because of upcoming results", "Anxious"),
    ("I am terrified about what will happen next", "Anxious"),
    ("I feel overwhelming anxiety and pressure", "Anxious"),
    ("I am nervous about meeting new people today", "Anxious"),
    ("I feel tense, worried, and restless", "Anxious"),
    ("I am worried about my exam", "Anxious"),
    ("I am worried today because I have exam tommorow", "Anxious"),
    ("I am stressed and apprehensive", "Anxious"),
    ("I am worried about what comes next", "Anxious"),

    # Neutral & Ambiguous
    ("Today was just a normal day", "Neutral"),
    ("I had a normal day", "Neutral"),
    ("I am doing my homework right now", "Neutral"),
    ("Hey, how are you doing?", "Neutral"),
    ("I had lunch and now I am resting", "Neutral"),
    ("Just checking in on the app", "Neutral"),
    ("Nothing special happened today", "Neutral"),
    ("I walked to the library this afternoon", "Neutral"),
    ("Hi, this is my first entry", "Neutral"),
    ("The weather is okay today", "Neutral"),
    ("I don't know how I feel today", "Neutral"),
    ("I don't really know how I feel", "Neutral"),
    ("Not sure how I am feeling right now", "Neutral"),
    ("I am not angry", "Neutral"),
    ("I am not sad", "Neutral"),
]

# Comprehensive keyword lexicon mapped to emotion classes
EMOTION_LEXICON = {
    "Happy": ["happy", "good", "great", "excited", "love", "joy", "cheerful", "glad", "delighted", "relieved", "relief", "wonderful", "peaceful", "content", "proud", "joyful", "optimistic", "pleasant", "blessed"],
    "Sad": ["sad", "bad", "upset", "cry", "cried", "crying", "depressed", "lonely", "tired", "hurt", "lost", "hopeless", "disappointed", "miserable", "heartbroken", "exhausted", "down", "suffering", "gloomy", "grief"],
    "Angry": ["angry", "mad", "frustrated", "annoyed", "furious", "pissed", "rage", "irritated", "enraged", "hating", "hate", "bitter", "hostile", "resentful"],
    "Anxious": ["stress", "stressed", "anxious", "anxiety", "worried", "worry", "overthinking", "nervous", "scared", "fear", "tense", "uneasy", "panic", "terrified", "apprehensive", "frightened", "jittery", "restless", "exam", "exams", "tommorow", "tomorrow"],
    "Neutral": ["normal", "okay", "fine", "usual", "routine", "homework", "lunch", "resting", "walking"]
}

def preprocess_text(text):
    """
    Cleans text and handles negation & temporal transition clauses.
    """
    text_lower = text.lower().strip()

    # Temporal transitions ("was X earlier/before, but now Y")
    contrast_match = re.search(r'\b(but|however|although|yet)\s+(now|actually|today|i\s+feel|i\s+am)\b', text_lower)
    if contrast_match:
        split_idx = contrast_match.start()
        earlier_part = text_lower[:split_idx]
        current_part = text_lower[split_idx:]
        text_lower = earlier_part + " " + current_part + " " + current_part

    # Negation transformations
    negations = [
        (r'\b(not|n\'t|never|no longer)\s+(angry|mad|furious|frustrated|irritated)\b', 'not_angry_neutral'),
        (r'\b(not|n\'t|never|no longer)\s+(sad|depressed|upset|lonely|down)\b', 'not_sad_happy'),
        (r'\b(not|n\'t|never|no longer)\s+(worried|anxious|nervous|stressed|scared)\b', 'not_anxious_calm'),
        (r'\b(not|n\'t|never|no longer)\s+(happy|excited|glad|good)\b', 'not_happy_sad')
    ]

    processed = text_lower
    for pattern, replacement in negations:
        processed = re.sub(pattern, replacement, processed)

    return processed

class EmotionClassifier:
    def __init__(self):
        self.categories = ["Happy", "Sad", "Angry", "Anxious", "Neutral"]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        self.model = MultinomialNB(alpha=0.3)
        self._train()

    def _train(self):
        texts, labels = zip(*TRAINING_DATA)
        processed_texts = [preprocess_text(t) for t in texts]
        X = self.vectorizer.fit_transform(processed_texts)
        self.model.fit(X, labels)

    def analyze(self, text):
        if not text or not text.strip():
            return {
                "emotion": "Neutral",
                "confidence": 50,
                "intensity": "Low",
                "secondary_emotion": None,
                "secondary_confidence": None
            }

        text_clean = text.strip()
        text_lower = text_clean.lower()
        processed = preprocess_text(text_clean)

        # Ambiguity check
        ambiguous_phrases = ["don't know how i feel", "dont know how i feel", "not sure how i feel", "don't really know", "dont really know", "not sure"]
        is_ambiguous = any(phrase in text_lower for phrase in ambiguous_phrases)

        # 1. Machine Learning Model Prediction Probabilities
        vec = self.vectorizer.transform([processed])
        ml_probs = self.model.predict_proba(vec)[0]
        ml_classes = self.model.classes_
        prob_dict = {cls: float(prob) for cls, prob in zip(ml_classes, ml_probs)}

        # 2. Direct Keyword Match Analysis
        matched_emotions = {}
        for category, keywords in EMOTION_LEXICON.items():
            if category != "Neutral":
                found_words = [kw for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)]
                if found_words:
                    matched_emotions[category] = found_words

        # Handle explicit negations in keyword matches
        if "not_angry_neutral" in processed and "Angry" in matched_emotions:
            del matched_emotions["Angry"]
            prob_dict["Angry"] = 0.05
            prob_dict["Neutral"] = max(prob_dict.get("Neutral", 0), 0.50)

        if "not_sad_happy" in processed and "Sad" in matched_emotions:
            del matched_emotions["Sad"]
            prob_dict["Sad"] = 0.05
            prob_dict["Happy"] = max(prob_dict.get("Happy", 0), 0.55)

        if "not_anxious_calm" in processed and "Anxious" in matched_emotions:
            del matched_emotions["Anxious"]
            prob_dict["Anxious"] = 0.05
            prob_dict["Neutral"] = max(prob_dict.get("Neutral", 0), 0.50)

        # Combine ML model probabilities with direct keyword evidence
        if not is_ambiguous:
            for category, words in matched_emotions.items():
                word_boost = 0.35 + (0.10 * len(words))
                prob_dict[category] = max(prob_dict.get(category, 0.0), word_boost)

        # Re-normalize probability scores
        total_score = sum(prob_dict.values())
        if total_score > 0:
            for cat in prob_dict:
                prob_dict[cat] /= total_score

        # Rank non-neutral emotion categories by probability score
        non_neutral_scores = {k: v for k, v in prob_dict.items() if k != "Neutral"}
        ranked_non_neutral = sorted(non_neutral_scores.items(), key=lambda x: x[1], reverse=True)

        if prob_dict.get("Neutral", 0) > 0.60 and not matched_emotions and not is_ambiguous:
            top_cat, top_score = "Neutral", prob_dict["Neutral"]
            second_cat, second_score = ranked_non_neutral[0][0], ranked_non_neutral[0][1]
        else:
            top_cat, top_score = ranked_non_neutral[0][0], ranked_non_neutral[0][1]
            second_cat, second_score = ranked_non_neutral[1][0], ranked_non_neutral[1][1]

        # 3. Confidence & Ambiguity Resolution
        secondary_cat = None
        secondary_conf = None

        if is_ambiguous:
            top_cat = "Neutral"
            primary_conf = 48
            secondary_cat = None
            secondary_conf = None
            intensity = "Low"
        else:
            primary_conf = int(round(min(max(top_score * 100, 54), 92)))

            # 4. Secondary Emotion Analysis Rule
            conjunctions = ["but", "also", "and", "yet", "although", "still", "however", "same time", "because", "while", "though"]
            has_conjunction = any(re.search(r'\b' + re.escape(w) + r'\b', text_lower) for w in conjunctions)

            # Secondary emotion criteria:
            # - Must be a distinct non-neutral category
            # - Must have evidence: matched keywords for >= 2 emotions OR (conjunction present AND second_score >= 0.12)
            if second_cat != top_cat and second_cat != "Neutral" and top_cat != "Neutral":
                if (len(matched_emotions) >= 2 and second_cat in matched_emotions) or (has_conjunction and second_score >= 0.12):
                    secondary_cat = second_cat
                    sec_calculated = int(round(second_score * 100))
                    secondary_conf = max(38, min(sec_calculated + 8, primary_conf - 8))

            # 5. Emotion Intensity Scoring
            intensifiers = ["really", "very", "immense", "so", "extremely", "terrible", "super", "alot", "a lot", "too", "deeply", "highly", "much", "completely", "totally"]
            has_intensifier = any(re.search(r'\b' + re.escape(w) + r'\b', text_lower) for w in intensifiers)
            has_exclamation = "!" in text_clean

            if top_cat == "Neutral":
                intensity = "Low"
            elif primary_conf >= 74 or (has_intensifier and primary_conf >= 58) or has_exclamation:
                intensity = "High"
            elif primary_conf >= 54:
                intensity = "Medium"
            else:
                intensity = "Low"

        return {
            "emotion": top_cat,
            "confidence": primary_conf,
            "intensity": intensity,
            "secondary_emotion": secondary_cat,
            "secondary_confidence": secondary_conf
        }

# Global singleton instance
emotion_classifier = EmotionClassifier()
