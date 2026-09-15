import pandas as pd
import numpy as np

# Mood numerical values for trend analysis
MOOD_MAPPING = {
    "Happy": 2,
    "Neutral": 1,
    "Sad": 0,
    "Angry": -1,
    "Anxious": -2
}

def predict_mood_trend(df):
    """
    Analyzes historical mood entries to determine if the recent trend is Improving, Declining, or Stable.
    """
    if df is None or df.empty:
        return {
            "status": "Not enough data",
            "message": "Not enough data yet. Keep adding mood entries to see your trend.",
            "has_enough_data": False
        }

    # Clean data & filter valid moods
    clean_df = df.copy()
    if "sentiment" not in clean_df.columns:
        return {
            "status": "Not enough data",
            "message": "Not enough data yet. Keep adding mood entries to see your trend.",
            "has_enough_data": False
        }

    valid_entries = clean_df[clean_df["sentiment"].isin(MOOD_MAPPING.keys())].copy()

    if len(valid_entries) < 3:
        return {
            "status": "Not enough data",
            "message": "Not enough data yet. Keep adding mood entries to see your trend.",
            "has_enough_data": False
        }

    # Map sentiments to numeric values
    valid_entries["val"] = valid_entries["sentiment"].map(MOOD_MAPPING)

    # Use recent window (up to 10 entries)
    recent_data = valid_entries.tail(10)["val"].values

    if len(recent_data) < 3:
        return {
            "status": "Not enough data",
            "message": "Not enough data yet. Keep adding mood entries to see your trend.",
            "has_enough_data": False
        }

    # Split into earlier half and recent half to compare average mood levels
    mid = len(recent_data) // 2
    earlier_avg = np.mean(recent_data[:mid])
    recent_avg = np.mean(recent_data[mid:])

    diff = recent_avg - earlier_avg

    if diff >= 0.3:
        trend = "Improving"
        message = "Based on your recent entries, your overall mood appears to be improving."
    elif diff <= -0.3:
        trend = "Declining"
        message = "Based on your recent entries, your mood seems to be facing some low points. Remember to take time for self-care."
    else:
        trend = "Stable"
        message = "Based on your recent entries, your mood has been relatively steady and consistent."

    return {
        "status": trend,
        "message": message,
        "has_enough_data": True
    }


def generate_weekly_summary(df):
    """
    Computes weekly summary statistics and generates a human-like insight string.
    """
    if df is None or df.empty or "sentiment" not in df.columns:
        return {
            "total_entries": 0,
            "most_common_mood": "None",
            "positive_entries": 0,
            "negative_entries": 0,
            "neutral_entries": 0,
            "insight": "Start logging your mood entries to unlock your weekly summary."
        }

    valid_entries = df[df["sentiment"].isin(MOOD_MAPPING.keys())].copy()

    if len(valid_entries) == 0:
        return {
            "total_entries": 0,
            "most_common_mood": "None",
            "positive_entries": 0,
            "negative_entries": 0,
            "neutral_entries": 0,
            "insight": "Start logging your mood entries to unlock your weekly summary."
        }

    # Summary metrics
    total_entries = len(valid_entries)
    sentiments = valid_entries["sentiment"].tolist()

    most_common_mood = valid_entries["sentiment"].mode()[0] if not valid_entries["sentiment"].mode().empty else "Neutral"
    positive_entries = sum(1 for s in sentiments if s == "Happy")
    negative_entries = sum(1 for s in sentiments if s in ["Sad", "Angry", "Anxious"])
    neutral_entries = sum(1 for s in sentiments if s == "Neutral")

    # Generate insight
    trend_info = predict_mood_trend(valid_entries)

    if most_common_mood == "Anxious":
        if trend_info["status"] == "Improving":
            insight = "Most of your recent entries were anxious. Your mood seems to have improved over the last few days."
        else:
            insight = "Most of your recent entries reflected anxiety or stress. Grounding exercises and deep breathing can help ease your mind."
    elif most_common_mood == "Happy":
        insight = "You've experienced a wonderful streak of positive moments! Keep nurturing the activities that bring you joy."
    elif most_common_mood == "Sad":
        if trend_info["status"] == "Improving":
            insight = "You've had some sad days recently, but your emotional trajectory is beginning to lift upwards."
        else:
            insight = "You've shared several sad moments recently. Remember that it's okay to feel this way and take things one step at a time."
    elif most_common_mood == "Angry":
        insight = "Your recent entries show feelings of frustration. Taking quiet moments to unwind can help release tension."
    else:
        insight = "Your entries indicate a steady and balanced emotional state. Regular check-ins help maintain peace of mind."

    return {
        "total_entries": total_entries,
        "most_common_mood": most_common_mood,
        "positive_entries": positive_entries,
        "negative_entries": negative_entries,
        "neutral_entries": neutral_entries,
        "insight": insight
    }
