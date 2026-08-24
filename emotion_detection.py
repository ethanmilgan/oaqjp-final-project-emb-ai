"""Client for the Skills Network Watson NLP emotion detection service."""

import json

try:
    import requests
except ModuleNotFoundError:  # Fallback for minimal Python environments.
    from pip._vendor import requests


def emotion_detector(text_to_analyze):
    """Return emotion scores and the dominant emotion for the given text."""
    url = (
        "https://sn-watson-emotion.labs.skills.network/v1/"
        "watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )
    headers = {
        "grpc-metadata-mm-model-id":
            "emotion_aggregated-workflow_lang_en_stock"
    }
    input_json = {"raw_document": {"text": text_to_analyze}}

    response = requests.post(url, headers=headers, json=input_json)
    formatted_response = json.loads(response.text)
    emotion_scores = formatted_response["emotionPredictions"][0]["emotion"]

    anger_score = emotion_scores["anger"]
    disgust_score = emotion_scores["disgust"]
    fear_score = emotion_scores["fear"]
    joy_score = emotion_scores["joy"]
    sadness_score = emotion_scores["sadness"]

    scores = {
        "anger": anger_score,
        "disgust": disgust_score,
        "fear": fear_score,
        "joy": joy_score,
        "sadness": sadness_score,
    }
    dominant_emotion = max(scores, key=scores.get)

    return {
        "anger": anger_score,
        "disgust": disgust_score,
        "fear": fear_score,
        "joy": joy_score,
        "sadness": sadness_score,
        "dominant_emotion": dominant_emotion,
    }