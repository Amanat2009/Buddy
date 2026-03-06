"""
personality/sentiment.py — Fast sentiment scoring using VADER.

VADER is rule-based and instant — no model load, no GPU.
Returns a compound score from -1.0 (very negative) to +1.0 (very positive).

Also provides rough emotion tagging for mood-aware responses.
"""

import logging

logger = logging.getLogger("buddy.sentiment")


class SentimentAnalyser:

    def __init__(self):
        self._analyser = None

    def load(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._analyser = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyser loaded.")
        except ImportError:
            logger.warning("vaderSentiment not installed — sentiment disabled.")

    def score(self, text: str) -> float:
        """
        Returns compound score: -1.0 to +1.0
        > 0.05  = positive
        < -0.05 = negative
        otherwise neutral
        """
        if not self._analyser or not text:
            return 0.0
        return self._analyser.polarity_scores(text)["compound"]

    def label(self, text: str) -> str:
        """Returns 'positive', 'negative', or 'neutral'."""
        s = self.score(text)
        if s > 0.05:
            return "positive"
        elif s < -0.05:
            return "negative"
        return "neutral"

    def detect_stress_keywords(self, text: str) -> bool:
        """Quick keyword scan for obvious distress signals."""
        STRESS_WORDS = {
            "stressed", "anxious", "overwhelmed", "tired", "exhausted",
            "burnout", "depressed", "anxious", "worried", "struggling",
            "can't sleep", "can't cope", "falling apart", "breaking down",
        }
        lower = text.lower()
        return any(w in lower for w in STRESS_WORDS)

    def energy_level(self, text: str) -> str:
        """
        Rough energy classification from text length and punctuation.
        Returns 'low', 'medium', or 'high'.
        """
        if not text:
            return "medium"
        words    = len(text.split())
        exclaims = text.count("!")
        ellipsis = text.count("...")

        if words < 5 or ellipsis >= 2:
            return "low"
        if exclaims >= 2 or words > 30:
            return "high"
        return "medium"
