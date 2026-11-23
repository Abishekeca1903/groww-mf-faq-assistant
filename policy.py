"""Lightweight query safety checks; intentionally has no app dependencies."""
import re


def looks_like_pii(text: str) -> bool:
    patterns = [
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        r"\b\d{12}\b",
        r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
        r"(?<!\d)(?:\+91[- ]?)?[6-9]\d{9}(?!\d)",
        r"\b(?:otp|one[- ]time password)\s*[:=-]?\s*\d{4,8}\b",
        r"\b(?:account|folio)\s*(?:number|no\.?)\s*[:#=-]?\s*[A-Za-z0-9-]{5,}\b",
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def is_advice_or_performance(text: str) -> bool:
    normalized = text.lower()
    advice_patterns = [
        "should i buy", "should i sell", "should i invest", "which fund should",
        "best mutual fund for me", "recommend a fund", "recommendation",
        "build my portfolio", "asset allocation", "where should i invest",
        "switch my", "redeem my", "is this fund good", "will this fund outperform",
        "expected returns", "predict returns", "how much return", "compare returns",
        "which is better", "top performing", "highest return",
    ]
    return any(pattern in normalized for pattern in advice_patterns)
