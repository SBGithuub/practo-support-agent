"""
pii_masker.py - Mask fixed-format patient phone numbers to protect patient privacy.
"""

import re

# Fixed-format Indian and standard 10-digit phone number patterns
# Matches variants like +91 9876543210, +91-9876543210, 9876543210, 09876543210, etc.
PHONE_REGEX = re.compile(
    r'(?:\+?91[\-\s]?)?(?:\b0)?[6-9]\d{9}\b|(?:\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b)'
)

def mask_pii(text: str) -> tuple[str, bool]:
    """
    Masks fixed-format phone numbers from the text and returns (masked_text, had_pii).
    """
    if not text:
        return text, False
    
    matches = PHONE_REGEX.findall(text)
    had_pii = len(matches) > 0
    masked_text = PHONE_REGEX.sub("[REDACTED_PHONE]", text)
    return masked_text, had_pii

def contains_unmasked_pii(text: str) -> bool:
    """Checks if text contains an unmasked phone number."""
    return bool(PHONE_REGEX.search(text))

