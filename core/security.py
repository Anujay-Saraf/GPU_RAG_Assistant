import re
from fastapi import HTTPException, Header, status
from config.settings import settings

class SecurityGuardrail:
    PII_PATTERNS = {
        "EMAIL": r"[\w\.-]+@[\w\.-]+\.\w+",
        "PHONE": r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "SSN_AADHAAR": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
    }

    INJECTION_PATTERNS = [
        "ignore previous instructions",
        "system prompt override",
        "jailbreak",
        "bypass safety"
    ]

    @classmethod
    def sanitize_input(cls, text: str, enable_mask: bool) -> str:
        text_lower = text.lower()
        for kw in cls.INJECTION_PATTERNS:
            if kw in text_lower:
                raise HTTPException(status_code=400, detail="Security violation: Prompt injection detected.")
        if not enable_mask:
            return text
        sanitized = text
        for label, pattern in cls.PII_PATTERNS.items():
            sanitized = re.sub(pattern, f"[REDACTED_{label}]", sanitized)
        return sanitized

def verify_admin_key(x_api_key: str = Header(None)):
    if x_api_key != settings.admin_secret_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid Admin API Key.")
    return True