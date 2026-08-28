import os
import re
import time
import json
import logging
import urllib.request
import ollama
import openai
from config.settings import settings
from core.hardware import DEVICE, gpu_lock

logger = logging.getLogger("EnterpriseRAG")

GROUNDED_SYSTEM_PROMPT = """You are a grounded academic research assistant. Answer the user question based on the provided Context Sources.

CRITICAL RULES:
1. CITATIONS: Append source numbers [1], [2], etc., directly at the end of factual sentences. 
2. Ground your answer strictly in the facts mentioned in the context.
3. If writing code, provide clean, runnable Python code with clear comments.
4. Start directly with the answer/code. Do not use conversational filler."""

GENERAL_FALLBACK_SYSTEM_PROMPT = """You are a senior AI solutions architect.
Answer the user's technical question with precision, working code, and architectural depth.

RULES FOR CODE & EXPLANATIONS:
1. Provide ONE clean, working implementation in a single markdown code block.
2. Provide a brief 3-4 line usage example. Avoid repetitive boilerplate or infinite loops.
3. Do NOT include bracketed citation numbers like [1] or [2]."""

# Free models prioritized on OpenRouter
OPENROUTER_FREE_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "mistralai/mistral-7b-instruct:free"
]


class AutoProviderRouter:
    """Probes connections and resolves active LLM provider and models automatically."""
    _last_probe_time = 0
    _cached_route = None
    _probe_cache_ttl = 30  # Re-evaluate every 30 seconds

    @classmethod
    def probe_ollama(cls, host_url: str) -> tuple[bool, str]:
        """Probes local or ngrok Ollama endpoint."""
        try:
            url = f"{host_url.rstrip('/')}/api/tags"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "EnterpriseRAG", "ngrok-skip-browser-warning": "true"}
            )
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    models = [m.get("name") for m in data.get("models", [])]
                    if models:
                        # Prefer llama3.2 if available, else first local model
                        preferred = next((m for m in models if "llama3.2" in m), models[0])
                        return True, preferred
                    return True, "llama3.2:1b"
        except Exception:
            pass
        return False, ""

    @classmethod
    def resolve_active_engine(cls) -> dict:
        """Determines best available provider without user intervention."""
        now = time.time()
        if cls._cached_route and (now - cls._last_probe_time) < cls._probe_cache_ttl:
            return cls._cached_route

        # 1. Check Ollama / ngrok availability
        ollama_url = os.getenv("OLLAMA_HOST", settings.ollama_host or "http://localhost:11434")
        is_ollama_alive, model_tag = cls.probe_ollama(ollama_url)

        if is_ollama_alive:
            cls._cached_route = {
                "provider": "ollama",
                "model": model_tag,
                "host": ollama_url,
                "api_key": None
            }
            cls._last_probe_time = now
            return cls._cached_route

        # 2. Check OpenRouter API Key (Supports free cloud tier)
        openrouter_key = os.getenv("OPENROUTER_API_KEY", settings.api_key if settings.active_provider == "openrouter" else "")
        if openrouter_key:
            cls._cached_route = {
                "provider": "openrouter",
                "model": OPENROUTER_FREE_MODELS[0],
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": openrouter_key
            }
            cls._last_probe_time = now
            return cls._cached_route

        # 3. Check Gemini API Key
        gemini_key = os.getenv("GEMINI_API_KEY", settings.api_key if settings.active_provider == "gemini" else "")
        if gemini_key:
            cls._cached_route = {
                "provider": "gemini",
                "model": "gemini-1.5-flash",
                "api_key": gemini_key
            }
            cls._last_probe_time = now
            return cls._cached_route

        # 4. Check OpenAI API Key
        openai_key = os.getenv("OPENAI_API_KEY", settings.api_key if settings.active_provider == "openai" else "")
        if openai_key:
            cls._cached_route = {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": openai_key
            }
            cls._last_probe_time = now
            return cls._cached_route

        # Fallback default
        cls._cached_route = {
            "provider": "openrouter",
            "model": OPENROUTER_FREE_MODELS[0],
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": settings.api_key
        }
        cls._last_probe_time = now
        return cls._cached_route


class LLMService:
    @staticmethod
    def is_code_request(query: str) -> bool:
        code_triggers = ["code", "implement", "python", "script", "function", "class", "example in", "syntax"]
        return any(t in query.lower() for t in code_triggers)

    @staticmethod
    def contextualize_query(history: list, latest_query: str) -> str:
        if not history or len(history) < 2:
            return latest_query
        
        ambiguous_tokens = [" it", " this", " that", " them", " previous", " above", " implement it", " code for it"]
        if not any(token in latest_query.lower() for token in ambiguous_tokens):
            return latest_query

        history_context = "\n".join([f"{m['role'].upper()}: {m['content'][:300]}" for m in history[-4:]])
        prompt = (
            "Rewrite the follow-up question into a standalone search query that explicitly names the concepts being discussed. "
            "Return ONLY the search query string.\n\n"
            f"History:\n{history_context}\n\n"
            f"Follow-up: {latest_query}\n\n"
            "Standalone Query:"
        )

        try:
            route = AutoProviderRouter.resolve_active_engine()
            provider = route["provider"]
            
            if provider == "ollama":
                with gpu_lock:
                    client = ollama.Client(host=route["host"])
                    res = client.chat(
                        model=route["model"],
                        messages=[{"role": "user", "content": prompt}],
                        options={"temperature": 0.0, "num_predict": 48}
                    )
                    rewritten = res["message"]["content"].strip().strip('"')
                    return rewritten if len(rewritten) > 3 else latest_query

            elif provider in ["openai", "openrouter"]:
                client = openai.OpenAI(
                    api_key=route["api_key"],
                    base_url=route.get("base_url")
                )
                res = client.chat.completions.create(
                    model=route["model"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.0
                )
                rewritten = res.choices[0].message.content.strip().strip('"')
                return rewritten if len(rewritten) > 3 else latest_query

            elif provider == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=route["api_key"])
                model = genai.GenerativeModel(route["model"])
                res = model.generate_content(prompt)
                rewritten = res.text.strip().strip('"')
                return rewritten if len(rewritten) > 3 else latest_query

        except Exception:
            return latest_query
        return latest_query

    @staticmethod
    def stream_generate(system_prompt: str, user_prompt: str):
        route = AutoProviderRouter.resolve_active_engine()
        provider = route["provider"]
        model = route["model"]

        try:
            if provider == "ollama":
                client = ollama.Client(host=route["host"])
                stream = client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    options={
                        "num_gpu": 99 if DEVICE == "cuda" else 0,
                        "temperature": 0.15,
                        "top_p": 0.90,
                        "repeat_penalty": 1.35,
                        "repeat_last_n": 128,
                        "num_predict": 1024,
                        "stop": ["Context Sources:", "Question:", "User:"]
                    },
                    stream=True
                )
                for chunk in stream:
                    if "message" in chunk and "content" in chunk["message"]:
                        yield chunk["message"]["content"]

            elif provider in ["openai", "openrouter"]:
                client = openai.OpenAI(
                    api_key=route["api_key"],
                    base_url=route.get("base_url")
                )
                stream = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1024,
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

            elif provider == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=route["api_key"])
                gemini_model = genai.GenerativeModel(model, system_instruction=system_prompt)
                response = gemini_model.generate_content(user_prompt, stream=True)
                for chunk in response:
                    if chunk.text:
                        yield chunk.text

        except Exception as e:
            # Automatic fallback: if local Ollama/ngrok fails midway, try OpenRouter free tier
            if provider == "ollama":
                logger.warning(f"Ollama failed ({e}). Falling back to OpenRouter free models...")
                openrouter_key = os.getenv("OPENROUTER_API_KEY", settings.api_key)
                if openrouter_key:
                    client = openai.AI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
                    for free_model in OPENROUTER_FREE_MODELS:
                        try:
                            stream = client.chat.completions.create(
                                model=free_model,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ],
                                temperature=0.2,
                                max_tokens=1024,
                                stream=True
                            )
                            for chunk in stream:
                                if chunk.choices[0].delta.content:
                                    yield chunk.choices[0].delta.content
                            return
                        except Exception:
                            continue
            yield f"\n\n❌ **Provider Generation Error ({provider} - {model}):** {str(e)}"
            