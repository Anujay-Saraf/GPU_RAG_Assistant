import os
import re
import ollama
import openai
from config.settings import settings
from core.hardware import DEVICE, gpu_lock

GROUNDED_SYSTEM_PROMPT = """You are a grounded academic research assistant. Answer the user question based on the provided Context Sources.

CRITICAL RULES:
1. CITATIONS: Append source numbers [1], [2], etc., directly at the end of factual sentences. 
2. Do NOT write meta-commentary like "This is supported by context source [1]" or "Based on source [1]". Just state the fact and append [1].
3. CODE REQUESTS: If the user asks for code, implementation, or syntax, use the context for the theoretical structure and write clean, functional, commented Python code.
4. Start directly with the answer/code. Do not use conversational filler."""

GENERAL_FALLBACK_SYSTEM_PROMPT = """You are a senior AI solutions architect.
Answer the user's technical question with precision, code, and architectural depth.

RULES FOR CODE & EXPLANATIONS:
1. Provide ONE clean, working implementation in a single markdown code block.
2. Provide a brief 3-4 line usage example. Avoid repetitive testing loops or error handling boilerplates.
3. Do NOT include bracketed citation numbers like [1] or [2]."""

class LLMService:
    @staticmethod
    def is_code_request(query: str) -> bool:
        """Detects if the query is asking for programming implementation."""
        code_triggers = ["code", "implement", "python", "script", "function", "class", "example in", "syntax"]
        return any(t in query.lower() for t in code_triggers)

    @staticmethod
    def contextualize_query(history: list, latest_query: str) -> str:
        """Resolves ambiguous pronouns ('it', 'this', 'that') using recent chat history."""
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
            with gpu_lock:
                if settings.active_provider == "ollama":
                    client = ollama.Client(host=settings.ollama_host)
                    res = client.chat(
                        model=settings.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        options={"temperature": 0.0, "num_predict": 48}
                    )
                    rewritten = res["message"]["content"].strip().strip('"')
                    return rewritten if len(rewritten) > 3 else latest_query
        except Exception:
            return latest_query
        return latest_query

    @staticmethod
    def stream_generate(system_prompt: str, user_prompt: str):
        provider = settings.active_provider
        model = settings.model_name
        api_key = settings.api_key

        if provider == "ollama":
            client = ollama.Client(host=settings.ollama_host)
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
            base_url = "https://openrouter.ai/api/v1" if provider == "openrouter" else None
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
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