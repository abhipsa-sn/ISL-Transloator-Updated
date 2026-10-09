import os
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"

def correct_grammar(words_list):
    """
    Takes a list of broken English words (e.g., ["now", "i", "eat"])
    and returns a grammatically correct sentence using Gemini LLM.
    If the API key is not set or an error occurs, it falls back to just joining the words.
    """
    if not words_list:
        return ""
        
    fallback_sentence = " ".join(words_list)
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("Warning: GEMINI_API_KEY not found in environment. Using raw words.")
        return fallback_sentence
        
    prompt = (
        "You are an Indian Sign Language (ISL) interpreter. "
        "ISL uses Subject-Object-Verb (SOV) word order, so the words below may be "
        "in a non-standard order (e.g., 'I food eat' means 'I eat food'). "
        "Convert the following sequence of signed words into a single, natural, "
        "grammatically correct English sentence. "
        "Fix word order, add missing articles (a, an, the), add proper verb tenses, "
        "and make it sound fluent. "
        "Return ONLY the final corrected sentence — no quotes, no explanation, no markdown.\n\n"
        f"Signed words: {', '.join(words_list)}"
    )
    
    try:
        url = f"{GEMINI_API_URL}?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            data = response.json()
            candidates = data.get("candidates", [])
            if candidates:
                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                if text:
                    return text
        else:
            print(f"Gemini API returned status {response.status_code}: {response.text}")
        return fallback_sentence
    except Exception as e:
        print(f"Error during LLM translation: {e}")
        return fallback_sentence
