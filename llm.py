import requests


# Will be replaced with a wrapper for common LLM APIs.
def ask_llm(prompt: str, system: str, fmt: str | None = None, model: str = "qwen3:8b", think: bool = False) -> str:
    """
    POSTs prompt to Ollama's local HTTP server.
    Returns:
        the stripped LLM response.
    """
    body = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "think": think
    }
    if fmt:
        body["format"] = fmt

    try:
        response = requests.post("http://localhost:11434/api/generate", json=body, timeout=150)

        # This forces an exception if Ollama returns a 500 Internal Server Error
        response.raise_for_status()

        return response.json()["response"].strip()

    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama API timed out after 150 seconds. The server might be hung.")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Could not connect to Ollama. Did the server crash?")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama API error: {e}")


def extract_marked_fields(raw_response: str, markers: list[str]) -> dict:
    """
    Splits raw_response into a dict keyed by a slugified version of each marker
    (e.g. "MY VOTE GOES TO:" -> "my_vote_goes_to"). Each field's text runs from
    right after its marker up to whichever other marker in `markers` appears
    next, or to the end of raw_response if none does. A marker missing from
    raw_response maps to None. The unparsed response is included under "raw".
    """
    fields = {"raw": raw_response}
    for marker in markers:
        key = marker.rstrip(":").strip().lower().replace(" ", "_")
        start = raw_response.find(marker)
        if start == -1:
            fields[key] = None
            continue
        start += len(marker)
        end = len(raw_response)
        for other in markers:
            if other == marker:
                continue
            other_pos = raw_response.find(other, start)
            if other_pos != -1 and other_pos < end:
                end = other_pos
        fields[key] = raw_response[start:end].strip()
    return fields
