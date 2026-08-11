import json

import requests


# Will be replaced with a wrapper for common LLM APIs.
def ask_llm(prompt: str, system: str, fmt: str | None = None, model: str = "qwen3:8b", think: bool = False) -> str:
    """
    POSTs prompt to Ollama's local HTTP server, streaming the response and
    printing each token to the console live as it arrives.
    Returns:
        the stripped, full LLM response.
    """
    body = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": True,
        "think": think
    }
    if fmt:
        body["format"] = fmt

    try:
        response = requests.post("http://localhost:11434/api/generate", json=body, timeout=150, stream=True)

        # This forces an exception if Ollama returns a 500 Internal Server Error
        response.raise_for_status()

        # Ollama sends one JSON object per line instead of a single JSON body;
        # print each fragment live and stitch them back into the full string.
        full_response = ""
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("thinking"):
                print(chunk["thinking"], end="", flush=True)
            token = chunk.get("response", "")
            if token:
                print(token, end="", flush=True)
                full_response += token
            if chunk.get("done"):
                break
        print()

        return full_response.strip()

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
