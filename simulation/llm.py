import json
import os
import time

import requests
from dotenv import load_dotenv

# Load OLLAMA_API_KEY from the .env file in the repo root, wherever this is run from.
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

OLLAMA_URL = "https://ollama.com/api/generate"


def verify_api_key() -> str:
    """
    Returns OLLAMA_API_KEY, raising if it is unset or blank.
    Set it in the .env file in the repo root: OLLAMA_API_KEY=your-key
    """
    key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OLLAMA_API_KEY is not set. Put your key in the .env file in the repo root, "
            "on the line OLLAMA_API_KEY=your-key (no quotes, no spaces)."
        )
    return key


REQUEST_TIMEOUT_SECONDS = 150
# Total tries per call (so 2 retries), with this wait between them.
MAX_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 15
# Total tries for an account-level error (402, monthly limit): 1 retry to rule out a fluke.
ACCOUNT_ATTEMPTS = 2

# When True, ask_llm doesn't print tokens as they stream in. Set with set_quiet().
QUIET = False


def set_quiet(flag: bool):
    global QUIET
    QUIET = flag


class _RetryableError(Exception):
    """A failure worth trying again: timeouts, dropped connections, 429, 5xx."""


class _AccountError(Exception):
    """The account can't make calls right now: 402, or a 429 for the monthly usage limit."""


class FatalLLMError(RuntimeError):
    """A failure no later call will get past either (bad key, no access to the
    model, account out of credit or over its monthly limit), so a batch should
    stop instead of failing every remaining run."""


def _stream_once(body: dict, model: str) -> tuple[str, str]:
    """One POST to Ollama. Returns (response, thinking). Raises _RetryableError
    for transient failures and RuntimeError for ones a retry can't fix."""
    try:
        headers = {"Authorization": f"Bearer {verify_api_key()}"}
        response = requests.post(OLLAMA_URL, json=body, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS, stream=True)

        if response.status_code == 401:
            raise FatalLLMError("Ollama rejected OLLAMA_API_KEY (401). Check the key at https://ollama.com/settings/keys.")
        if response.status_code == 403:
            # The key authenticated; something else is forbidden (often model access).
            raise FatalLLMError(f"Ollama accepted the key but forbade this request for model '{model}' (403): {response.text.strip()}")
        if response.status_code == 402 or (
                response.status_code == 429 and "monthly usage limit" in response.text.lower()):
            raise _AccountError(f"HTTP {response.status_code}: {response.text.strip()[:200]}")
        if response.status_code == 429 or response.status_code >= 500:
            raise _RetryableError(f"HTTP {response.status_code}: {response.text.strip()[:200]}")
        response.raise_for_status()

        # Ollama sends one JSON object per line instead of a single JSON body;
        # print each fragment live and stitch them back into the full strings.
        full_response = ""
        full_thinking = ""
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            # Ollama can report a failure mid-stream after a 200 status.
            if chunk.get("error"):
                raise _RetryableError(f"error in stream: {chunk['error']}")
            thinking = chunk.get("thinking", "")
            if thinking:
                full_thinking += thinking
                if not QUIET:
                    print(thinking, end="", flush=True)
            token = chunk.get("response", "")
            if token:
                full_response += token
                if not QUIET:
                    print(token, end="", flush=True)
            if chunk.get("done"):
                break
        if not QUIET:
            print()

        return full_response.strip(), full_thinking.strip()

    except requests.exceptions.Timeout:
        raise _RetryableError(f"timed out after {REQUEST_TIMEOUT_SECONDS} seconds")
    except requests.exceptions.ConnectionError as e:
        # Also covers a connection that drops partway through the stream.
        raise _RetryableError(f"connection problem: {e}")
    except requests.exceptions.ChunkedEncodingError as e:
        raise _RetryableError(f"stream cut off: {e}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama API error: {e}")


# Will be replaced with a wrapper for common LLM APIs.
def ask_llm(prompt: str, system: str, model: str, fmt: str | None = None, think: bool | str = False) -> tuple[str, str]:
    """
    POSTs prompt to Ollama's cloud API, streaming the response and printing
    each token to the console live as it arrives (unless QUIET).
    think: True/False, or a level ("low"/"medium"/"high") for models like gpt-oss.
    Transient failures are retried up to MAX_ATTEMPTS times in total.
    Returns:
        (response, thinking): the stripped LLM response and its stripped
        thinking text ("" if the model gave none).
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

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _stream_once(body, model)
        except _AccountError as e:
            if attempt >= ACCOUNT_ATTEMPTS:
                raise FatalLLMError(
                    f"Ollama account can't make more calls ({e}). Check your plan and usage on "
                    "ollama.com, then re-run the same command to pick up where it stopped.")
            print(f"\n[llm] attempt {attempt}/{ACCOUNT_ATTEMPTS} failed: {e}; retrying in {RETRY_WAIT_SECONDS}s", flush=True)
            time.sleep(RETRY_WAIT_SECONDS)
        except _RetryableError as e:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"Ollama call failed after {MAX_ATTEMPTS} attempts: {e}")
            print(f"\n[llm] attempt {attempt}/{MAX_ATTEMPTS} failed: {e}; retrying in {RETRY_WAIT_SECONDS}s", flush=True)
            time.sleep(RETRY_WAIT_SECONDS)


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
