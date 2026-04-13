"""Microsoft Foundry MAI Models Playground

A Streamlit app that lets you interactively test three new Microsoft Foundry models:
- MAI-Transcribe-1  – speech-to-text
- MAI-Voice-1       – text-to-speech
- MAI-Image-2       – text-to-image
"""

import base64
import importlib
import io
import json
import os
import re
import threading
import time
from urllib.parse import urlparse

import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
_AZURE_REGION_RE = re.compile(r"^[a-z0-9-]{1,50}$")
_FOUNDRY_HOST_RE = re.compile(
    r"^[a-zA-Z0-9-]+\.services\.ai\.azure\.com$"
)
_FOUNDRY_TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"


def _validate_speech_region(region: str) -> str:
    """Return the region if it looks like a valid Azure region, else raise."""
    region = region.strip().lower()
    if not _AZURE_REGION_RE.match(region):
        raise ValueError(
            f"Invalid Azure region '{region}'. "
            "A region must contain only lowercase letters, digits, and hyphens."
        )
    return region


def _normalize_foundry_endpoint(endpoint: str) -> str:
    """Normalize a Foundry resource or project endpoint to its resource base URL."""
    endpoint = endpoint.strip().rstrip("/")
    parsed = urlparse(endpoint)

    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(
            f"Invalid Foundry endpoint '{endpoint}'. "
            "Expected format: https://<resource-name>.services.ai.azure.com "
            "or https://<resource-name>.services.ai.azure.com/api/projects/<project-name>."
        )

    if not _FOUNDRY_HOST_RE.match(parsed.netloc):
        raise ValueError(
            f"Invalid Foundry endpoint '{endpoint}'. "
            "Expected a host in the format https://<resource-name>.services.ai.azure.com."
        )

    return f"https://{parsed.netloc}"


def _build_openai_inference_endpoint(foundry_endpoint: str) -> str:
    """Convert a Foundry resource or project endpoint to its OpenAI inference host."""
    normalized_endpoint = _normalize_foundry_endpoint(foundry_endpoint)
    host = urlparse(normalized_endpoint).netloc.replace(
        ".services.ai.azure.com", ".openai.azure.com"
    )
    return f"https://{host}"


def _parse_image_size(size: str) -> tuple[int, int]:
    """Convert a WIDTHxHEIGHT UI value into numeric dimensions."""
    width_str, height_str = size.split("x", maxsplit=1)
    return int(width_str), int(height_str)


def _normalize_foundry_auth_method(auth_method: str) -> str:
    """Normalize a Foundry auth mode string to a supported value."""
    normalized = auth_method.strip().lower()
    if normalized in {"api-key", "apikey", "key"}:
        return "api-key"
    if normalized in {"azuredefault", "default", "defaultazurecredential"}:
        return "azuredefault"
    raise ValueError(
        f"Invalid Foundry auth method '{auth_method}'. "
        "Expected 'api-key' or 'azuredefault'."
    )


def _normalize_tenant_id(tenant_id: str) -> str:
    """Normalize an optional tenant identifier."""
    return tenant_id.strip()


def _normalize_resource_id(resource_id: str) -> str:
    """Normalize an optional Azure resource ID."""
    return resource_id.strip()


def _normalize_speech_service_endpoint(endpoint: str) -> str:
    """Normalize an optional Azure Speech service endpoint."""
    normalized = endpoint.strip().rstrip("/")
    if not normalized:
        return ""

    parsed = urlparse(normalized)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(
            "Invalid Azure Speech Endpoint. "
            "Expected a URL like https://<resource>.cognitiveservices.azure.com."
        )

    if parsed.path and parsed.path != "/":
        raise ValueError(
            "Invalid Azure Speech Endpoint path. "
            "Expected the resource root URL such as https://<resource>.cognitiveservices.azure.com/."
        )

    return f"https://{parsed.netloc}"


def _normalize_speech_tts_endpoint(endpoint: str) -> str:
    """Normalize an optional Speech TTS endpoint override."""
    normalized = endpoint.strip().rstrip("/")
    if not normalized:
        return ""

    parsed = urlparse(normalized)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(
            "Invalid Azure Speech TTS Endpoint override. "
            "Expected a URL like https://eastus.tts.speech.microsoft.com/cognitiveservices/v1."
        )

    if parsed.path and parsed.path != "/cognitiveservices/v1":
        raise ValueError(
            "Invalid Azure Speech TTS Endpoint override path. "
            "Expected either the host root or /cognitiveservices/v1."
        )

    return f"https://{parsed.netloc}/cognitiveservices/v1"


@st.cache_resource
def _get_default_credential(tenant_id: str):
    """Create a cached Azure credential chain on demand."""
    try:
        azure_identity = importlib.import_module("azure.identity")
    except ImportError as exc:
        raise RuntimeError(
            "Azure default authentication requires the 'azure-identity' package. "
            "Install it from requirements.txt and try again."
        ) from exc

    normalized_tenant_id = _normalize_tenant_id(tenant_id)
    if not normalized_tenant_id:
        return azure_identity.ChainedTokenCredential(
            azure_identity.DefaultAzureCredential(),
            azure_identity.InteractiveBrowserCredential(),
        )

    credentials = []

    has_env_sp = bool(os.getenv("AZURE_CLIENT_ID")) and (
        bool(os.getenv("AZURE_CLIENT_SECRET"))
        or bool(os.getenv("AZURE_CLIENT_CERTIFICATE_PATH"))
    )
    if has_env_sp:
        credentials.append(azure_identity.EnvironmentCredential())

    has_workload_identity = bool(os.getenv("AZURE_CLIENT_ID")) and bool(
        os.getenv("AZURE_FEDERATED_TOKEN_FILE")
    )
    if has_workload_identity:
        credentials.append(
            azure_identity.WorkloadIdentityCredential(tenant_id=normalized_tenant_id)
        )

    credentials.extend(
        [
            azure_identity.ManagedIdentityCredential(),
            azure_identity.SharedTokenCacheCredential(
                username=os.getenv("AZURE_USERNAME") or None,
                tenant_id=normalized_tenant_id,
            ),
            azure_identity.VisualStudioCodeCredential(tenant_id=normalized_tenant_id),
            azure_identity.AzureCliCredential(tenant_id=normalized_tenant_id),
            azure_identity.AzurePowerShellCredential(tenant_id=normalized_tenant_id),
            azure_identity.AzureDeveloperCliCredential(tenant_id=normalized_tenant_id),
            azure_identity.InteractiveBrowserCredential(tenant_id=normalized_tenant_id),
        ]
    )

    return azure_identity.ChainedTokenCredential(*credentials)


def _build_foundry_headers(auth_method: str, api_key: str, tenant_id: str) -> dict[str, str]:
    """Build the MAI-Image-2 request headers for the chosen auth method."""
    headers = {"Content-Type": "application/json"}

    if auth_method == "api-key":
        key = api_key.strip()
        if not key:
            raise ValueError(
                "Please provide your Azure AI Foundry API Key in the sidebar."
            )
        headers["api-key"] = key
        return headers

    try:
        token = _get_default_credential(tenant_id).get_token(_FOUNDRY_TOKEN_SCOPE).token
    except Exception as exc:
        raise RuntimeError(
            "Failed to acquire a token with DefaultAzureCredential. "
            f"{exc}"
        ) from exc

    headers["Authorization"] = f"Bearer {token}"
    return headers


def _build_transcription_headers(subscription_key: str, tenant_id: str) -> dict[str, str]:
    """Build headers for MAI-Transcribe-1 requests."""
    key = subscription_key.strip()
    if key:
        return {"Ocp-Apim-Subscription-Key": key}

    normalized_tenant_id = _normalize_tenant_id(tenant_id)
    if not normalized_tenant_id:
        raise ValueError(
            "Provide either an Azure Speech Key or an Azure Tenant ID for Azure Default Credential."
        )

    try:
        token = _get_default_credential(normalized_tenant_id).get_token(_FOUNDRY_TOKEN_SCOPE).token
    except Exception as exc:
        raise RuntimeError(
            "Failed to acquire a Microsoft Entra token for MAI-Transcribe-1. "
            f"{exc}"
        ) from exc

    return {"Authorization": f"Bearer {token}"}


def _audio_to_wav_bytes(audio_data, sample_rate: int, channels: int) -> bytes:
    """Convert raw int16 audio samples to WAV bytes in memory."""
    import wave

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    return wav_buffer.getvalue()


def _is_speech(audio_int16, sample_rate: int = 16000, aggressiveness: int = 2) -> bool:
    """Return True if *audio_int16* contains speech, using webrtcvad.

    Falls back to True (always send) if webrtcvad is not installed.
    *aggressiveness* ranges from 0 (least aggressive / most sensitive) to 3.
    """
    try:
        import webrtcvad  # lightweight C extension, no torch needed
    except ImportError:
        return True  # degrade gracefully — send everything

    vad = webrtcvad.Vad(aggressiveness)
    raw = audio_int16.tobytes()
    # webrtcvad requires frames of 10, 20, or 30 ms at 16 kHz (320, 640, 960 bytes).
    frame_bytes = 640  # 20 ms at 16 kHz, 16-bit mono
    speech_frames = 0
    total_frames = 0
    for offset in range(0, len(raw) - frame_bytes + 1, frame_bytes):
        total_frames += 1
        if vad.is_speech(raw[offset : offset + frame_bytes], sample_rate):
            speech_frames += 1
    if total_frames == 0:
        return False
    # Consider the chunk speech if >=10% of frames contain voice
    return (speech_frames / total_frames) >= 0.10


def _transcribe_chunk(
    wav_bytes: bytes,
    batch_idx: int,
    endpoint: str,
    headers: dict[str, str],
    locale: str,
    tenant_id: str = "",
    task: str = "transcribe",
    prompt: str = "",
    target_locales: list[str] | None = None,
) -> tuple[int, str]:
    """Send a single WAV chunk to the MAI-Transcribe-1 REST API. Returns (batch_idx, text)."""
    # Use a thread-local copy of headers so concurrent retries don't race.
    local_headers = dict(headers)

    enhanced_mode: dict = {
        "enabled": True,
        "task": task,
        "model": "mai-transcribe-1",
    }
    if prompt:
        enhanced_mode["prompt"] = [prompt]
    if target_locales:
        enhanced_mode["targetLocales"] = target_locales

    definition = json.dumps(
        {
            "locales": [locale],
            "enhancedMode": enhanced_mode,
        }
    )
    files = {
        "audio": (f"batch_{batch_idx}.wav", wav_bytes, "audio/wav"),
        "definition": (None, definition, "application/json"),
    }

    try:
        response = requests.post(endpoint, headers=local_headers, files=files, timeout=30)

        # Retry with Entra if key auth is disabled on the Speech resource
        if (
            response.status_code == 403
            and "AuthenticationTypeDisabled" in response.text
            and local_headers.get("Ocp-Apim-Subscription-Key")
        ):
            local_headers = _build_transcription_headers("", tenant_id)
            # Propagate the fix back so future batches use Entra from the start
            headers.clear()
            headers.update(local_headers)
            files = {
                "audio": (f"batch_{batch_idx}.wav", wav_bytes, "audio/wav"),
                "definition": (None, definition, "application/json"),
            }
            response = requests.post(endpoint, headers=local_headers, files=files, timeout=30)

        response.raise_for_status()
        result = response.json()

        text = result.get("text", "")
        if not text:
            combined = result.get("combinedPhrases", [])
            if combined:
                text = " ".join(p.get("text", "") for p in combined)
            else:
                phrases = result.get("phrases", [])
                text = " ".join(p.get("text", "") for p in phrases)

        return batch_idx, text.strip()

    except requests.HTTPError as exc:
        body = exc.response.text[:300] if exc.response is not None else ""
        return batch_idx, f"[Batch {batch_idx} API error {exc.response.status_code}: {body}]"
    except Exception as exc:
        return batch_idx, f"[Batch {batch_idx} error: {exc}]"


def _run_realtime_mai_transcription(
    locale: str,
    chunk_seconds: float = 3.0,
    overlap_seconds: float = 1.0,
    stop_event: threading.Event | None = None,
    task: str = "transcribe",
    prompt: str = "",
    target_locales: list[str] | None = None,
    enable_vad: bool = True,
) -> None:
    """Continuously record audio via a persistent InputStream and transcribe
    micro-batches via the MAI-Transcribe-1 REST API. The microphone stays open
    for the entire session — no gaps between chunks.

    Each chunk overlaps the previous one by *overlap_seconds* so words at
    chunk boundaries are not lost (the model sees them in full context).

    When *enable_vad* is True, chunks that contain no detectable speech are
    skipped entirely (requires the ``webrtcvad`` package)."""
    import queue
    from collections import OrderedDict
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import numpy as np

    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "Micro-batch real-time transcription requires the 'sounddevice' package. "
            "Install it from requirements.txt and restart the app."
        ) from exc

    if stop_event is None:
        stop_event = threading.Event()

    SAMPLE_RATE = 16000
    CHANNELS = 1

    # Build the MAI-Transcribe-1 endpoint
    normalized_speech_endpoint = _normalize_speech_service_endpoint(speech_endpoint)
    if normalized_speech_endpoint:
        endpoint = (
            f"{normalized_speech_endpoint}/speechtotext/transcriptions:transcribe"
            "?api-version=2025-10-15"
        )
    else:
        validated_region = _validate_speech_region(speech_region)
        endpoint = (
            f"https://{validated_region}.api.cognitive.microsoft.com"
            "/speechtotext/transcriptions:transcribe"
            "?api-version=2025-10-15"
        )

    # Capture Streamlit widget values in the main thread before spawning workers.
    _tenant_id = foundry_tenant_id
    headers = _build_transcription_headers(speech_key, _tenant_id)

    status_placeholder = st.empty()
    transcript_placeholder = st.empty()

    results: OrderedDict[int, str] = OrderedDict()
    pending_futures = {}
    submit_times: dict[int, float] = {}
    batch_idx = 0
    skipped_silent = 0
    responses_received = 0
    last_latency_ms = 0.0
    chunk_frames = int(chunk_seconds * SAMPLE_RATE)
    overlap_frames = int(min(overlap_seconds, chunk_seconds * 0.5) * SAMPLE_RATE)
    # stride_frames is the NEW audio per chunk; the rest is overlap from the previous chunk
    stride_frames = chunk_frames - overlap_frames

    def _render_transcript() -> None:
        lines = [text for text in results.values() if text]
        if lines:
            transcript_placeholder.code("\n".join(lines), language="text")
        else:
            transcript_placeholder.info("Listening\u2026", icon="\U0001f4dd")

    # Persist results in session_state so they survive the rerun triggered by Stop
    if "rt_results" not in st.session_state:
        st.session_state.rt_results = []

    # --- Continuous audio capture via InputStream callback ---
    # The callback pushes raw audio frames into a thread-safe queue.
    # The main loop drains the queue to assemble fixed-size chunks with zero
    # gaps, exactly like whisper_streaming's arecord | nc pipe approach.
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    def _audio_callback(indata, frames, time_info, status):
        # Copy because indata buffer is reused by sounddevice
        audio_queue.put(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=1024,
        callback=_audio_callback,
    ), ThreadPoolExecutor(max_workers=4) as pool:

        audio_buffer = np.empty(0, dtype=np.int16)

        while not stop_event.is_set():
            vad_label = f" \u2022 {skipped_silent} silent" if enable_vad else ""
            status_placeholder.info(
                f"\U0001f399\ufe0f Listening \u2014 {chunk_seconds}s chunks "
                f"({overlap_seconds}s overlap)\u2026 "
                f"({batch_idx} sent \u2022 {responses_received} received{vad_label}"
                f"{f' \u2022 {last_latency_ms:.0f}ms latency' if last_latency_ms else ''})"
            )

            # First chunk needs full chunk_frames; subsequent chunks only need
            # stride_frames of new audio (the overlap is already in the buffer).
            needed = chunk_frames if batch_idx == 0 else stride_frames
            while len(audio_buffer) < needed and not stop_event.is_set():
                try:
                    block = audio_queue.get(timeout=0.1)
                    audio_buffer = np.concatenate([audio_buffer, block])
                except queue.Empty:
                    continue

            if stop_event.is_set():
                break

            # Take a full chunk_frames window; keep the last overlap_frames
            # in the buffer so the next chunk shares that audio.
            if len(audio_buffer) >= chunk_frames:
                chunk_data = audio_buffer[:chunk_frames]
                audio_buffer = audio_buffer[stride_frames:]
            else:
                # Less than a full chunk (edge case) — send what we have
                chunk_data = audio_buffer
                audio_buffer = np.empty(0, dtype=np.int16)

            # --- VAD gate: skip chunks with no detectable speech ---
            if enable_vad and not _is_speech(chunk_data, SAMPLE_RATE):
                skipped_silent += 1
                continue

            batch_idx += 1
            wav_bytes = _audio_to_wav_bytes(
                chunk_data.reshape(-1, 1), SAMPLE_RATE, CHANNELS
            )

            results[batch_idx] = ""
            submit_times[batch_idx] = time.monotonic()

            future = pool.submit(
                _transcribe_chunk, wav_bytes, batch_idx, endpoint, headers, locale, _tenant_id,
                task, prompt, target_locales,
            )
            pending_futures[future] = batch_idx

            # Collect any completed results and persist incrementally
            done = [f for f in pending_futures if f.done()]
            for f in done:
                idx, text = f.result()
                results[idx] = text
                responses_received += 1
                if idx in submit_times:
                    last_latency_ms = (time.monotonic() - submit_times.pop(idx)) * 1000
                del pending_futures[f]

            st.session_state.rt_results = [t for t in results.values() if t]
            _render_transcript()

        # Drain remaining in-flight API calls
        if pending_futures:
            status_placeholder.info("Finishing transcription\u2026")
            for future in as_completed(pending_futures):
                idx, text = future.result()
                results[idx] = text
                if idx in submit_times:
                    submit_times.pop(idx)
                _render_transcript()

    final_lines = [text for text in results.values() if text]
    st.session_state.rt_results = final_lines
    if final_lines:
        status_placeholder.success(
            "Transcription stopped."
        )
        _render_transcript()
    else:
        status_placeholder.warning(
            "No speech was recognized during the listening window."
        )


def _build_tts_endpoint(speech_tts_endpoint: str, speech_region: str) -> str:
    """Build the MAI-Voice-1 REST endpoint."""
    normalized_endpoint = _normalize_speech_tts_endpoint(speech_tts_endpoint)
    if normalized_endpoint:
        return normalized_endpoint

    validated_region = _validate_speech_region(speech_region)
    return f"https://{validated_region}.tts.speech.microsoft.com/cognitiveservices/v1"


def _build_tts_headers(
    subscription_key: str,
    output_format: str,
    tenant_id: str,
    resource_id: str,
) -> dict[str, str]:
    """Build headers for MAI-Voice-1 text-to-speech requests."""
    headers = {
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": output_format,
        "User-Agent": "MAI-Playground",
    }

    key = subscription_key.strip()
    if key:
        headers["Ocp-Apim-Subscription-Key"] = key
        return headers

    normalized_resource_id = _normalize_resource_id(resource_id)
    normalized_tenant_id = _normalize_tenant_id(tenant_id)
    if not normalized_resource_id or not normalized_tenant_id:
        raise ValueError(
            "Azure Default authentication for MAI-Voice-1 requires both Azure Tenant ID "
            "and Azure Speech Resource ID."
        )

    try:
        entra_token = _get_default_credential(normalized_tenant_id).get_token(
            _FOUNDRY_TOKEN_SCOPE
        ).token
    except Exception as exc:
        raise RuntimeError(
            "Failed to acquire a Microsoft Entra token for MAI-Voice-1. "
            f"{exc}"
        ) from exc

    headers["Authorization"] = f"Bearer aad#{normalized_resource_id}#{entra_token}"
    return headers


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MAI Models Playground",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Microsoft Foundry — MAI Models Playground")
st.caption(
    "Interactively test **MAI-Transcribe-1**, **MAI-Voice-1**, and **MAI-Image-2** "
    "from [Microsoft Foundry](https://microsoft.ai/)."
)

# ---------------------------------------------------------------------------
# Sidebar – credentials
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown(
        "Enter your Azure credentials below, or set them in a `.env` file "
        "(see `.env.example` for the required variable names)."
    )

    speech_key = st.text_input(
        "Azure Speech Key",
        value=os.getenv("AZURE_SPEECH_KEY", ""),
        type="password",
        help="Used by MAI-Transcribe-1 and MAI-Voice-1.",
    )
    speech_region = st.text_input(
        "Azure Speech Region",
        value=os.getenv("AZURE_SPEECH_REGION", "eastus"),
        help='E.g. "eastus", "westeurope".',
    )
    speech_endpoint = st.text_input(
        "Azure Speech Endpoint",
        value=os.getenv("AZURE_SPEECH_ENDPOINT", ""),
        help=(
            "Optional Speech resource endpoint for MAI-Transcribe-1, for example "
            "https://<resource>.cognitiveservices.azure.com/."
        ),
    )
    speech_tts_endpoint = st.text_input(
        "Azure Speech TTS Endpoint",
        value=os.getenv("AZURE_SPEECH_TTS_ENDPOINT", ""),
        help=(
            "Optional override for MAI-Voice-1. Leave blank to use the regional "
            "endpoint https://<region>.tts.speech.microsoft.com/cognitiveservices/v1."
        ),
    )
    speech_resource_id = st.text_input(
        "Azure Speech Resource ID",
        value=os.getenv("AZURE_SPEECH_RESOURCE_ID", ""),
        help=(
            "Required for MAI-Voice-1 when using Azure Default authentication. "
            "Example: /subscriptions/.../providers/Microsoft.CognitiveServices/accounts/<resource>."
        ),
    )
    openai_endpoint = st.text_input(
        "Azure AI Foundry Endpoint",
        value=os.getenv("AZURE_FOUNDRY_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT", "")),
        help=(
            'E.g. "https://my-resource.services.ai.azure.com" or '
            '"https://my-resource.services.ai.azure.com/api/projects/my-project"'
        ),
    )
    default_auth_method = os.getenv(
        "AZURE_FOUNDRY_AUTH_METHOD",
        "api-key"
        if os.getenv("AZURE_FOUNDRY_API_KEY", os.getenv("AZURE_OPENAI_KEY", "")).strip()
        else "azuredefault",
    )
    try:
        normalized_auth_method = _normalize_foundry_auth_method(default_auth_method)
    except ValueError:
        normalized_auth_method = "azuredefault"

    foundry_auth_options = {
        "Azure Default Credential": "azuredefault",
        "API Key": "api-key",
    }
    foundry_auth_label = st.selectbox(
        "Foundry Auth Method",
        options=list(foundry_auth_options.keys()),
        index=0 if normalized_auth_method == "azuredefault" else 1,
        help=(
            "Used by Foundry-backed MAI-Transcribe-1 and MAI-Image-2 requests. "
            "Choose Azure Default Credential for Entra ID auth, or API Key if your "
            "resource is configured for key-based access."
        ),
    )
    foundry_auth_method = foundry_auth_options[foundry_auth_label]
    foundry_tenant_id = st.text_input(
        "Azure Tenant ID",
        value=os.getenv("AZURE_TENANT_ID", ""),
        help=(
            "Optional tenant override for Azure Default Credential. Use this when "
            "your Foundry resource belongs to a different tenant than your default sign-in."
        ),
        disabled=foundry_auth_method != "azuredefault",
    )
    openai_key = st.text_input(
        "Azure AI Foundry API Key",
        value=os.getenv("AZURE_FOUNDRY_API_KEY", os.getenv("AZURE_OPENAI_KEY", "")),
        type="password",
        disabled=foundry_auth_method != "api-key",
        help="Used only when MAI-Image-2 Auth Method is set to API Key.",
    )
    openai_deployment = st.text_input(
        "MAI-Image-2 Deployment Name",
        value=os.getenv("AZURE_FOUNDRY_DEPLOYMENT", os.getenv("AZURE_OPENAI_DEPLOYMENT", "mai-image-2")),
        help="The deployment name you chose when deploying MAI-Image-2 in Foundry.",
    )
    st.divider()
    st.markdown(
        "**Resources**\n"
        "- [MAI-Transcribe-1 docs](https://learn.microsoft.com/azure/ai-services/speech-service/mai-transcribe)\n"
        "- [MAI-Voice-1 docs](https://learn.microsoft.com/azure/ai-services/speech-service/mai-voices)\n"
        "- [MAI-Image-2 docs](https://learn.microsoft.com/azure/foundry/foundry-models/how-to/use-foundry-models-mai)\n"
        "- [DefaultAzureCredential docs](https://learn.microsoft.com/python/api/azure-identity/azure.identity.defaultazurecredential)\n"
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_transcribe, tab_voice, tab_image = st.tabs(
    ["🎙️ MAI-Transcribe-1", "🔊 MAI-Voice-1", "🖼️ MAI-Image-2"]
)

# ===========================================================================
# Helper – call MAI-Transcribe-1 API and render result
# ===========================================================================
def _run_transcription(
    audio_bytes: bytes,
    filename: str,
    mime: str,
    locale: str,
    task: str = "transcribe",
    prompt: str = "",
    target_locales: list[str] | None = None,
) -> None:
    """Send *audio_bytes* to MAI-Transcribe-1 and render the transcript."""
    with st.spinner("Transcribing with MAI-Transcribe-1…"):
        try:
            normalized_speech_endpoint = _normalize_speech_service_endpoint(speech_endpoint)
            if normalized_speech_endpoint:
                endpoint = (
                    f"{normalized_speech_endpoint}/speechtotext/transcriptions:transcribe"
                    "?api-version=2025-10-15"
                )
            else:
                validated_region = _validate_speech_region(speech_region)
                endpoint = (
                    f"https://{validated_region}.api.cognitive.microsoft.com"
                    "/speechtotext/transcriptions:transcribe"
                    "?api-version=2025-10-15"
                )

            enhanced_mode: dict = {
                        "enabled": True,
                        "task": task,
                    }
            if prompt:
                enhanced_mode["prompt"] = [prompt]
            if target_locales:
                enhanced_mode["targetLocales"] = target_locales

            definition = json.dumps(
                {
                    "locales": [locale],
                    "enhancedMode": enhanced_mode,
                }
            )
            files = {
                "audio": (filename, audio_bytes, mime),
                "definition": (None, definition, "application/json"),
            }

            headers = _build_transcription_headers(speech_key, foundry_tenant_id)
            transcribe_start = time.monotonic()
            response = requests.post(endpoint, headers=headers, files=files, timeout=120)

            # Some Speech resources disable key auth entirely. In that case, retry with Entra.
            if (
                response.status_code == 403
                and "AuthenticationTypeDisabled" in response.text
                and headers.get("Ocp-Apim-Subscription-Key")
            ):
                headers = _build_transcription_headers("", foundry_tenant_id)
                response = requests.post(endpoint, headers=headers, files=files, timeout=120)

            response.raise_for_status()
            transcribe_elapsed = time.monotonic() - transcribe_start
            result = response.json()

            if result.get("text"):
                transcript = result.get("text", "")
            else:
                combined = result.get("combinedPhrases", [])
                if combined:
                    transcript = " ".join(p.get("text", "") for p in combined)
                else:
                    phrases = result.get("phrases", [])
                    transcript = " ".join(p.get("text", "") for p in phrases)

            if transcript:
                st.success(f"Transcription complete! Latency: {transcribe_elapsed:.2f}s")
                st.text_area("Transcript", value=transcript, height=200)
            else:
                st.warning("No transcript text was returned. Raw response:")
                st.json(result)

        except requests.HTTPError as exc:
            st.error(f"API error {exc.response.status_code}: {exc.response.text}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")


# ===========================================================================
# TAB 1 – MAI-Transcribe-1  (speech → text)
# ===========================================================================
with tab_transcribe:
    st.header("MAI-Transcribe-1 — Speech to Text")
    st.info(
        "**How it works:** MAI-Transcribe-1 uses the Azure Speech LLM API at "
        "`/speechtotext/transcriptions:transcribe?api-version=2025-10-15` with your Speech "
        "resource credentials. If `Azure Speech Endpoint` is set, the app uses that resource endpoint; "
        "otherwise it falls back to `https://<region>.api.cognitive.microsoft.com`. "
        "The request sets `enhancedMode.model = \"mai-transcribe-1\"` and `task = \"transcribe\"`. "
        "If key auth is disabled on the Speech resource, the app retries with Azure Default Credential.",
        icon="ℹ️",
    )

    transcribe_language = st.selectbox(
        "Language hint",
        options=[
            "en-US",
            "ar-SA",
            "cs-CZ",
            "da-DK",
            "de-DE",
            "es-ES",
            "fi-FI",
            "fr-FR",
            "hi-IN",
            "hu-HU",
            "id-ID",
            "it-IT",
            "ja-JP",
            "ko-KR",
            "nl-NL",
            "pl-PL",
            "pt-BR",
            "ro-RO",
            "ru-RU",
            "sv-SE",
            "th-TH",
            "tr-TR",
            "vi-VN",
            "zh-CN",
        ],
        index=0,
    )

    settings_col1, settings_col2 = st.columns(2)
    with settings_col1:
        transcribe_task = st.selectbox(
            "Task",
            options=["transcribe", "translate"],
            index=0,
            help="**transcribe** outputs text in the source language; **translate** outputs English (or your target language).",
        )
        transcribe_target_locale = st.text_input(
            "Target language (for translate)",
            value="en-US",
            help="BCP-47 locale for translation output. Only used when Task = translate.",
            disabled=transcribe_task != "translate",
        )
    with settings_col2:
        transcribe_prompt = st.text_area(
            "Custom prompt",
            value="",
            height=100,
            help=(
                "Custom instructions for the LLM speech model. Use this to specify output "
                "language, domain terms, formatting style, or contextual hints. "
                "Example: *Transcribe in formal English. Technical terms: Azure, Kubernetes, CI/CD.*"
            ),
            placeholder="e.g. Transcribe in formal English. The speaker discusses cloud computing.",
        )

    _target_locales = [transcribe_target_locale.strip()] if transcribe_task == "translate" and transcribe_target_locale.strip() else None

    st.markdown(
        """
        <style>
        /* Prominent tabs with border and highlight */
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.5rem;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 0;
        }
        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            font-size: 1.1rem;
            font-weight: 600;
            padding: 0.75rem 1.5rem;
            border: 2px solid #e0e0e0;
            border-bottom: none;
            border-radius: 0.5rem 0.5rem 0 0;
            background-color: #f8f9fa;
            transition: background-color 0.2s;
        }
        div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
            background-color: #e8f0fe;
        }
        div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
            background-color: #dbeafe;
            border-color: #4a90d9;
            color: #1a56db;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    demo_record, demo_upload, demo_realtime = st.tabs([
        "🎤 Demo 1 — Record & Transcribe",
        "📂 Demo 2 — Upload & Transcribe",
        "⚡ Demo 3 — Real-Time Transcription",
    ])

    # ------------------------------------------------------------------
    # Demo 1 – Record from microphone
    # ------------------------------------------------------------------
    with demo_record:
        st.subheader("🎤 Record from microphone")
        st.caption(
            "Click the 🎙️ microphone icon below to start recording. "
            "Click stop when you're done, then click **Transcribe recording** to transcribe."
        )
        mic_audio = st.audio_input("🎙️ Click to record", key="mic_record")
        if st.button("Transcribe recording", type="primary", disabled=mic_audio is None, key="btn_transcribe_mic"):
            _run_transcription(
                audio_bytes=mic_audio.getvalue(),
                filename="recording.wav",
                mime="audio/wav",
                locale=transcribe_language,
                task=transcribe_task,
                prompt=transcribe_prompt,
                target_locales=_target_locales,
            )

    # ------------------------------------------------------------------
    # Demo 2 – Upload an existing audio file
    # ------------------------------------------------------------------
    with demo_upload:
        st.subheader("📂 Upload an audio file")
        audio_file = st.file_uploader(
            "Upload audio file",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            label_visibility="collapsed",
        )
        if st.button("Transcribe uploaded file", type="primary", disabled=audio_file is None, key="btn_transcribe_upload"):
            _run_transcription(
                audio_bytes=audio_file.getvalue(),
                filename=audio_file.name,
                mime=audio_file.type or "audio/wav",
                locale=transcribe_language,
                task=transcribe_task,
                prompt=transcribe_prompt,
                target_locales=_target_locales,
            )

    # ------------------------------------------------------------------
    # Demo 3 – Real-Time Micro-Batch Transcription
    # ------------------------------------------------------------------
    with demo_realtime:
        st.subheader("⚡ Real-Time Micro-Batch Transcription")
        st.caption(
            "Continuously records from the default microphone and sends overlapping audio "
            "chunks to the MAI-Transcribe-1 REST API in parallel. Longer chunks and overlap "
            "improve accuracy; shorter chunks reduce latency."
        )

        if "rt_stop_event" not in st.session_state:
            st.session_state.rt_stop_event = threading.Event()
        if "rt_recording" not in st.session_state:
            st.session_state.rt_recording = False
        if "rt_results" not in st.session_state:
            st.session_state.rt_results = []
        if "rt_should_start" not in st.session_state:
            st.session_state.rt_should_start = False

        rt_col1, rt_col2 = st.columns(2)
        with rt_col1:
            realtime_chunk_seconds = st.slider(
                "Chunk size (seconds)",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5,
                key="realtime_chunk_seconds",
                help="Audio sent per API call. Longer = better accuracy, slower updates.",
            )
        with rt_col2:
            realtime_overlap_seconds = st.slider(
                "Overlap (seconds)",
                min_value=0.0,
                max_value=min(5.0, realtime_chunk_seconds * 0.5),
                value=min(1.0, realtime_chunk_seconds * 0.5),
                step=0.5,
                key="realtime_overlap_seconds",
                help="Shared audio between consecutive chunks to prevent word loss at boundaries.",
            )

        realtime_enable_vad = st.checkbox(
            "Skip silent chunks (VAD)",
            value=True,
            help=(
                "Uses voice activity detection to skip silent audio chunks, reducing API calls "
                "and cost. Requires the `webrtcvad` package; falls back to sending all chunks if not installed."
            ),
        )

        # Callback runs BEFORE the rerun — state changes are atomic and can't race.
        def _on_toggle_click():
            if st.session_state.rt_recording:
                # Stop
                st.session_state.rt_stop_event.set()
                st.session_state.rt_recording = False
            else:
                # Start — create a fresh event and flag recording to begin
                st.session_state.rt_stop_event = threading.Event()
                st.session_state.rt_recording = True
                st.session_state.rt_results = []
                st.session_state.rt_should_start = True

        toggle_label = "Stop transcription" if st.session_state.rt_recording else "Start transcription"
        st.button(toggle_label, type="primary", on_click=_on_toggle_click)

        # After the callback-triggered rerun, rt_should_start is True → run recording
        if st.session_state.rt_should_start:
            st.session_state.rt_should_start = False
            try:
                _run_realtime_mai_transcription(
                    locale=transcribe_language,
                    chunk_seconds=realtime_chunk_seconds,
                    overlap_seconds=realtime_overlap_seconds,
                    stop_event=st.session_state.rt_stop_event,
                    task=transcribe_task,
                    prompt=transcribe_prompt,
                    target_locales=_target_locales,
                    enable_vad=realtime_enable_vad,
                )
            except ValueError as exc:
                st.error(str(exc))
            except RuntimeError as exc:
                st.error(str(exc))
            # Normal completion or caught error — reset recording state
            st.session_state.rt_recording = False

        # Display previous results if available (persists after stopping)
        if not st.session_state.rt_recording and st.session_state.rt_results:
            st.code("\n".join(st.session_state.rt_results), language="text")

# ===========================================================================
# TAB 2 – MAI-Voice-1  (text → speech)
# ===========================================================================
with tab_voice:
    st.header("MAI-Voice-1 — Text to Speech")
    st.markdown(
        "Type your text below and choose a MAI-Voice-1 voice to synthesise "
        "natural, expressive speech."
    )

    tts_text = st.text_area(
        "Text to synthesise",
        value="Hello! I'm MAI-Voice-1, Microsoft's new neural text-to-speech model. "
        "I can speak with natural expression and emotion.",
        height=120,
    )

    # Known MAI-Voice-1 prebuilt voices (en-US only for now)
    voice_options = {
        "Jasper (en-US, Male)": "en-us-Jasper:MAI-Voice-1",
        "June (en-US, Female)": "en-us-June:MAI-Voice-1",
        "Grant (en-US, Male)": "en-us-Grant:MAI-Voice-1",
        "Iris (en-US, Female)": "en-us-Iris:MAI-Voice-1",
        "Reed (en-US, Male)": "en-us-Reed:MAI-Voice-1",
        "Joy (en-US, Female)": "en-us-Joy:MAI-Voice-1",
    }

    selected_voice_label = st.selectbox("Voice", options=list(voice_options.keys()), index=0)
    selected_voice = voice_options[selected_voice_label]

    output_format = st.selectbox(
        "Output format",
        options=[
            "audio-48khz-96kbitrate-mono-mp3",
            "audio-24khz-48kbitrate-mono-mp3",
            "riff-24khz-16bit-mono-pcm",
            "riff-48khz-16bit-mono-pcm",
        ],
        index=0,
    )

    if st.button("Synthesise", type="primary"):
        if not tts_text.strip():
            st.warning("Please enter some text to synthesise.")
        else:
            with st.spinner("Synthesising with MAI-Voice-1…"):
                try:
                    tts_endpoint = _build_tts_endpoint(speech_tts_endpoint, speech_region)
                    tts_headers = _build_tts_headers(
                        speech_key,
                        output_format,
                        foundry_tenant_id,
                        speech_resource_id,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()
                except RuntimeError as exc:
                    st.error(str(exc))
                    st.stop()

                ssml = (
                    '<speak version="1.0" '
                    'xmlns="http://www.w3.org/2001/10/synthesis" '
                    'xml:lang="en-US">'
                    f'<voice name="{selected_voice}">'
                    f"{tts_text}"
                    "</voice>"
                    "</speak>"
                )

                try:
                    tts_start = time.monotonic()
                    tts_response = requests.post(
                        tts_endpoint,
                        headers=tts_headers,
                        data=ssml.encode("utf-8"),
                        timeout=60,
                    )

                    # Retry with Entra if key auth is disabled on the Speech resource
                    if (
                        tts_response.status_code in (401, 403)
                        and tts_headers.get("Ocp-Apim-Subscription-Key")
                    ):
                        try:
                            tts_headers = _build_tts_headers(
                                "",
                                output_format,
                                foundry_tenant_id,
                                speech_resource_id,
                            )
                            tts_response = requests.post(
                                tts_endpoint,
                                headers=tts_headers,
                                data=ssml.encode("utf-8"),
                                timeout=60,
                            )
                        except (ValueError, RuntimeError) as retry_exc:
                            st.error(
                                f"Key auth returned {tts_response.status_code} and Entra fallback failed: {retry_exc}"
                            )
                            st.stop()

                    tts_response.raise_for_status()
                    tts_elapsed = time.monotonic() - tts_start

                    audio_bytes = tts_response.content
                    ext = "mp3" if "mp3" in output_format else "wav"
                    mime = "audio/mpeg" if ext == "mp3" else "audio/wav"
                    st.success(f"Synthesis complete! Latency: {tts_elapsed:.2f}s")
                    st.audio(audio_bytes, format=mime)

                    st.download_button(
                        label="⬇️ Download audio",
                        data=audio_bytes,
                        file_name=f"mai-voice-1-{selected_voice_label.split()[0].lower()}.{ext}",
                        mime=mime,
                    )

                except requests.HTTPError as exc:
                    st.error(
                        f"API error {exc.response.status_code}: {exc.response.text}"
                    )
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")

# ===========================================================================
# TAB 3 – MAI-Image-2  (text → image)
# ===========================================================================
with tab_image:
    st.header("MAI-Image-2 — Text to Image")
    st.markdown(
        "Describe an image and MAI-Image-2 will generate it for you through the "
        "Foundry MAI image API."
    )
    st.info(
        "**How it works:** MAI-Image-2 uses the Foundry MAI image API at "
        "`/mai/v1/images/generations`. If you paste a project endpoint such as "
        "`https://<resource>.services.ai.azure.com/api/projects/<project>`, the app "
        "automatically normalizes it to the resource base URL required by the API. "
        "Authentication can be either API key or Azure Default Credential. When a "
        "tenant ID is provided, the app requests the token from that tenant explicitly.",
        icon="ℹ️",
    )

    image_prompt = st.text_area(
        "Image prompt",
        value="A photorealistic portrait of an astronaut standing on Mars at sunset, "
        "cinematic lighting, vivid colors.",
        height=100,
    )

    col1, col2 = st.columns(2)
    with col1:
        image_size = st.selectbox(
            "Image size",
            options=["1024x1024", "1792x1024", "1024x1792"],
            index=0,
        )
    with col2:
        num_images = st.slider("Number of images", min_value=1, max_value=4, value=1)

    # Persistent gallery across generations
    if "image_gallery" not in st.session_state:
        st.session_state.image_gallery = []  # list of (img_bytes, prompt, index)

    if st.button("Generate Image", type="primary"):
        if not openai_endpoint or not openai_deployment:
            st.error(
                "Please provide your Azure AI Foundry Endpoint and Deployment Name "
                "in the sidebar."
            )
        elif foundry_auth_method == "azuredefault" and not _normalize_tenant_id(foundry_tenant_id):
            st.error("Please provide the Azure Tenant ID for Azure Default Credential.")
        elif foundry_auth_method == "api-key" and not openai_key.strip():
            st.error("Please provide your Azure AI Foundry API Key in the sidebar.")
        elif not image_prompt.strip():
            st.warning("Please enter an image prompt.")
        else:
            with st.spinner("Generating image with MAI-Image-2…"):
                try:
                    validated_endpoint = _normalize_foundry_endpoint(openai_endpoint)
                    headers = _build_foundry_headers(
                        foundry_auth_method,
                        openai_key,
                        foundry_tenant_id,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()
                except RuntimeError as exc:
                    st.error(str(exc))
                    st.stop()

                width, height = _parse_image_size(image_size)
                url = f"{validated_endpoint}/mai/v1/images/generations"

                payload = {
                    "model": openai_deployment,
                    "prompt": image_prompt,
                    "n": num_images,
                    "width": width,
                    "height": height,
                }

                try:
                    img_start = time.monotonic()
                    img_response = requests.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=120,
                    )
                    img_response.raise_for_status()
                    img_elapsed = time.monotonic() - img_start
                    result = img_response.json()

                    images_data = result.get("data", [])
                    if not images_data:
                        st.warning("No images were returned. Raw response:")
                        st.json(result)
                    else:
                        st.success(
                            f"Generated {len(images_data)} image(s) successfully! Latency: {img_elapsed:.2f}s"
                        )
                        for idx, img_data in enumerate(images_data):
                            image_url = img_data.get("url")
                            b64_data = img_data.get("b64_json")

                            if image_url:
                                img_bytes_resp = requests.get(
                                    image_url, timeout=60
                                )
                                img_bytes = img_bytes_resp.content
                            elif b64_data:
                                img_bytes = base64.b64decode(b64_data)
                            else:
                                st.warning(f"Image {idx + 1}: no URL or data.")
                                continue

                            st.session_state.image_gallery.append(
                                (img_bytes, image_prompt[:80], len(st.session_state.image_gallery) + 1)
                            )

                except requests.HTTPError as exc:
                    st.error(
                        f"API error {exc.response.status_code}: {exc.response.text}"
                    )
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")

    # -- Persistent image gallery --
    if st.session_state.image_gallery:
        st.divider()
        st.subheader("🖼️ Image Gallery")
        gallery_cols = st.columns(3)
        for i, (img_bytes, prompt_text, img_num) in enumerate(st.session_state.image_gallery):
            with gallery_cols[i % 3]:
                image = Image.open(io.BytesIO(img_bytes))
                st.image(image, caption=f"#{img_num}: {prompt_text}", width=300)
                st.download_button(
                    label=f"⬇️ Download #{img_num}",
                    data=img_bytes,
                    file_name=f"mai-image-2-{img_num}.png",
                    mime="image/png",
                    key=f"dl_gallery_{img_num}",
                )
        if st.button("Clear gallery"):
            st.session_state.image_gallery = []
            st.rerun()
