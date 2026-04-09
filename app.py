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
        return azure_identity.DefaultAzureCredential()

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


def _get_speechsdk_module():
    """Import the Azure Speech SDK lazily."""
    try:
        return importlib.import_module("azure.cognitiveservices.speech")
    except ImportError as exc:
        raise RuntimeError(
            "Real-time transcription requires the 'azure-cognitiveservices-speech' package. "
            "Install it from requirements.txt and restart the app."
        ) from exc


def _build_realtime_speech_config(locale: str):
    """Create a Speech SDK config for microphone-based real-time transcription."""
    speechsdk = _get_speechsdk_module()
    normalized_locale = locale.strip()

    key = speech_key.strip()
    if key:
        validated_region = _validate_speech_region(speech_region)
        speech_config = speechsdk.SpeechConfig(subscription=key, region=validated_region)
        speech_config.speech_recognition_language = normalized_locale
        return speechsdk, speech_config

    normalized_tenant_id = _normalize_tenant_id(foundry_tenant_id)
    normalized_resource_id = _normalize_resource_id(speech_resource_id)
    if not normalized_tenant_id or not normalized_resource_id:
        raise ValueError(
            "Real-time transcription without a Speech key requires both Azure Tenant ID "
            "and Azure Speech Resource ID."
        )

    try:
        token = _get_default_credential(normalized_tenant_id).get_token(_FOUNDRY_TOKEN_SCOPE).token
    except Exception as exc:
        raise RuntimeError(
            "Failed to acquire a Microsoft Entra token for real-time transcription. "
            f"{exc}"
        ) from exc

    normalized_speech_endpoint = _normalize_speech_service_endpoint(speech_endpoint)
    if normalized_speech_endpoint:
        endpoint = normalized_speech_endpoint
    else:
        validated_region = _validate_speech_region(speech_region)
        endpoint = f"https://{validated_region}.api.cognitive.microsoft.com"

    speech_config = speechsdk.SpeechConfig(endpoint=endpoint)
    speech_config.authorization_token = f"aad#{normalized_resource_id}#{token}"
    speech_config.speech_recognition_language = normalized_locale
    return speechsdk, speech_config


def _run_realtime_transcription_attempt(locale: str, duration_seconds: int) -> None:
    """Run a fixed-window, best-effort real-time transcription attempt from the default mic."""
    speechsdk, speech_config = _build_realtime_speech_config(locale)
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
    )

    recognized_segments: list[str] = []
    partial_text = ""
    error_message = ""
    recognition_completed = threading.Event()

    def _handle_recognizing(event):
        nonlocal partial_text
        if event.result.text:
            partial_text = event.result.text

    def _handle_recognized(event):
        nonlocal partial_text
        if event.result.reason == speechsdk.ResultReason.RecognizedSpeech and event.result.text:
            recognized_segments.append(event.result.text)
            partial_text = ""

    def _handle_canceled(event):
        nonlocal error_message
        details = getattr(event, "cancellation_details", None)
        if details and getattr(details, "error_details", None):
            error_message = details.error_details
        else:
            error_message = str(event)
        recognition_completed.set()

    def _handle_session_stopped(_event):
        recognition_completed.set()

    speech_recognizer.recognizing.connect(_handle_recognizing)
    speech_recognizer.recognized.connect(_handle_recognized)
    speech_recognizer.canceled.connect(_handle_canceled)
    speech_recognizer.session_stopped.connect(_handle_session_stopped)

    status_placeholder = st.empty()
    transcript_placeholder = st.empty()

    def _render_realtime_transcript(lines: list[str]) -> None:
        transcript_text = "\n".join(lines).strip()
        if transcript_text:
            transcript_placeholder.code(transcript_text, language="text")
        else:
            transcript_placeholder.info("Waiting for recognized speech...", icon="📝")

    try:
        speech_recognizer.start_continuous_recognition_async().get()
        started_at = time.monotonic()

        while time.monotonic() - started_at < duration_seconds and not recognition_completed.is_set():
            seconds_remaining = max(0, int(duration_seconds - (time.monotonic() - started_at)))
            status_placeholder.info(
                f"Listening from the default microphone for up to {seconds_remaining} more seconds...",
                icon="🎙️",
            )

            live_lines = list(recognized_segments)
            if partial_text:
                live_lines.append(f"[{partial_text}]")

            _render_realtime_transcript(live_lines)
            time.sleep(0.25)
    finally:
        speech_recognizer.stop_continuous_recognition_async().get()

    if error_message:
        st.error(f"Real-time transcription failed: {error_message}")
        return

    final_lines = list(recognized_segments)
    if partial_text:
        final_lines.append(partial_text)

    if final_lines:
        status_placeholder.success("Real-time transcription attempt complete.")
        _render_realtime_transcript(final_lines)
    else:
        status_placeholder.warning(
            "No speech was recognized from the default microphone during the listening window."
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
def _run_transcription(audio_bytes: bytes, filename: str, mime: str, locale: str) -> None:
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

            definition = json.dumps(
                {
                    "locales": [locale],
                    "enhancedMode": {
                        "enabled": True,
                        "task": "transcribe",
                        "model": "mai-transcribe-1",
                    },
                }
            )
            files = {
                "audio": (filename, audio_bytes, mime),
                "definition": (None, definition, "application/json"),
            }

            headers = _build_transcription_headers(speech_key, foundry_tenant_id)
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
                st.success("Transcription complete!")
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
            "en-GB",
            "de-DE",
            "fr-FR",
            "es-ES",
            "it-IT",
            "ja-JP",
            "ko-KR",
            "pt-BR",
            "zh-CN",
            "nl-NL",
            "pl-PL",
            "ru-RU",
            "sv-SE",
            "tr-TR",
        ],
        index=0,
    )

    input_col1, input_col2 = st.columns(2)

    # ------------------------------------------------------------------
    # Option A – Record from microphone
    # ------------------------------------------------------------------
    with input_col1:
        st.subheader("🎤 Record from microphone")
        st.caption(
            "Click the microphone icon to start recording. "
            "Click stop when you're done — transcription starts automatically."
        )
        mic_audio = st.audio_input("Record audio", label_visibility="collapsed")
        if mic_audio is not None:
            _run_transcription(
                audio_bytes=mic_audio.getvalue(),
                filename="recording.wav",
                mime="audio/wav",
                locale=transcribe_language,
            )

    # ------------------------------------------------------------------
    # Option B – Upload an existing audio file
    # ------------------------------------------------------------------
    with input_col2:
        st.subheader("📂 Upload an audio file")
        audio_file = st.file_uploader(
            "Upload audio file",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            label_visibility="collapsed",
        )
        if st.button("Transcribe uploaded file", type="primary", disabled=audio_file is None):
            _run_transcription(
                audio_bytes=audio_file.getvalue(),
                filename=audio_file.name,
                mime=audio_file.type or "audio/wav",
                locale=transcribe_language,
            )

    st.divider()
    st.subheader("⚡ Experimental Real-Time Attempt")
    st.caption(
        "Best-effort microphone transcription using the Azure Speech SDK on the machine running "
        "this Streamlit app. This is a fixed-window real-time attempt, not the file-based MAI-Transcribe REST call."
    )
    realtime_duration_seconds = st.slider(
        "Listening window (seconds)",
        min_value=5,
        max_value=30,
        value=10,
        key="realtime_duration_seconds",
    )
    if st.button("Start real-time attempt", type="secondary"):
        try:
            _run_realtime_transcription_attempt(
                locale=transcribe_language,
                duration_seconds=realtime_duration_seconds,
            )
        except ValueError as exc:
            st.error(str(exc))
        except RuntimeError as exc:
            st.error(str(exc))

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

    # Known MAI-Voice-1 voices (en-US only for now)
    voice_options = {
        "Grant (en-US)": "en-US-Grant:MAI-Voice-1",
        "Teo (en-US)": "en-US-Teo:MAI-Voice-1",
        "Ava (en-US)": "en-US-Ava:MAI-Voice-1",
        "Andrew (en-US)": "en-US-Andrew:MAI-Voice-1",
        "Emma (en-US)": "en-US-Emma:MAI-Voice-1",
        "Brian (en-US)": "en-US-Brian:MAI-Voice-1",
        "Jenny (en-US)": "en-US-Jenny:MAI-Voice-1",
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
                    tts_response = requests.post(
                        tts_endpoint,
                        headers=tts_headers,
                        data=ssml.encode("utf-8"),
                        timeout=60,
                    )
                    tts_response.raise_for_status()

                    audio_bytes = tts_response.content
                    ext = "mp3" if "mp3" in output_format else "wav"
                    mime = "audio/mpeg" if ext == "mp3" else "audio/wav"
                    st.success("Synthesis complete!")
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
                    img_response = requests.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=120,
                    )
                    img_response.raise_for_status()
                    result = img_response.json()

                    images_data = result.get("data", [])
                    if not images_data:
                        st.warning("No images were returned. Raw response:")
                        st.json(result)
                    else:
                        st.success(
                            f"Generated {len(images_data)} image(s) successfully!"
                        )
                        cols = st.columns(min(len(images_data), 2))
                        for idx, img_data in enumerate(images_data):
                            col = cols[idx % len(cols)]
                            with col:
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

                                image = Image.open(io.BytesIO(img_bytes))
                                st.image(image, caption=f"Image {idx + 1}", use_container_width=True)
                                st.download_button(
                                    label=f"⬇️ Download image {idx + 1}",
                                    data=img_bytes,
                                    file_name=f"mai-image-2-{idx + 1}.png",
                                    mime="image/png",
                                )

                except requests.HTTPError as exc:
                    st.error(
                        f"API error {exc.response.status_code}: {exc.response.text}"
                    )
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")
