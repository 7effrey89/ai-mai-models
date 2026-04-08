"""Microsoft Foundry MAI Models Playground

A Streamlit app that lets you interactively test three new Microsoft Foundry models:
- MAI-Transcribe-1  – speech-to-text
- MAI-Voice-1       – text-to-speech
- MAI-Image-2       – text-to-image
"""

import base64
import io
import json
import os
import re

import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
_AZURE_REGION_RE = re.compile(r"^[a-z0-9-]{1,50}$")
_AZURE_OPENAI_ENDPOINT_RE = re.compile(
    r"^https://[a-zA-Z0-9-]+\.openai\.azure\.com/?$"
)


def _validate_speech_region(region: str) -> str:
    """Return the region if it looks like a valid Azure region, else raise."""
    region = region.strip().lower()
    if not _AZURE_REGION_RE.match(region):
        raise ValueError(
            f"Invalid Azure region '{region}'. "
            "A region must contain only lowercase letters, digits, and hyphens."
        )
    return region


def _validate_openai_endpoint(endpoint: str) -> str:
    """Return the endpoint if it looks like a valid Azure OpenAI URL, else raise."""
    endpoint = endpoint.strip().rstrip("/")
    if not _AZURE_OPENAI_ENDPOINT_RE.match(endpoint):
        raise ValueError(
            f"Invalid Azure OpenAI endpoint '{endpoint}'. "
            "Expected format: https://<resource-name>.openai.azure.com"
        )
    return endpoint


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
    openai_endpoint = st.text_input(
        "Azure OpenAI Endpoint",
        value=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        help='E.g. "https://my-resource.openai.azure.com"',
    )
    openai_key = st.text_input(
        "Azure OpenAI API Key",
        value=os.getenv("AZURE_OPENAI_KEY", ""),
        type="password",
        help="Used by MAI-Image-2.",
    )
    openai_deployment = st.text_input(
        "MAI-Image-2 Deployment Name",
        value=os.getenv("AZURE_OPENAI_DEPLOYMENT", "mai-image-2"),
        help="The deployment name you chose when deploying MAI-Image-2 in Foundry.",
    )
    openai_api_version = st.text_input(
        "Azure OpenAI API Version",
        value=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    )

    st.divider()
    st.markdown(
        "**Resources**\n"
        "- [MAI-Transcribe-1 docs](https://learn.microsoft.com/azure/ai-services/speech-service/mai-transcribe)\n"
        "- [MAI-Voice-1 docs](https://learn.microsoft.com/azure/ai-services/speech-service/mai-voices)\n"
        "- [MAI-Image-2 docs](https://learn.microsoft.com/azure/foundry/foundry-models/how-to/use-foundry-models-mai)\n"
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
    if not speech_key or not speech_region:
        st.error("Please provide your Azure Speech Key and Region in the sidebar.")
        return

    with st.spinner("Transcribing with MAI-Transcribe-1…"):
        try:
            validated_region = _validate_speech_region(speech_region)
        except ValueError as exc:
            st.error(str(exc))
            return

        # MAI-Transcribe-1 is accessed via the Azure LLM Speech API.
        # api-version=2025-10-15 is the version that introduces the
        # `enhancedMode` field used to select the mai-transcribe-1 model.
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
                    "model": "mai-transcribe-1",  # selects MAI-Transcribe-1 specifically
                },
            }
        )

        try:
            response = requests.post(
                endpoint,
                headers={"Ocp-Apim-Subscription-Key": speech_key},
                files={
                    "audio": (filename, audio_bytes, mime),
                    "definition": (None, definition, "application/json"),
                },
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()

            # The API returns combinedPhrases or phrases
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
        "**How it works:** MAI-Transcribe-1 is accessed via the Azure LLM Speech API "
        "(`api-version=2025-10-15`). The model is explicitly selected by setting "
        "`enhancedMode.model = \"mai-transcribe-1\"` in the request, which distinguishes "
        "it from the standard Azure Speech transcription model. "
        "Supported audio formats: WAV, MP3, FLAC (max 300 MB).",
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
        "Teo (en-US)": "en-US-Teo:MAI-Voice-1",
        "Ava (en-US)": "en-US-Ava:MAI-Voice-1",
        "Andrew (en-US)": "en-US-Andrew:MAI-Voice-1",
        "Emma (en-US)": "en-US-Emma:MAI-Voice-1",
        "Brian (en-US)": "en-US-Brian:MAI-Voice-1",
        "Jenny (en-US)": "en-US-Jenny:MAI-Voice-1",
    }

    selected_voice_label = st.selectbox("Voice", options=list(voice_options.keys()))
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
        if not speech_key or not speech_region:
            st.error("Please provide your Azure Speech Key and Region in the sidebar.")
        elif not tts_text.strip():
            st.warning("Please enter some text to synthesise.")
        else:
            with st.spinner("Synthesising with MAI-Voice-1…"):
                try:
                    validated_region = _validate_speech_region(speech_region)
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()

                tts_endpoint = (
                    f"https://{validated_region}.tts.speech.microsoft.com"
                    "/cognitiveservices/v1"
                )

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
                        headers={
                            "Ocp-Apim-Subscription-Key": speech_key,
                            "Content-Type": "application/ssml+xml",
                            "X-Microsoft-OutputFormat": output_format,
                            "User-Agent": "MAI-Playground",
                        },
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
        "Describe an image and MAI-Image-2 will generate it for you."
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
        if not openai_endpoint or not openai_key or not openai_deployment:
            st.error(
                "Please provide your Azure OpenAI Endpoint, API Key, and Deployment "
                "Name in the sidebar."
            )
        elif not image_prompt.strip():
            st.warning("Please enter an image prompt.")
        else:
            with st.spinner("Generating image with MAI-Image-2…"):
                try:
                    validated_endpoint = _validate_openai_endpoint(openai_endpoint)
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()

                url = (
                    f"{validated_endpoint}"
                    f"/openai/deployments/{openai_deployment}"
                    f"/images/generations?api-version={openai_api_version}"
                )

                payload = {
                    "prompt": image_prompt,
                    "n": num_images,
                    "size": image_size,
                }

                try:
                    img_response = requests.post(
                        url,
                        headers={
                            "api-key": openai_key,
                            "Content-Type": "application/json",
                        },
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
