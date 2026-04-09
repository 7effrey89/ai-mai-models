# ai-mai-models

A Streamlit playground for testing the three new Microsoft Foundry MAI models:

| Model | Capability |
|---|---|
| **MAI-Transcribe-1** | Speech-to-text – record from mic or upload a file, transcribes in 25+ languages |
| **MAI-Voice-1** | Text-to-speech – expressive, natural neural voices |
| **MAI-Image-2** | Text-to-image – high-quality diffusion-based image generation |

> Learn more: [Microsoft Foundry announcement](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-mai-transcribe-1-mai-voice-1-and-mai-image-2-in-microsoft-foundry/4507787)

## Screenshots

![Playground tabs](https://i.imgur.com/placeholder.png)

## Prerequisites

- Python 3.9+
- An **Azure Speech resource** (for MAI-Transcribe-1 and MAI-Voice-1)
  - Key and region available from the Azure portal
- A **Microsoft Foundry resource/project** with **MAI-Image-2 deployed** (for MAI-Image-2)
  - Foundry endpoint and deployment name available from the Azure portal or Foundry portal
  - For Entra ID auth, a working Azure identity via `DefaultAzureCredential` such as Azure CLI login, Visual Studio Code sign-in, or managed identity

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/7effrey89/ai-mai-models.git
cd ai-mai-models

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure credentials
cp .env.example .env
# Open .env and fill in your Azure credentials
```

### Environment variables

| Variable | Description |
|---|---|
| `AZURE_SPEECH_KEY` | Azure Speech resource key (MAI-Transcribe-1 & MAI-Voice-1) |
| `AZURE_SPEECH_REGION` | Azure region, e.g. `eastus` |
| `AZURE_SPEECH_ENDPOINT` | Optional Speech resource endpoint for MAI-Transcribe-1, e.g. `https://<resource>.cognitiveservices.azure.com/` |
| `AZURE_SPEECH_TTS_ENDPOINT` | Optional MAI-Voice-1 TTS endpoint override. If unset, the app uses `https://<region>.tts.speech.microsoft.com/cognitiveservices/v1` |
| `AZURE_SPEECH_RESOURCE_ID` | Speech resource ID used to construct Entra-authenticated MAI-Voice-1 bearer tokens |
| `AZURE_FOUNDRY_ENDPOINT` | Foundry endpoint URL for MAI-Image-2. Can be either `https://<resource>.services.ai.azure.com` or `https://<resource>.services.ai.azure.com/api/projects/<project>` |
| `AZURE_FOUNDRY_AUTH_METHOD` | `azuredefault` or `api-key` |
| `AZURE_TENANT_ID` | Tenant to use for Azure Default Credential token acquisition |
| `AZURE_FOUNDRY_API_KEY` | Foundry API key. Leave blank when using `azuredefault` |
| `AZURE_FOUNDRY_DEPLOYMENT` | Your MAI-Image-2 deployment name |

The app also accepts the legacy variable names `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, and `AZURE_OPENAI_DEPLOYMENT` for backward compatibility, but new setups should prefer the `AZURE_FOUNDRY_*` names.

You can also enter credentials directly in the **sidebar** of the running app – no `.env` file required.

## Running the playground

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## How to deploy the models

### MAI-Transcribe-1 & MAI-Voice-1

MAI-Voice-1 uses the **Azure Speech Service**. Create a Speech resource in the [Azure portal](https://portal.azure.com) and note the region. You can authenticate either with a speech key or with Microsoft Entra ID plus the speech resource ID.

MAI-Transcribe-1 uses the Azure Speech LLM API. If you have a resource endpoint such as `https://jlafoundryeastus.cognitiveservices.azure.com/`, set `AZURE_SPEECH_ENDPOINT`; otherwise the app falls back to the regional endpoint built from `AZURE_SPEECH_REGION`.

The app first tries `AZURE_SPEECH_KEY`. If key-based auth is disabled on the Speech resource, it automatically retries with Azure Default Credential using `AZURE_TENANT_ID`.

For **MAI-Voice-1** with Microsoft Entra authentication, the app can also use your Azure identity instead of a speech key. In that mode it sends `Authorization: Bearer aad#<resource-id>#<entra-token>` to the regional Speech TTS endpoint and requires `AZURE_SPEECH_RESOURCE_ID` plus `AZURE_TENANT_ID`.

By default, the app uses the regional TTS endpoint format `https://<speech-region>.tts.speech.microsoft.com/cognitiveservices/v1`. If your Speech resource exposes a different TTS endpoint, set `AZURE_SPEECH_ENDPOINT` to override it explicitly.

For transcription, the app calls `POST /speechtotext/transcriptions:transcribe?api-version=2025-10-15` and sets `enhancedMode.model = "mai-transcribe-1"`.

- [MAI-Transcribe-1 documentation](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-transcribe)
- [MAI-Voice-1 documentation](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-voices)

### MAI-Image-2

MAI-Image-2 in Foundry uses the MAI image API rather than the legacy Azure OpenAI images route.

The app supports two auth modes for MAI-Image-2:

- `api-key`: sends the `api-key` header.
- `azuredefault`: uses a tenant-aware Azure credential chain and requests a bearer token for `https://cognitiveservices.azure.com/.default`.

When `AZURE_TENANT_ID` is set, the app requests the token from that tenant explicitly. This is necessary when your local Azure sign-in defaults to a different tenant than the Foundry resource.

The API endpoint has the following form:

```text
https://<resource-name>.services.ai.azure.com/mai/v1/images/generations
```

If you only have a project endpoint such as `https://<resource-name>.services.ai.azure.com/api/projects/proj-default`, the app strips the `/api/projects/...` suffix automatically and calls the correct MAI image API URL.

Deploy via the Azure CLI:

```bash
az cognitiveservices account deployment create \
  --name <ACCOUNT_NAME> \
  --resource-group <RESOURCE_GROUP> \
  --deployment-name <DEPLOYMENT_NAME> \
  --model-name mai-image-2 \
  --model-format Microsoft \
  --model-version 2026-02-20 \
  --sku-name GlobalStandard \
  --sku-capacity 1
```

- [MAI-Image-2 documentation](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-mai)
