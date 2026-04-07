# ai-mai-models

A Streamlit playground for testing the three new Microsoft Foundry MAI models:

| Model | Capability |
|---|---|
| **MAI-Transcribe-1** | Speech-to-text – transcribes audio in 25+ languages |
| **MAI-Voice-1** | Text-to-speech – expressive, natural neural voices |
| **MAI-Image-2** | Text-to-image – high-quality diffusion-based image generation |

> Learn more: [Microsoft Foundry announcement](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-mai-transcribe-1-mai-voice-1-and-mai-image-2-in-microsoft-foundry/4507787)

## Screenshots

![Playground tabs](https://i.imgur.com/placeholder.png)

## Prerequisites

- Python 3.9+
- An **Azure Speech resource** (for MAI-Transcribe-1 and MAI-Voice-1)
  - Key and region available from the Azure portal
- A **Microsoft Foundry / Azure OpenAI resource** with **MAI-Image-2 deployed** (for MAI-Image-2)
  - Endpoint, API key, and deployment name available from the Azure portal

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
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL (MAI-Image-2) |
| `AZURE_OPENAI_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_DEPLOYMENT` | Your MAI-Image-2 deployment name |
| `AZURE_OPENAI_API_VERSION` | API version, e.g. `2024-02-15-preview` |

You can also enter credentials directly in the **sidebar** of the running app – no `.env` file required.

## Running the playground

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## How to deploy the models

### MAI-Transcribe-1 & MAI-Voice-1

Both models are available through the **Azure Speech Service**. Create a Speech resource in the [Azure portal](https://portal.azure.com) and note the **key** and **region**.

- [MAI-Transcribe-1 documentation](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-transcribe)
- [MAI-Voice-1 documentation](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-voices)

### MAI-Image-2

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
