# image-describe

A simple web app that uses a vision model to produce character-focused descriptions of uploaded images — extracting people, their physical attributes, clothing, and accessories.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- An API key for a vision-capable model (OpenAI, Anthropic, or any OpenAI-compatible provider)

## Setup

```bash
# 1. Clone / navigate to the project directory
cd image-describe

# 2. Run the server (uv handles the venv and dependencies automatically)
uv run app.py
```

The app starts at **http://localhost:5000** by default.

To use a different port:

```bash
PORT=8080 uv run app.py
```

To listen on all interfaces:

```bash
HOST=0.0.0.0 PORT=5000 uv run app.py
```

## Usage

1. Open http://localhost:5000 in your browser.
2. Fill in the **API Configuration** panel:
   - **API Key** — your provider's secret key (stored only in your browser's localStorage, never sent anywhere except your chosen API endpoint)
   - **Base URL** — leave blank for OpenAI; set to your provider's base URL for other services (e.g. `https://api.anthropic.com/v1`, a local Ollama instance, etc.)
   - **Model** — the vision model name (e.g. `gpt-4o`, `claude-opus-4-5`, `llava`)
3. Upload an image (PNG, JPG, JPEG, GIF, or WEBP — max 20 MB).
4. Click **Describe Image**.

The app returns a detailed description focused on:
- Number of characters/people
- Physical appearance (age, build, facial features, hair, skin tone)
- Clothing (each garment, color, style)
- Accessories (jewelry, glasses, bags, hats)
- Expression, pose, and distinguishing features

## Supported Providers

Any provider that exposes an OpenAI-compatible `/chat/completions` endpoint with vision support works, including:

| Provider | Base URL | Example model |
|----------|----------|---------------|
| OpenAI | *(leave blank)* | `gpt-4o` |
| Anthropic | `https://api.anthropic.com/v1` | `claude-opus-4-5` |
| Ollama (local) | `http://localhost:11434/v1` | `llava` |
| OpenRouter | `https://openrouter.ai/api/v1` | `openai/gpt-4o` |

## Docker

Build and run with Docker Compose:

```bash
docker compose up --build
```

The container runs the app behind `gunicorn` on port `5000`. To expose a different host port:

```bash
PORT=8080 docker compose up --build -d
```

To build and run without Compose:

```bash
docker build -t image-describe .
docker run --rm -p 5000:5000 image-describe
```
