# Aligner

Python service for realigning an SRT file to a free-form translated text while preserving the original timestamps.

The project provides:

- a local alignment library in [aligner.py](/Users/alessio/aligner/aligner.py)
- a FastAPI API in [server.py](/Users/alessio/aligner/server.py), protected with a Bearer token

## How it works

The algorithm:

- loads the source SRT file
- segments the target text using generic, language-agnostic heuristics
- uses a multilingual `sentence-transformers` model to compare segments
- performs monotonic many-to-many alignment
- distributes the target text across the original cues and produces a new SRT

## Requirements

- Python 3.12 recommended
- an active virtual environment, or explicit use of `env/bin/python`

## Installation

```bash
python3 -m venv env
env/bin/pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and set the values:

```bash
cp .env.example .env
```

Available variables:

- `API_BEARER_TOKEN`: token required to authenticate with the API
- `MODEL_NAME`: `sentence-transformers` model to use for alignment

The same variables can also be passed as standard process environment variables.

## Starting the server

```bash
env/bin/uvicorn server:app --reload
```

On startup, the service prints the device used for the model:

```text
Model loaded on device: cpu
```

The device is selected in this order:

- `mps` if available
- otherwise `cuda`
- otherwise `cpu`

## Endpoints

### `GET /health`

Required header:

```text
Authorization: Bearer <token>
```

Example:

```bash
curl http://127.0.0.1:8000/health \
  -H "Authorization: Bearer your-token"
```

### `POST /align`

Required header:

```text
Authorization: Bearer <token>
```

`multipart/form-data` parameters:

- `srt_file`: required `.srt` file
- `translation_text`: optional plain translated text
- `translation_file`: optional UTF-8 text file containing the translation

You must provide only one of `translation_text` or `translation_file`.

Example using a translation file:

```bash
curl -X POST http://127.0.0.1:8000/align \
  -H "Authorization: Bearer your-token" \
  -F "srt_file=@input.srt" \
  -F "translation_file=@translation.txt" \
  -o output.srt
```

Example using inline text:

```bash
curl -X POST http://127.0.0.1:8000/align \
  -H "Authorization: Bearer your-token" \
  -F "srt_file=@input.srt" \
  -F "translation_text=$(cat translation.txt)" \
  -o output.srt
```

## Library usage

You can use the `align_text_to_srt_advanced` function directly:

```python
from aligner import align_text_to_srt_advanced

target_text = open("translation.txt", "r", encoding="utf-8").read()

align_text_to_srt_advanced(
    srt_path="input.srt",
    target_text=target_text,
    output_path="output.srt",
)
```

## Notes

- The API expects SRT files and translation text encoded in UTF-8
- The model is loaded at server startup, not on every request
- If the model is not available locally, `sentence-transformers` will try to download it
