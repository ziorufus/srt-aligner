import os
import secrets
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from aligner import align_text_to_srt_advanced, _load_default_model

load_dotenv()

app = FastAPI(title="SRT Aligner API")
MODEL = _load_default_model()
AUTH_SCHEME = HTTPBearer(auto_error=False)
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN")

if not API_BEARER_TOKEN:
    raise RuntimeError("API_BEARER_TOKEN non impostato")


def require_bearer_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(AUTH_SCHEME),
) -> None:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    if credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid auth scheme")

    if not secrets.compare_digest(credentials.credentials, API_BEARER_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid bearer token")


@app.get("/health")
def healthcheck(_: None = Depends(require_bearer_token)) -> dict[str, str]:
    return {"status": "ok", "device": str(MODEL.device)}


@app.post("/align")
async def align(
    _: None = Depends(require_bearer_token),
    srt_file: UploadFile = File(...),
    translation_text: str | None = Form(default=None),
    translation_file: UploadFile | None = File(default=None),
) -> StreamingResponse:
    if not srt_file.filename or not srt_file.filename.lower().endswith(".srt"):
        raise HTTPException(status_code=400, detail="srt_file deve essere un file .srt")

    if translation_text is None and translation_file is None:
        raise HTTPException(
            status_code=400,
            detail="Fornisci translation_text oppure translation_file",
        )

    if translation_text is not None and translation_file is not None:
        raise HTTPException(
            status_code=400,
            detail="Fornisci solo uno tra translation_text e translation_file",
        )

    if translation_file is not None:
        raw_translation = await translation_file.read()
        try:
            translation_text = raw_translation.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail="translation_file deve essere UTF-8",
            ) from exc

    assert translation_text is not None
    translation_text = translation_text.strip()
    if not translation_text:
        raise HTTPException(status_code=400, detail="La traduzione e' vuota")

    srt_bytes = await srt_file.read()
    if not srt_bytes:
        raise HTTPException(status_code=400, detail="Il file SRT e' vuoto")

    try:
        srt_text = srt_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="srt_file deve essere UTF-8") from exc

    output_name = f"{Path(srt_file.filename).stem}.aligned.srt"

    try:
        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.srt"
            output_path = Path(tmpdir) / "output.srt"

            input_path.write_text(srt_text, encoding="utf-8")
            align_text_to_srt_advanced(
                srt_path=str(input_path),
                target_text=translation_text,
                output_path=str(output_path),
                model=MODEL,
            )

            content = output_path.read_text(encoding="utf-8")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return StreamingResponse(
        iter([content.encode("utf-8")]),
        media_type="application/x-subrip",
        headers={"Content-Disposition": f'attachment; filename="{output_name}"'},
    )
