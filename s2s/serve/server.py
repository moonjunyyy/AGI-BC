"""
FastAPI + WebSocket audio streaming server.

Endpoints:
    /ws/chat        — bidirectional audio streaming (PCM over WebSocket)
    /api/transcribe — POST audio file → text
    /api/generate   — POST audio file → audio
    /health         — GET healthcheck

Launch: uvicorn s2s.serve.server:app --host 0.0.0.0 --port 8998
"""
import asyncio
import io
import os
import queue
import threading
from typing import Optional

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request
from fastapi.responses import Response, JSONResponse

from ..pipeline.inference import StreamingInferencePipeline

app = FastAPI(title="S2S Server")

# Global model instance (set by startup or CLI)
_model: Optional[object] = None
_device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_model(model, device: str = None):
    """Set the global model for serving."""
    global _model, _device
    _model = model
    if device:
        _device = device


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None, "device": _device}


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    """Bidirectional PCM audio streaming over WebSocket.

    Client sends raw 16-bit LE PCM audio chunks.
    Server replies with raw 16-bit LE PCM audio + text JSON.
    """
    await websocket.accept()

    if _model is None:
        await websocket.send_json({"error": "model not loaded"})
        await websocket.close()
        return

    audio_queue: queue.Queue = queue.Queue()
    output_queue: queue.Queue = queue.Queue()

    pipeline = StreamingInferencePipeline(_model, device=_device)
    infer_thread = pipeline.run_async(audio_queue, output_queue)

    async def _send_outputs():
        while True:
            try:
                result = output_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue

            if result is None:
                break

            # Send text
            if result.get("text"):
                await websocket.send_json({"type": "text", "text": result["text"]})

            # Send audio as raw PCM bytes
            if result.get("audio") is not None:
                from ..utils.av import tensor_to_bytes
                pcm = tensor_to_bytes(result["audio"])
                await websocket.send_bytes(pcm)

    send_task = asyncio.create_task(_send_outputs())

    try:
        while True:
            data = await websocket.receive()
            if "bytes" in data:
                # Raw PCM chunk
                from ..utils.av import bytes_to_tensor
                tensor = bytes_to_tensor(data["bytes"])
                audio_queue.put(tensor)
            elif "text" in data:
                msg = data["text"]
                if msg == "END":
                    audio_queue.put(None)  # Signal end
                    break
    except WebSocketDisconnect:
        audio_queue.put(None)
    finally:
        await send_task
        await websocket.close()


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Transcribe uploaded audio file to text."""
    if _model is None:
        return JSONResponse({"error": "model not loaded"}, status_code=503)

    content = await file.read()
    # Save temp file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from ..utils.av import load_audio
        audio = load_audio(tmp_path).unsqueeze(0).to(_device)
        results = list(_model.generate_stream(iter([audio])))
        text = " ".join(r.get("text", "") for r in results)
        return JSONResponse({"text": text})
    finally:
        os.unlink(tmp_path)


@app.post("/api/generate")
async def generate(file: UploadFile = File(...)):
    """Generate audio response from uploaded audio file."""
    if _model is None:
        return JSONResponse({"error": "model not loaded"}, status_code=503)

    content = await file.read()
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from ..utils.av import load_audio, tensor_to_bytes
        audio = load_audio(tmp_path).unsqueeze(0).to(_device)
        results = list(_model.generate_stream(iter([audio])))
        audio_tensors = [r["audio"]
                         for r in results if r.get("audio") is not None]
        if not audio_tensors:
            return JSONResponse({"error": "no audio generated"}, status_code=500)
        combined = torch.cat(audio_tensors, dim=-1)
        pcm = tensor_to_bytes(combined)
        return Response(content=pcm, media_type="audio/pcm")
    finally:
        os.unlink(tmp_path)


@app.post("/api/eval")
async def api_eval(request: Request):
    """Run keyword Q&A evaluation using the loaded model for both roles.

    Body JSON:
        keyword           (required) word to describe and guess
        max_turns         (optional, default 20)
        describer_prompt  (optional) override default describer system prompt
        guesser_prompt    (optional) override default guesser system prompt

    Returns: KeywordQAResult as dict
        {keyword, turns, guessed, guessed_at_turn, transcript}
    """
    if _model is None:
        return JSONResponse({"error": "model not loaded"}, status_code=503)

    import dataclasses
    body = await request.json()
    keyword = body.get("keyword", "").strip()
    if not keyword:
        return JSONResponse({"error": "'keyword' is required"}, status_code=400)

    from ..pipeline.eval_dialogue import KeywordQAEvaluator, KeywordQAGoal

    goal = KeywordQAGoal(
        keyword=keyword,
        max_turns=int(body.get("max_turns", 20)),
        describer_prompt=body.get("describer_prompt", ""),
        guesser_prompt=body.get("guesser_prompt", ""),
    )
    loop = asyncio.get_event_loop()
    evaluator = KeywordQAEvaluator(_model, goal, device=_device)
    result = await loop.run_in_executor(None, evaluator.run)
    return dataclasses.asdict(result)


# Mount the SPA at GET /
from .webui import mount_ui  # noqa: E402
mount_ui(app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8998)
