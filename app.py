from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from vision.engine import VisionEngine

ROOT = Path(__file__).resolve().parent
engine = VisionEngine()


@asynccontextmanager
async def lifespan(app):
    engine.start()
    try:
        yield
    finally:
        await asyncio.to_thread(engine.close)


app = FastAPI(title="DepthSight · D435i 视觉控制台", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


def valid_origin(origin, host):
    return origin in (f"http://{host}", f"https://{host}")


@app.middleware("http")
async def local_console(request: Request, call_next):
    if request.method in {"POST", "PATCH", "DELETE"}:
        if request.headers.get("x-console-request") != "1":
            return Response("Missing console request header", status_code=403)
        origin = request.headers.get("origin")
        if origin and not valid_origin(origin, request.headers.get("host")):
            return Response("Origin rejected", status_code=403)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


class SettingsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    detection_enabled: bool | None = None
    confidence: float | None = Field(default=None, ge=.1, le=.95, allow_inf_nan=False)
    near_m: float | None = Field(default=None, ge=.1, le=10, allow_inf_nan=False)
    far_m: float | None = Field(default=None, ge=.3, le=20, allow_inf_nan=False)
    inference_size: Literal[320, 480, 640] | None = None


class NotificationsPatch(BaseModel):
    model_config = ConfigDict(extra='forbid')
    enabled: bool | None = None
    cooldown_seconds: int | None = Field(default=None, ge=15, le=3600, strict=True)
    threshold: float | None = Field(default=None, ge=.1, le=.99, allow_inf_nan=False)
    min_visible_seconds: float | None = Field(default=None, ge=.1, le=3600, allow_inf_nan=False)
    webhook: str | None = Field(default=None, max_length=2048)


@app.patch('/api/notifications')
def configure_notifications(payload: NotificationsPatch):
    values = payload.model_dump(exclude_unset=True)
    if any(value is None for value in values.values()):
        raise HTTPException(422, '推送设置不能为 null')
    try:
        return engine.notifications.configure(values)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    except OSError:
        raise HTTPException(422, '本地配置无法保存，请检查文件权限') from None


@app.post('/api/notifications/test')
def test_notification():
    try:
        return engine.notifications.send_test()
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/status")
def status():
    return engine.status()


@app.patch("/api/settings")
def settings(payload: SettingsPatch):
    values = payload.model_dump(exclude_unset=True)
    if any(v is None for v in values.values()):
        raise HTTPException(422, "设置不能为 null")
    if "inference_size" in values and values["inference_size"] not in (320, 480, 640):
        raise HTTPException(422, "推理尺寸应为 320、480 或 640")
    try:
        return engine.configure(values)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/camera/{action}")
def camera_action(action: str):
    if action not in ("start", "stop"):
        raise HTTPException(404, "未知相机操作")
    return engine.set_active(action == "start")


@app.post("/api/model/reload")
def reload_model():
    with engine.lock:
        engine.model_retry = True
    return {"queued": True}


@app.get("/api/depth")
def depth(x: int, y: int):
    try:
        return engine.sample_depth(x, y)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/snapshot")
def snapshot():
    try:
        content = engine.export_snapshot()
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    filename = "depthsight-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".zip"
    return Response(content, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.websocket("/ws")
async def stream(websocket: WebSocket):
    if not valid_origin(websocket.headers.get("origin"), websocket.headers.get("host")):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    last_seq, last_status = -1, 0.
    try:
        while True:
            seq, packet = engine.get_packet()
            if packet is not None and seq != last_seq:
                await asyncio.wait_for(websocket.send_bytes(packet), timeout=3)
                last_seq = seq
            now = asyncio.get_running_loop().time()
            if now - last_status >= 1:
                await asyncio.wait_for(websocket.send_json({"type": "status", **engine.status()}), timeout=3)
                last_status = now
            await asyncio.sleep(.015)
    except (WebSocketDisconnect, RuntimeError, asyncio.TimeoutError):
        pass


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
