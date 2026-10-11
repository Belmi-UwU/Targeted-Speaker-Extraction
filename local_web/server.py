"""Loopback-only TSE lab. Requirements: ../PRD.md, current implementation contract."""
import argparse
import asyncio
import contextlib
import io
import json
import os
import re
import shutil
import struct
import sys
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

import numpy as np
import soundfile as sf
import torch
from aiohttp import WSMsgType, web
from speechbrain.inference.classifiers import EncoderClassifier

from experiment_auto_enrollment import build_profiles, load_profiles, write_json
from experiment_mps_sepformer import HybridMatcher, load_separator
from experiment_rolling_context import RollingSeparator
from run_user_recordings import resample, sha256
from simulate_realtime import WindowSeparator

STATIC = Path(__file__).parent / "static"
DATA = ROOT / "evaluation/data/user_recordings/local_web"
EXAMPLE = ROOT / "evaluation/data/user_recordings/auto_enrollment/2026-10-09_mix_first10s/profiles"
MIX = ROOT / "evaluation/data/user_recordings/mixture/Mix.wav"
HEADER = struct.Struct("<IId")  # sequence, frame count, browser capture-start milliseconds
RATES = {16000, 32000, 44100, 48000, 96000}
GROUP_ID = re.compile(r"^(?:[a-f0-9]{32}|example10)$")


def group_path(group_id):
    if not isinstance(group_id, str) or not GROUP_ID.fullmatch(group_id):
        raise web.HTTPBadRequest(text="ID profil tidak valid")
    path = DATA / "profiles" / group_id
    if not path.is_dir():
        raise web.HTTPNotFound(text="Profil tidak ditemukan")
    return path


def public_group(group_id):
    record, _ = load_profiles(group_path(group_id))
    return {"id": group_id, "calibration_seconds": record["calibration_seconds"],
            "creation_seconds": record["profile_creation_seconds"],
            "possible_duplicate_pairs": record["possible_duplicate_pairs"],
            "profiles": [{"id": p["speaker_id"], "name": p["display_name"],
                          "energy_active_seconds": p["energy_active_seconds"],
                          "preview_url": f"/api/profiles/{group_id}/audio/{p['speaker_id']}"}
                         for p in record["profiles"]]}


class Engine:
    def __init__(self, device):
        self.device = device
        self.status, self.error = "loading", None
        self.profile_busy = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tse-model")
        self.session = None

    async def call(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, fn, *args)

    def initialize(self):
        torch.set_num_threads(4)
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("GPU Metal tidak tersedia. Jalankan dari Terminal dengan akses GPU, atau pilih --device cpu untuk debugging.")
        self.separator = load_separator(self.device)
        checkpoint = ROOT / "evaluation/ecapa_checkpoint"
        self.matcher = EncoderClassifier.from_hparams(
            source=str(checkpoint), savedir=str(ROOT / "evaluation/ecapa_loaded"),
            overrides={"pretrained_path": str(checkpoint)}, run_opts={"device": "cpu"}).eval()
        self.live_matcher = HybridMatcher(self.matcher) if self.device == "mps" else self.matcher
        # Browser input has been resampled by its AudioContext to the configured rate.
        # Warm at 48k; startup lengths have exactly the same 8k model input at other supported rates.
        self.make_processor(np.eye(3, 192, dtype="float32"), ["speaker_01", "speaker_02", "speaker_03"], ["speaker_01"], warm=True)

    def make_processor(self, vectors, names, selected, warm=False):
        base = WindowSeparator(self.separator, self.live_matcher, vectors, names=names)
        base.selections = [tuple(selected)]
        if warm:
            for ms in [200, 300, 400, 600, 800, 900, 1000, 1200, 1400, 1500, 1600, 1700, 1800]:
                base(np.zeros(ms * 48, dtype="float32"), 48000, ms / 1000)
            if self.device == "mps":
                torch.mps.synchronize()
        return RollingSeparator(base, 1500)

    def create_group(self, wave, rate, source_kind):
        if len(list((DATA / "profiles").glob("*/profiles.json"))) >= 20:
            raise ValueError("Batas 20 kalibrasi tersimpan tercapai. Hapus profil website yang tidak dipakai.")
        if float(np.mean(np.abs(wave) >= .999)) > .02 or float(np.max(np.abs(wave))) > 2:
            raise ValueError("Audio kalibrasi clipping. Kurangi level input dan rekam ulang.")
        group_id = uuid.uuid4().hex
        directory = DATA / "profiles" / group_id
        directory.mkdir(parents=True)
        self.profile_busy = True
        try:
            source = directory / "calibration.wav"
            sf.write(source, wave, rate, subtype="FLOAT")
            # build_profiles needs a fresh directory; promote its outputs after success.
            record, _ = build_profiles(wave, rate, self.separator, self.matcher, directory / "generated", {
                "path": str(source), "sha256": sha256(source), "sample_rate_hz": rate,
                "start_frame": 0, "end_frame": len(wave), "kind": source_kind})
            for file in (directory / "generated").iterdir():
                file.rename(directory / file.name)
            (directory / "generated").rmdir()
            return group_id
        except Exception:
            shutil.rmtree(directory)
            raise
        finally:
            self.profile_busy = False

    def infer(self, processor, wave, rate, mode, hop_ms):
        begin = time.perf_counter()
        if self.device == "mps" and mode != "bypass":
            torch.mps.synchronize()
        if mode == "bypass":
            result, details = resample(wave, rate, 16000), {}
        else:
            outputs, details = processor(wave, rate, hop_ms / 1000)
            result = next(iter(outputs.values()))
        if self.device == "mps" and mode != "bypass":
            torch.mps.synchronize()
        if result.shape != (hop_ms * 16,) or not np.isfinite(result).all():
            raise ValueError("Hasil audio tidak valid")
        peak = float(np.max(np.abs(result)))
        gain = min(1., .9 / max(peak, 1e-12))
        result = np.asarray(result * gain, dtype="<f4")
        return result, {"processing_ms": (time.perf_counter() - begin) * 1000,
                        "separator_ms": details.get("separator_seconds", 0) * 1000,
                        "resampling_ms": details.get("resampling_before_separator_seconds", 0) * 1000,
                        "matching_selection_ms": details.get("gain_matching_selection_seconds", 0) * 1000,
                        "limiter_gain": gain,
                        "history_ms": details.get("available_history_seconds", 0) * 1000,
                        "matched_stream_by_speaker": details.get("matched_stream_by_speaker", {})}


class Session:
    def __init__(self, engine, ws):
        self.engine, self.ws = engine, ws
        self.phase = "idle"
        self.generation = 0
        self.consumer = None
        self.rate, self.next_seq = None, 0
        self.calibration = []
        self.calibration_frames = 0
        self.group_id = None
        self.timings = deque(maxlen=9000)

    async def send(self, event, **data):
        if not self.ws.closed:
            await self.ws.send_json({"event": event, **data})

    async def stop(self, notify=True):
        self.generation += 1
        self.phase = "idle"
        self.calibration.clear()
        self.calibration_frames = 0
        task, self.consumer = self.consumer, None
        if task and task is not asyncio.current_task():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if hasattr(self, "packets"):
            while not self.packets.empty():
                self.packets.get_nowait()
        if notify:
            await self.send("stopped")

    async def fail(self, message):
        await self.stop(notify=False)
        await self.send("error", message=message)

    async def control(self, data):
        action = data.get("action")
        if action == "stop":
            await self.stop()
            return
        if action not in ("calibrate", "start"):
            raise ValueError("Perintah tidak dikenal")
        if self.engine.status != "ready":
            raise ValueError("Model belum siap")
        if self.engine.profile_busy:
            raise ValueError("Pembuatan profil sedang berjalan. Tunggu sampai selesai.")
        await self.stop(notify=False)
        rate = data.get("rate")
        if type(rate) is not int or rate not in RATES:
            raise ValueError("Sample rate tidak didukung")
        self.rate, self.next_seq = rate, 0
        generation = self.generation
        if action == "calibrate":
            self.hop_ms = 200
            self.phase = "calibrating"
            await self.send("calibrating", seconds=30, rate=rate)
            return
        mode = data.get("mode", "filtered")
        if mode not in ("filtered", "bypass"):
            raise ValueError("Mode tidak valid")
        self.mode, self.group_id = mode, data.get("group_id")
        hop_ms = data.get("hop_ms", 200)
        if type(hop_ms) is not int or hop_ms not in (200, 300):
            raise ValueError("Hop filtering harus 200 atau 300 ms")
        self.hop_ms = 20 if mode == "bypass" else hop_ms
        self.processor = None
        if mode == "filtered":
            record, vectors = load_profiles(group_path(self.group_id))
            selected = data.get("selected", [])
            if (not isinstance(selected, list) or not selected or not all(isinstance(s, str) for s in selected)
                    or len(set(selected)) != len(selected) or not set(selected) <= set(record["speaker_order"])):
                raise ValueError("Pilih minimal satu profil speaker yang valid")
            self.processor = await self.engine.call(self.engine.make_processor, vectors, record["speaker_order"], selected)
        self.packets = asyncio.Queue(maxsize=2)
        self.timings.clear()
        self.phase = "streaming"
        self.consumer = asyncio.create_task(self.consume(generation))
        await self.send("started", mode=mode, device=self.engine.device, context_ms=1500, hop_ms=self.hop_ms)

    async def packet(self, payload):
        if self.phase not in ("calibrating", "streaming"):
            return  # ignore in-flight packets after stop/calibration boundary
        if len(payload) < HEADER.size:
            raise ValueError("Paket audio terlalu pendek")
        seq, size, captured_ms = HEADER.unpack_from(payload)
        if (seq != self.next_seq or size != self.rate * self.hop_ms // 1000 or len(payload) != HEADER.size + size * 4
                or not np.isfinite(captured_ms) or captured_ms < 0):
            raise ValueError("Urutan/ukuran paket audio tidak valid; mulai ulang sesi")
        self.next_seq += 1
        wave = np.frombuffer(payload, dtype="<f4", offset=HEADER.size).copy()
        if not np.isfinite(wave).all() or np.max(np.abs(wave)) > 2:
            raise ValueError("PCM harus finite dan dalam rentang input")
        if self.phase == "calibrating":
            self.calibration.append(wave)
            self.calibration_frames += size
            await self.send("calibration_progress", seconds=self.calibration_frames / self.rate)
            if self.calibration_frames == 30 * self.rate:
                self.phase = "processing_profiles"
                wave = np.concatenate(self.calibration)
                self.calibration.clear()
                await self.send("processing_profiles")
                self.consumer = asyncio.create_task(self.finish_calibration(wave, self.generation))
        else:
            try:
                self.packets.put_nowait((seq, captured_ms, wave, time.perf_counter()))
            except asyncio.QueueFull:
                await self.fail("Pemrosesan tertinggal: antrean mencapai batas. Sesi dihentikan agar delay tidak terus bertambah. Coba Bypass atau tutup aplikasi berat.")

    async def finish_calibration(self, wave, generation):
        try:
            group_id = await self.engine.call(self.engine.create_group, wave, self.rate, "browser_30s_calibration")
            if generation == self.generation:
                self.phase = "idle"
                await self.send("profiles_ready", group=public_group(group_id))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self.fail(f"Kalibrasi gagal: {error}")

    async def consume(self, generation):
        try:
            while self.phase == "streaming" and generation == self.generation:
                seq, captured_ms, wave, enqueued = await self.packets.get()
                queue_ms = (time.perf_counter() - enqueued) * 1000
                result, metrics = await self.engine.call(self.engine.infer, self.processor, wave, self.rate, self.mode, self.hop_ms)
                if generation != self.generation or self.phase != "streaming":
                    return
                self.timings.append(metrics["processing_ms"])
                recent = np.asarray(list(self.timings)[-300:])
                await self.send("metrics", sequence=seq, queue_ms=queue_ms, pending=self.packets.qsize(),
                                processing_p50_ms=float(np.median(recent)), processing_p95_ms=float(np.percentile(recent, 95)), **metrics)
                await self.ws.send_bytes(HEADER.pack(seq, len(result), captured_ms) + result.tobytes())
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self.fail(f"Filtering gagal: {error}")


@web.middleware
async def local_only(request, handler):
    origin = request.headers.get("Origin")
    if request.host not in (f"127.0.0.1:{request.app['port']}", f"localhost:{request.app['port']}"):
        raise web.HTTPForbidden(text="Akses hanya melalui localhost")
    if origin and origin not in (f"http://127.0.0.1:{request.app['port']}", f"http://localhost:{request.app['port']}"):
        raise web.HTTPForbidden(text="Origin tidak diizinkan")
    response = await handler(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    return response


async def health(request):
    engine = request.app["engine"]
    return web.json_response({"status": engine.status, "error": engine.error, "device": engine.device,
        "profile_busy": engine.profile_busy,
        "context_ms": 1500, "hop_ms": 200, "calibration_seconds": 30,
        "example_available": EXAMPLE.is_dir() and MIX.is_file(), "local_only": True})


async def profiles(request):
    groups = []
    for directory in sorted((DATA / "profiles").glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if GROUP_ID.fullmatch(directory.name) and (directory / "profiles.json").is_file():
            try:
                groups.append(public_group(directory.name))
            except (ValueError, KeyError, OSError):
                pass  # invalid/partial bundles are never offered
    return web.json_response({"groups": groups})


async def import_example(request):
    destination = DATA / "profiles/example10"
    if not EXAMPLE.is_dir():
        raise web.HTTPNotFound(text="Profil demo belum tersedia")
    if not destination.exists():
        shutil.copytree(EXAMPLE, destination)
    return web.json_response(public_group("example10"))


async def profile_audio(request):
    record, _ = load_profiles(group_path(request.match_info["group"]))
    profile = next((p for p in record["profiles"] if p["speaker_id"] == request.match_info["speaker"]), None)
    if not profile:
        raise web.HTTPNotFound()
    return web.FileResponse(group_path(request.match_info["group"]) / profile["enrollment_file"])


async def rename_profile(request):
    directory = group_path(request.match_info["group"])
    record, _ = load_profiles(directory)
    data = await request.json()
    profile = next((p for p in record["profiles"] if p["speaker_id"] == data.get("speaker_id")), None)
    name = data.get("name")
    if not profile or not isinstance(name, str) or not 1 <= len(name.strip()) <= 50:
        raise web.HTTPBadRequest(text="Nama harus 1–50 karakter")
    profile["display_name"] = name.strip()
    write_json(directory / "profiles.json", record)
    return web.json_response(public_group(request.match_info["group"]))


async def delete_profile(request):
    engine = request.app["engine"]
    if engine.session and engine.session.phase != "idle":
        raise web.HTTPConflict(text="Hentikan sesi sebelum menghapus profil")
    shutil.rmtree(group_path(request.match_info["group"]))
    return web.json_response({"deleted": True})


async def upload_calibration(request):
    engine = request.app["engine"]
    if engine.status != "ready" or engine.profile_busy or (engine.session and engine.session.phase != "idle"):
        raise web.HTTPConflict(text="Model belum siap atau sesi masih aktif")
    engine.profile_busy = True
    try:
        payload = await request.read()
        wave, rate = sf.read(io.BytesIO(payload), dtype="float32")
        if wave.ndim == 2:
            wave = wave.mean(axis=1)
        if wave.ndim != 1 or rate not in RATES or len(wave) < rate * 30 or not np.isfinite(wave).all():
            raise ValueError("Gunakan WAV minimal 30 detik dengan sample rate yang didukung")
        group_id = await engine.call(engine.create_group, wave[:rate * 30].copy(), rate, "uploaded_30s_calibration")
        return web.json_response(public_group(group_id))
    except (RuntimeError, ValueError) as error:
        raise web.HTTPBadRequest(text=str(error))
    finally:
        engine.profile_busy = False


async def websocket(request):
    engine = request.app["engine"]
    ws = web.WebSocketResponse(max_msg_size=150000, heartbeat=15)
    await ws.prepare(request)
    if engine.session:
        await ws.send_json({"event": "error", "message": "Tab lain sedang memakai audio. Tutup/hentikan koneksi tab tersebut."})
        await ws.close()
        return ws
    session = Session(engine, ws)
    engine.session = session
    try:
        await session.send("connected")
        async for msg in ws:
            try:
                if msg.type == WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if not isinstance(data, dict):
                        raise ValueError("Perintah harus berupa objek JSON")
                    await session.control(data)
                elif msg.type == WSMsgType.BINARY:
                    await session.packet(msg.data)
                elif msg.type == WSMsgType.ERROR:
                    break
            except (ValueError, KeyError, web.HTTPException) as error:
                await session.fail(str(error))
    finally:
        await session.stop(notify=False)
        engine.session = None
    return ws


async def initialize_engine(app):
    async def initialize():
        engine = app["engine"]
        try:
            await engine.call(engine.initialize)
            engine.status = "ready"
            print(f"TSE ready on {engine.device}: http://127.0.0.1:{app['port']}", flush=True)
        except Exception as error:
            engine.status, engine.error = "error", str(error)
            print(f"Model error: {error}", flush=True)
    app["init_task"] = asyncio.create_task(initialize())


async def cleanup(app):
    if app["engine"].session:
        await app["engine"].session.stop(notify=False)
    await app["init_task"]
    app["engine"].executor.shutdown(wait=True, cancel_futures=True)


def make_app(device="mps", port=8765):
    (DATA / "profiles").mkdir(parents=True, exist_ok=True)
    app = web.Application(client_max_size=30 * 1024 * 1024, middlewares=[local_only])
    app["engine"], app["port"] = Engine(device), port
    app.router.add_get("/api/health", health)
    app.router.add_get("/api/profiles", profiles)
    app.router.add_post("/api/example-profile", import_example)
    app.router.add_post("/api/calibration-upload", upload_calibration)
    app.router.add_patch("/api/profiles/{group}", rename_profile)
    app.router.add_delete("/api/profiles/{group}", delete_profile)
    app.router.add_get("/api/profiles/{group}/audio/{speaker}", profile_audio)
    app.router.add_get("/api/demo.wav", lambda _: web.FileResponse(MIX))
    app.router.add_get("/ws", websocket)
    app.router.add_get("/", lambda _: web.FileResponse(STATIC / "index.html"))
    app.router.add_static("/static/", STATIC, show_index=False)
    app.on_startup.append(initialize_engine)
    app.on_cleanup.append(cleanup)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Port must be 1024–65535")
    web.run_app(make_app(args.device, args.port), host="127.0.0.1", port=args.port, access_log=None)
