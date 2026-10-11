"""Real-model HTTP/WS integration checks; no microphone or physical audio test."""
import argparse
import asyncio
import io
import json
import struct
import time
from pathlib import Path

import aiohttp
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
HEADER = struct.Struct("<IId")


async def run(base_url, output):
    report = {"scope": "real-model loopback HTTP/WebSocket, file PCM; no microphone/hardware latency", "checks": {}}
    mix_path = ROOT / "evaluation/data/user_recordings/mixture/Mix.wav"
    audio, rate = sf.read(mix_path, dtype="float32")
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    async with aiohttp.ClientSession() as client:
        async with client.get(base_url + "/api/health") as response:
            health = await response.json()
            assert health["status"] == "ready" and health["device"] == "mps"
            report["health"] = health
        for path in ["/", "/static/app.js", "/static/capture-worklet.js", "/static/style.css"]:
            async with client.get(base_url + path) as response:
                assert response.status == 200
                assert len(await response.read()) > 100
        async with client.get(base_url + "/api/health", headers={"Origin": "https://example.org"}) as response:
            assert response.status == 403
        async with client.get(base_url + "/api/profiles/../../etc/audio/speaker_01") as response:
            assert response.status in (400, 404)
        async with client.post(base_url + "/api/example-profile") as response:
            group = await response.json()
            assert group["calibration_seconds"] == 10 and len(group["profiles"]) == 3
        async with client.get(base_url + group["profiles"][0]["preview_url"]) as response:
            assert response.status == 200
            preview, sr = sf.read(io.BytesIO(await response.read()), dtype="float32")
            assert sr == 16000 and np.isfinite(preview).all()
        report["checks"].update({"static_assets": True, "origin_rejection": True, "path_rejection": True, "saved_profile_preview": True})
        async with client.post(base_url + "/api/calibration-upload", data=b"not a WAV") as response:
            assert response.status == 400, response.status
        short = io.BytesIO(); sf.write(short, mono[:10 * rate], rate, format="WAV", subtype="FLOAT")
        async with client.post(base_url + "/api/calibration-upload", data=short.getvalue()) as response:
            assert response.status == 400
        report["checks"]["short_and_invalid_upload_rejected"] = True
        # Two live runs prove selection composition and restart with actual weights.
        report["runs"] = {}
        for label, selected, mode, hop_ms in [("one_speaker_200ms", ["speaker_01"], "filtered", 200),
                                      ("one_speaker_300ms", ["speaker_01"], "filtered", 300),
                                      ("two_speakers_300ms", ["speaker_01", "speaker_03"], "filtered", 300),
                                      ("bypass_20ms", [], "bypass", 20)]:
            frames = rate * hop_ms // 1000
            hop_seconds = hop_ms / 1000
            async with client.ws_connect(base_url + "/ws") as ws:
                assert (await ws.receive_json())["event"] == "connected"
                await ws.send_json({"action": "start", "rate": rate, "group_id": group["id"], "selected": selected,
                                    "mode": mode, "hop_ms": 200 if mode == "bypass" else hop_ms})
                assert (await ws.receive_json())["event"] == "started"
                chunks, metrics, errors = [], [], []
                start = time.perf_counter()
                count = 50 if mode == "bypass" else 20
                first_output = None
                async def receive():
                    nonlocal first_output
                    while len(chunks) < count:
                        message = await ws.receive(timeout=15)
                        if message.type == aiohttp.WSMsgType.BINARY:
                            seq, size, captured = HEADER.unpack_from(message.data)
                            samples = np.frombuffer(message.data, dtype="<f4", offset=HEADER.size)
                            assert seq == len(chunks) and size == hop_ms * 16 and samples.shape == (hop_ms * 16,)
                            assert np.isfinite(samples).all() and np.max(np.abs(samples)) <= .900001
                            if first_output is None:
                                first_output = time.perf_counter() - start
                            chunks.append(samples.copy())
                        elif message.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(message.data)
                            if data["event"] == "error":
                                errors.append(data["message"]); return
                            if data["event"] == "metrics": metrics.append(data)
                        else:
                            raise AssertionError(f"Unexpected WebSocket close: {message.type}")
                receiver = asyncio.create_task(receive())
                for i in range(count):
                    if errors:
                        break
                    await asyncio.sleep(max(0, start + (i + 1) * hop_seconds - time.perf_counter()))
                    segment = mono[10 * rate + i * frames:10 * rate + (i + 1) * frames]
                    await ws.send_bytes(HEADER.pack(i, frames, start * 1000 + i * hop_ms) + segment.astype("<f4").tobytes())
                await receiver
                report["runs"][label] = {"returned_blocks": len(chunks), "sent_blocks": count,
                    "hop_ms": hop_ms, "completed_without_overload": not errors and len(chunks) == count,
                    "first_output_seconds": first_output, "incremental_before_eof": first_output is not None and first_output < count * hop_seconds,
                    "processing_ms": {"p50": float(np.median([m["processing_ms"] for m in metrics])),
                                      "p95": float(np.percentile([m["processing_ms"] for m in metrics], 95))},
                    "max_server_pending": max([m["pending"] for m in metrics], default=0), "errors": errors}
                report["runs"][label]["stage_metrics"] = metrics
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(report, indent=2) + "\n")
                if label == "one_speaker_200ms" and errors:
                    assert all("antrean" in error for error in errors), errors
                    report["checks"]["200ms_overload_stopped_visibly"] = True
                else:
                    assert not errors and len(chunks) == count, report["runs"][label]
                assert first_output < count * hop_seconds
                await ws.send_json({"action":"stop"})
                assert (await ws.receive_json())["event"] == "stopped"
            await asyncio.sleep(.1)
        async with client.ws_connect(base_url + "/ws") as ws:
            await ws.receive_json()
            await ws.send_json({"action":"start", "rate":rate,"group_id":group["id"],"selected":[]})
            assert (await ws.receive_json())["event"] == "error"
            await ws.send_json({"action":"start", "rate":rate,"mode":"bypass"})
            assert (await ws.receive_json())["event"] == "started"
            bad = HEADER.pack(2, rate // 50, 1.) + np.zeros(rate // 50, dtype="<f4").tobytes()
            await ws.send_bytes(bad)
            assert (await ws.receive_json())["event"] == "error"
        await asyncio.sleep(.1)
        report["checks"].update({"empty_selection_rejected": True, "sequence_gap_rejected": True, "stop_and_restart": True})
        async with client.ws_connect(base_url + "/ws") as ws:
            await ws.receive_json()
            async with client.ws_connect(base_url + "/ws") as extra:
                assert (await extra.receive_json())["event"] == "error"
            await ws.send_json({"action":"start","rate":rate,"group_id":group["id"],"selected":["speaker_01"],"hop_ms":300})
            assert (await ws.receive_json())["event"] == "started"
            frames = rate * 300 // 1000
            for i in range(10):
                await ws.send_bytes(HEADER.pack(i, frames, i * 300.) + mono[10 * rate:10 * rate + frames].astype("<f4").tobytes())
            while True:
                message = await ws.receive(timeout=15)
                if message.type == aiohttp.WSMsgType.TEXT and json.loads(message.data)["event"] == "error":
                    assert "antrean" in json.loads(message.data)["message"]
                    break
        await asyncio.sleep(.2)
        report["checks"].update({"single_active_client":True,"burst_overload_guard":True})
        # Exact-duration calibration fixture: repeat the existing first 10s three
        # times, not a new natural recording. This validates the capture boundary/API.
        fixture = np.tile(mono[:10 * rate], 3)
        frames = rate // 5
        async with client.ws_connect(base_url + "/ws") as ws:
            await ws.receive_json()
            await ws.send_json({"action":"calibrate","rate":rate})
            assert (await ws.receive_json())["event"] == "calibrating"
            for i in range(150):
                await ws.send_bytes(HEADER.pack(i, frames, i * 200.) + fixture[i * frames:(i + 1) * frames].astype("<f4").tobytes())
            progress, ready_group = [], None
            while ready_group is None:
                data = await ws.receive_json(timeout=90)
                assert data["event"] != "error", data
                if data["event"] == "calibration_progress": progress.append(data["seconds"])
                if data["event"] == "profiles_ready": ready_group = data["group"]
            assert progress[-1] == 30 and len(progress) == 150
            assert ready_group["calibration_seconds"] == 30 and len(ready_group["profiles"]) == 3
            await ws.send_json({"action":"stop"}); await ws.receive_json()
        await asyncio.sleep(.1)
        # Rename only this test-created profile, then delete only the disposable fixture.
        async with client.patch(base_url + f"/api/profiles/{ready_group['id']}", json={"speaker_id":"speaker_01","name":"Verification fixture"}) as response:
            renamed = await response.json(); assert renamed["profiles"][0]["name"] == "Verification fixture"
        async with client.delete(base_url + f"/api/profiles/{ready_group['id']}") as response:
            assert response.status == 200
        report["checks"].update({"30s_calibration_sample_boundary": True, "calibration_fixture_is_repeated_audio_not_live_mic": True,
                                  "real_model_profiles_created": True, "rename_and_delete_test_owned_group": True})
    report["hardware_microphone_and_airpods_verified"] = False
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"checks": report["checks"], "runs": {
        name: {key: value for key, value in run.items() if key != "stage_metrics"}
        for name, run in report["runs"].items()}, "hardware_microphone_and_airpods_verified": False}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "verification.json")
    args = parser.parse_args()
    asyncio.run(run(args.url, args.output))
