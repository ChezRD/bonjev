"""Run one Jev Flight Lab flight against the backend, print a rich JSON RESULT.

Adds p50/p95/max decision latency, stuck count (interventions), track progress
(checkpoints) and efficiency over the stock flight_check.py.

Usage: uv run --with websockets python flight_probe.py --port 8010 [--barrier]
"""
import argparse, asyncio, json, time
import httpx
from websockets.asyncio.client import connect
from backend.simulation import Snapshot


async def run(port: int, barrier: bool) -> None:
    r: dict = {"flight": "failed", "error": None, "decisions": 0, "stuck": None,
               "checkpoint_index": None, "elapsed_s": None, "distance_m": None,
               "p50_latency_ms": None, "p95_latency_ms": None, "max_latency_ms": None,
               "eff_speed_mps": None}
    try:
        async with httpx.AsyncClient() as client:
            h = await client.get(f"http://127.0.0.1:{port}/api/health"); h.raise_for_status()
            r["health_model"] = h.json().get("model")
        async with connect(f"ws://127.0.0.1:{port}/ws/flight") as socket:
            init = Snapshot.model_validate_json(await socket.recv())
            if init.status != "ready":
                r["error"] = f"initial status {init.status}"
            if barrier:
                await socket.send(json.dumps({"action": "obstacle"}))
            await socket.send(json.dumps({"action": "start"}))
            deadline = time.monotonic() + 300
            seq = 0
            lat: list[float] = []
            async with asyncio.timeout(305):
                async for message in socket:
                    snap = Snapshot.model_validate_json(message)
                    if snap.decision and snap.decision.sequence != seq:
                        seq = snap.decision.sequence
                        lat.append(snap.decision.latency_ms)
                    r["stuck"] = snap.interventions
                    r["checkpoint_index"] = snap.checkpoint_index
                    if snap.error:
                        r["error"] = snap.error; break
                    if snap.status == "complete":
                        lat.sort()
                        r.update({
                            "flight": "complete", "decisions": seq, "elapsed_s": snap.elapsed,
                            "distance_m": snap.distance_flown, "stuck": snap.interventions,
                            "checkpoint_index": snap.checkpoint_index,
                            "p50_latency_ms": lat[len(lat)//2] if lat else None,
                            "p95_latency_ms": lat[min(len(lat)-1, int(0.95*len(lat)))] if lat else None,
                            "max_latency_ms": lat[-1] if lat else None,
                            "eff_speed_mps": (snap.distance_flown / snap.elapsed) if snap.elapsed else None,
                        })
                        break
                    if time.monotonic() > deadline:
                        r["error"] = "timeout"; break
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
    print("RESULT " + json.dumps(r), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--barrier", action="store_true")
    a = p.parse_args()
    asyncio.run(run(a.port, a.barrier))


if __name__ == "__main__":
    main()
