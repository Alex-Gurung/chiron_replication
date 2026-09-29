"""Queue-job entrypoint: start one vLLM server on this job's GPUs, run a client, clean up.

Usage (as a lane=chiron queue job; supervisor sets CUDA_VISIBLE_DEVICES, JOB_NAME, RUN_DIR):
  python3 serve_and_run.py --model gptoss|qwen4b|mistral [--max-model-len N] -- <client argv...>
Tensor parallel = number of assigned GPUs. The client gets CHIRON_API_BASE / CHIRON_MODEL.
Only processes carrying this job's server marker are killed on exit (vLLM EngineCore orphans).
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
from urllib import request as urlrequest

PY = "/home/toolkit/eaiexp/env3/bin/python"
HUB = Path("/home/toolkit/.cache/huggingface/hub")
MODELS = {
    "gptoss": ("openai/gpt-oss-120b", HUB / "models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
               ["--reasoning-parser", "openai_gptoss"]),
    "qwen4b": ("Qwen/Qwen3-4B-Instruct-2507", None, []),
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.2", None, []),
    "qwen27": ("Qwen/Qwen3.8-27B", None, ["--limit-mm-per-prompt", '{"image": 0, "video": 0}', "--reasoning-parser", "qwen3",
                                          "--max-num-seqs", "256"]),   # text only; hybrid model: one recurrent-state slot per sequence
}


def up(base, name):
    try:
        with urlrequest.urlopen(base + "/models", timeout=10) as r:
            return any(m.get("id") == name for m in json.load(r).get("data", []))
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("client", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if args.model != "gptoss":
        os.nice(19)            # eval jobs yield CPU to the latency-bound gpt-oss generation sharing the pod
    client = args.client[1:] if args.client[:1] == ["--"] else args.client
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    job = os.environ.get("JOB_NAME", "local")
    run = Path(os.environ.get("RUN_DIR", f"/tmp/{job}"))
    run.mkdir(parents=True, exist_ok=True)
    name, snapshot, extra = MODELS[args.model]
    port = 8200 + int(devices[0]) * 10
    base = f"http://127.0.0.1:{port}/v1"
    env = dict(os.environ, HF_HOME="/home/toolkit/.cache/huggingface", HF_HUB_OFFLINE="1",
               VLLM_WORKER_MULTIPROC_METHOD="spawn", PYTHONUNBUFFERED="1",
               VLLM_CACHE_ROOT=f"/tmp/vllm_cache_{job}", TRITON_CACHE_DIR=f"/tmp/triton_{job}",
               TIKTOKEN_RS_CACHE_DIR=f"/tmp/harmony_{job}")
    Path(env["TIKTOKEN_RS_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    marker = uuid.uuid4().hex
    cmd = [PY, "-m", "vllm.entrypoints.openai.api_server", "--model", str(snapshot or name),
           "--served-model-name", name, "--tensor-parallel-size", str(len(devices)),
           "--max-model-len", str(args.max_model_len), "--gpu-memory-utilization", str(args.gpu_mem),
           "--port", str(port), "--enable-prefix-caching", *extra]
    (run / "server_config.json").write_text(json.dumps({"argv": cmd, "devices": devices, "client": client}, indent=1))
    log = open(run / "vllm.log", "a")
    server = subprocess.Popen(cmd, env=dict(env, CHIRON_SERVER_OWNER=marker), stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True)
    signal.signal(signal.SIGTERM, lambda s, f: (_ for _ in ()).throw(SystemExit(128 + s)))
    rc = 1
    try:
        t0 = time.monotonic()
        while not up(base, name):
            if server.poll() is not None:
                raise RuntimeError(f"server exited {server.returncode}; see {run / 'vllm.log'}")
            if time.monotonic() - t0 > 2400:
                raise RuntimeError("server not ready after 40 minutes")
            time.sleep(10)
        print(f"READY {name} tp={len(devices)} after {time.monotonic() - t0:.0f}s", flush=True)
        rc = subprocess.run(client, env=dict(env, CHIRON_API_BASE=base, CHIRON_MODEL=name)).returncode
        print(f"CLIENT rc={rc}", flush=True)
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=30)
        except Exception:
            pass
        for proc in Path("/proc").iterdir():
            if proc.name.isdigit():
                try:
                    if f"CHIRON_SERVER_OWNER={marker}".encode() in (proc / "environ").read_bytes().split(b"\0"):
                        os.kill(int(proc.name), signal.SIGKILL)
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    pass
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
