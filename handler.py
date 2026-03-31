import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional

import runpod
import requests


def _run(cmd: List[str], env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def _must_run(cmd: List[str], env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    proc = _run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def _which_or_none(name: str) -> Optional[str]:
    return shutil.which(name)


def _detect_seedvr2_runtime() -> Dict[str, Optional[str]]:
    app_roots = [
        Path("/opt/seedvr2_videoupscaler"),
        Path("/workspace/seedvr2_videoupscaler"),
        Path("/workspace/SeedVR2"),
        Path("/workspace"),
    ]
    app_root = next((p for p in app_roots if p.exists()), None)

    script_candidates = [
        Path("/opt/seedvr2_videoupscaler/inference_cli.py"),
        Path("/workspace/seedvr2_videoupscaler/inference_cli.py"),
        Path("/workspace/SeedVR2/inference_cli.py"),
        Path("/workspace/inference_cli.py"),
    ]
    script_path = next((str(p) for p in script_candidates if p.exists()), None)

    py_candidates = [
        Path("/opt/seedvr2_videoupscaler/.venv/bin/python"),
        Path("/workspace/seedvr2_videoupscaler/.venv/bin/python"),
        Path("/workspace/SeedVR2/.venv/bin/python"),
    ]
    py_path = next((str(p) for p in py_candidates if p.exists()), None)
    if py_path is None:
        py_path = _which_or_none("python3") or _which_or_none("python")

    return {
        "app_root": str(app_root) if app_root else None,
        "python": py_path,
        "inference_cli": script_path,
        "ffmpeg": _which_or_none("ffmpeg"),
        "ffprobe": _which_or_none("ffprobe"),
        "nvidia_smi": _which_or_none("nvidia-smi"),
    }


def _probe_runtime(runtime: Dict[str, Optional[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"runtime": runtime}

    if runtime["python"]:
        proc = _run([runtime["python"], "--version"])
        out["python_version"] = (proc.stdout or proc.stderr).strip()

    if runtime["ffmpeg"]:
        proc = _run([runtime["ffmpeg"], "-version"])
        line = (proc.stdout.splitlines() or [""])[0]
        out["ffmpeg_version"] = line

    if runtime["nvidia_smi"]:
        proc = _run([runtime["nvidia_smi"], "--query-gpu=name,memory.total", "--format=csv,noheader"])
        out["gpu_info"] = [x.strip() for x in proc.stdout.splitlines() if x.strip()]

    return out


def _download_source(url: str, dst: Path) -> None:
    if "drive.google.com" in url:
        import gdown

        ok = gdown.download(url=url, output=str(dst), quiet=False, fuzzy=True)
        if ok is None:
            raise RuntimeError(f"Failed to download Google Drive URL: {url}")
        return

    last_error: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            with requests.get(
                url,
                stream=True,
                timeout=(20, 600),
                headers={"User-Agent": "seedvr2-serverless-worker/1.0"},
            ) as resp:
                resp.raise_for_status()
                with open(dst, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return
        except Exception as err:
            last_error = err
            if attempt < 3:
                time.sleep(3 * attempt)

    raise RuntimeError(f"Failed to download source URL after retries: {url}: {last_error}")


def _decode_b64_video(data: str, dst: Path) -> None:
    payload = data
    if "," in payload and payload.split(",", 1)[0].startswith("data:"):
        payload = payload.split(",", 1)[1]
    dst.write_bytes(base64.b64decode(payload))


def _video_fps(path: Path, ffprobe_bin: str) -> float:
    proc = _must_run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ]
    )
    return float(Fraction(proc.stdout.strip()))


def _drop_first_frame_with_audio(inp: Path, out: Path, ffmpeg_bin: str, ffprobe_bin: str) -> None:
    fps = _video_fps(inp, ffprobe_bin)
    frame_dur = 1.0 / fps

    filter_complex = (
        f"[0:v]select='not(eq(n,0))',setpts=N/FRAME_RATE/TB[v];"
        f"[0:a]atrim=start={frame_dur:.9f},asetpts=PTS-STARTPTS[a]"
    )

    _must_run(
        [
            ffmpeg_bin,
            "-y",
            "-i",
            str(inp),
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "12",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            str(out),
        ]
    )


def _build_inference_cmd(
    runtime: Dict[str, Optional[str]],
    input_video: Path,
    output_video: Path,
    job_input: Dict[str, Any],
) -> List[str]:
    python_bin = runtime["python"]
    inference_cli = runtime["inference_cli"]
    if not python_bin or not inference_cli:
        raise RuntimeError("SeedVR2 runtime is missing python or inference_cli.py")

    cmd = [
        python_bin,
        inference_cli,
        str(input_video),
        "--output",
        str(output_video),
        "--model_dir",
        str(job_input.get("model_dir", "/workspace/models/SEEDVR2")),
        "--dit_model",
        str(job_input.get("dit_model", "seedvr2_ema_7b_fp16.safetensors")),
        "--cuda_device",
        str(job_input.get("cuda_device", 0)),
        "--resolution",
        str(job_input.get("resolution", 2160)),
        "--batch_size",
        str(job_input.get("batch_size", 37)),
        "--temporal_overlap",
        str(job_input.get("temporal_overlap", 5)),
        "--color_correction",
        str(job_input.get("color_correction", "wavelet_adaptive")),
        "--video_backend",
        str(job_input.get("video_backend", "ffmpeg")),
        "--seed",
        str(job_input.get("seed", 42)),
    ]

    if bool(job_input.get("uniform_batch_size", True)):
        cmd.append("--uniform_batch_size")
    if bool(job_input.get("vae_encode_tiled", True)):
        cmd.append("--vae_encode_tiled")
    if bool(job_input.get("vae_decode_tiled", True)):
        cmd.append("--vae_decode_tiled")
    if bool(job_input.get("ten_bit", True)):
        cmd.append("--10bit")

    if "vae_encode_tile_size" in job_input:
        cmd.extend(["--vae_encode_tile_size", str(job_input["vae_encode_tile_size"])])
    if "vae_decode_tile_size" in job_input:
        cmd.extend(["--vae_decode_tile_size", str(job_input["vae_decode_tile_size"])])
    if "vae_encode_tile_overlap" in job_input:
        cmd.extend(["--vae_encode_tile_overlap", str(job_input["vae_encode_tile_overlap"])])
    if "vae_decode_tile_overlap" in job_input:
        cmd.extend(["--vae_decode_tile_overlap", str(job_input["vae_decode_tile_overlap"])])

    extra_args = job_input.get("extra_args", [])
    if isinstance(extra_args, list):
        cmd.extend([str(x) for x in extra_args])

    return cmd


def _mux_original_audio(source_video: Path, upscaled_video: Path, out_video: Path, ffmpeg_bin: str) -> None:
    _must_run(
        [
            ffmpeg_bin,
            "-y",
            "-i",
            str(upscaled_video),
            "-i",
            str(source_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            "-shortest",
            str(out_video),
        ]
    )


def _resolve_remote_target(remote_value: str, local_file: Path) -> str:
    target = str(remote_value).strip()
    if not target:
        raise ValueError("gdrive_remote_path is empty")
    if target.endswith("/"):
        return f"{target}{local_file.name}"
    if target.lower().endswith(".mp4"):
        return target
    return f"{target}/{local_file.name}"


def _upload_with_rclone(local_file: Path, remote_value: str, rclone_conf_b64: str) -> str:
    rclone_bin = _which_or_none("rclone")
    if not rclone_bin:
        raise RuntimeError("rclone binary not available in worker image")

    remote_target = _resolve_remote_target(remote_value, local_file)
    conf_bytes = base64.b64decode(rclone_conf_b64)

    with tempfile.TemporaryDirectory(prefix="rclone_conf_") as td:
        conf_path = Path(td) / "rclone.conf"
        conf_path.write_bytes(conf_bytes)
        _must_run(
            [
                rclone_bin,
                "copyto",
                str(local_file),
                remote_target,
                "--config",
                str(conf_path),
                "--drive-chunk-size",
                "64M",
                "--transfers",
                "1",
                "--checkers",
                "2",
                "--retries",
                "5",
                "--low-level-retries",
                "10",
            ]
        )

    return remote_target


def _handler(job: Dict[str, Any]) -> Dict[str, Any]:
    job_input = job.get("input", {})
    runtime = _detect_seedvr2_runtime()

    if bool(job_input.get("dry_run", False)):
        probe = _probe_runtime(runtime)
        probe["status"] = "ready"
        return probe

    ffmpeg_bin = runtime["ffmpeg"]
    ffprobe_bin = runtime["ffprobe"]
    if not ffmpeg_bin or not ffprobe_bin:
        raise RuntimeError("ffmpeg/ffprobe not found")

    source_url = job_input.get("source_video_url")
    source_b64 = job_input.get("source_video_base64")
    if not source_url and not source_b64:
        raise ValueError("Provide input.source_video_url or input.source_video_base64")

    with tempfile.TemporaryDirectory(prefix="seedvr2_job_") as td:
        work = Path(td)
        source_video = work / "input.mp4"

        if source_url:
            _download_source(str(source_url), source_video)
        else:
            _decode_b64_video(str(source_b64), source_video)

        processed_input = source_video
        if bool(job_input.get("drop_first_frame", False)):
            processed = work / "input_minus1.mp4"
            _drop_first_frame_with_audio(source_video, processed, ffmpeg_bin, ffprobe_bin)
            processed_input = processed

        raw_output = work / "upscaled_raw.mp4"
        cmd = _build_inference_cmd(runtime, processed_input, raw_output, job_input)

        run_env = os.environ.copy()
        visible_gpu = job_input.get("visible_gpu", job_input.get("cuda_device", 0))
        run_env["CUDA_VISIBLE_DEVICES"] = str(visible_gpu)
        run_env["PYTORCH_CUDA_ALLOC_CONF"] = str(job_input.get("pytorch_cuda_alloc_conf", "backend:native"))
        existing_pythonpath = run_env.get("PYTHONPATH", "")
        run_env["PYTHONPATH"] = f"/app:{existing_pythonpath}" if existing_pythonpath else "/app"

        proc = _run(cmd, env=run_env)
        if proc.returncode != 0:
            return {
                "status": "failed",
                "command": cmd,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
            }

        final_output = raw_output
        if bool(job_input.get("preserve_audio", True)):
            muxed = work / "upscaled_with_audio.mp4"
            _mux_original_audio(processed_input, raw_output, muxed, ffmpeg_bin)
            final_output = muxed

        return_mode = str(job_input.get("return_mode", "metadata"))
        response: Dict[str, Any] = {
            "status": "ok",
            "output_size_bytes": final_output.stat().st_size,
            "output_filename": final_output.name,
            "command": cmd,
        }

        remote_path = job_input.get("gdrive_remote_path")
        if remote_path:
            rclone_conf_b64 = str(job_input.get("rclone_conf_base64", os.getenv("RCLONE_CONF_BASE64", "")))
            if not rclone_conf_b64.strip():
                raise ValueError("gdrive_remote_path provided but no rclone_conf_base64 supplied")
            uploaded_target = _upload_with_rclone(final_output, str(remote_path), rclone_conf_b64)
            response["gdrive_remote_path"] = uploaded_target
            response["uploaded_via"] = "rclone"

        if return_mode == "base64":
            response["video_base64"] = base64.b64encode(final_output.read_bytes()).decode("utf-8")

        return response


if __name__ == "__main__":
    runpod.serverless.start({"handler": _handler})
