# SeedVR2 Serverless Worker (GHCR)

This project builds a RunPod Serverless worker image to run SeedVR2-style video upscaling and optional first-frame removal.

## What it does

- Uses `gemneye/seedvr-runpod:latest` as a runtime base.
- Starts a RunPod serverless handler (`handler.py`).
- Supports:
  - `dry_run` runtime probing.
  - `source_video_url` or `source_video_base64` input.
  - `drop_first_frame` with synced audio trim.
  - SeedVR2 inference via detected `inference_cli.py`.
  - Optional audio remux from source.

## Build and push to GHCR

Push to `main` (or run workflow manually):

- Workflow: `.github/workflows/build-and-push-ghcr.yml`
- Output image:
  - `ghcr.io/<github_owner>/<repo_name>:latest`
  - `ghcr.io/<github_owner>/<repo_name>:sha-<shortsha>`

## Create RunPod endpoint (REST)

Use the generated GHCR image in a Serverless template and then create an endpoint.

## Test request (dry run)

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/run" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d @sample_request_dry_run.json
```

## Test request (job)

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/run" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d @sample_request_job.json
```
