# spatial-gpt (HK 4090)

Private NetBird vLLM pair for Pakistan Decoupled-FARM Phase 4.

**Full integration doc (copy to FARM repo):** [docs/FARM_HK_VLLM_INTEGRATION.md](docs/FARM_HK_VLLM_INTEGRATION.md)

| Role | Hugging Face id | Served name | Bind |
|------|-----------------|-------------|------|
| Caption VLM | `Qwen/Qwen3-VL-8B-Instruct-FP8` | `qwen3-vl-8b` | `100.109.254.4:8100` |
| Text embed | `Qwen/Qwen3-Embedding-0.6B` | `qwen3-emb-0.6b` | `100.109.254.4:8102` |

This host already binds **8000** (`spatialx-backend-web`) and **8002** (`kodifly-data-broker` daphne). Pakistan must set `VLLM_BASE_URL=http://100.109.254.4:8100/v1` and `VLLM_EMBED_BASE_URL=http://100.109.254.4:8102/v1`.

This box is **stateless**: JPEG + bbox in, JSON caption out; caption text in, embedding vector out. Stella / SAM3 / Phase 3 / the 7k-object job stay on the Pakistan PC.

Do **not** port-forward 8100/8102 on the office router. Do **not** start the old welding API (`vlm-api-qwen25vl.service` on 8090). Do **not** load extra models (VL-Embedding-2B, SigLIP2, Qwen3.5-9B).

NetBird DNS is not configured. Clients must use `http://100.109.254.4:<port>`, not the FQDN.

**Firewall:** UFW + iptables rules on `wt0` for ports 8100/8102 from `100.109.0.0/16` are required for Pakistan TCP. See integration doc §3.

## Secrets

Copy `.env.example` → `.env` (gitignored).

- `VLLM_API_KEY` — same Bearer token on both ports. Client sends `Authorization: Bearer <key>`.
- `HF_TOKEN` — Hugging Face token for gated downloads.

## Start / stop

Manual (embedder first, then VLM — they share one 4090):

```bash
cd /home/kodifly/spatial-gpt
./scripts/serve_embed.sh
./scripts/serve_vlm.sh
```

systemd (user units — this account has lingering and no passwordless sudo). Start embedder first:

```bash
mkdir -p ~/.config/systemd/user
cp /home/kodifly/spatial-gpt/systemd/*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now spatial-gpt-embed.service
systemctl --user enable --now spatial-gpt-vlm.service
systemctl --user status spatial-gpt-embed.service spatial-gpt-vlm.service
journalctl --user -u spatial-gpt-vlm -u spatial-gpt-embed -f
```

Stop:

```bash
systemctl --user stop spatial-gpt-vlm.service spatial-gpt-embed.service
```

If you later have sudo, you can instead install the same files under `/etc/systemd/system/` with `User=kodifly`, `After=netbird.service`, and `WantedBy=multi-user.target`.

If NetBird is down, these listeners are unreachable from Pakistan. That is intended. Units restart until the overlay IP is bindable.

## Smoke

From this box (needs the services up):

```bash
source .env
curl -H "Authorization: Bearer $VLLM_API_KEY" http://100.109.254.4:8100/v1/models
curl -H "Authorization: Bearer $VLLM_API_KEY" http://100.109.254.4:8102/v1/models
.venv/bin/python scripts/smoke_caption.py
.venv/bin/python scripts/smoke_embed.py
ss -tlnp | grep -E '8100|8102'   # must show 100.109.254.4, not 0.0.0.0
nvidia-smi
```

Expected caption latency is ~0.3–2 s/image when batched. Pakistan will send **batches of 8**, not one-at-a-time videos.

Qwen3-Embedding-0.6B dimension is **1024** (not Gemini's 768). Pakistan accepts any dim ≥ 128.

## API

Caption: `POST http://100.109.254.4:8100/v1/chat/completions`

- `model`: `qwen3-vl-8b`
- one JPEG (typically 504×504 face) as `image_url` data URI
- user text includes `TARGET BOUNDING BOX: <box>(x1,y1),(x2,y2)</box>` in normalized 0–1000
- guided JSON via `response_format.json_schema` using `prompts/caption_schema.json`

Embed: `POST http://100.109.254.4:8102/v1/embeddings`

```json
{ "model": "qwen3-emb-0.6b", "input": ["blue corrugated shipping container"] }
```

Optional query parse: same port 8100, text only, system prompt `prompts/query_parser_system.txt`. Pakistan may keep parsing local.

## Memory split (one 4090, 24 GB)

- VLM `--gpu-memory-utilization 0.70`
- Embed `--gpu-memory-utilization 0.15`
- Leave other small GPU occupants alone. If OOM, drop VLM util toward 0.60 before stopping other jobs.

Weights live in `~/.cache/huggingface`, not in this repo.

## Python / vLLM (this box)

Driver is CUDA **12.9**. Latest vLLM (0.27) ships CUDA **13** wheels and will not see the 4090. Pin **vLLM 0.14.0** + PyTorch cu128:

```bash
cd /home/kodifly/spatial-gpt
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --torch-backend cu128 -r requirements.txt
```
