# megax — runbook (gev-plus VPS)

Standalone repo at **`~/megax/`**. Config via **`~/megax/.env`** (if needed) and
`config/` — see `README.md`.

Host monitoring: `~/vps-ops/status.sh --brief` (megax GUI :18555).

## Production start

```bash
cd ~/megax
python3 -m venv .venv && .venv/bin/pip install -e .
./scripts/start_vps_production.sh start    # or stop | restart | status
```

Starts **megax-gui** on **:18555**. If already running, `start` restarts it.

## Manual (legacy)

```bash
./scripts/start_gui.sh
# or: .venv/bin/megax-gui --host 0.0.0.0 --port 18555
```

See `README.md` and `docs/design.md` for CLI (`fetch-round`, `simulate`, etc.).
