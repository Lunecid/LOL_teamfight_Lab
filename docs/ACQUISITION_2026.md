# 2026 current-season acquisition agent

This agent reuses the routing, high-rank discovery, Match-V5 pagination, and
multi-window throttling ideas from `Lol_project`, while replacing its unsafe
one-shot key loading and file-existence checkpointing.

## Important authentication boundary

Riot **development keys expire after 24 hours, not every hour**. Riot does not
publish an API that programmatically regenerates a development key. The agent
therefore implements the safe supported boundary:

1. Reload the explicit key file before every request.
2. On HTTP 401/403, pause all workers and preserve the SQLite checkpoint.
3. Create `AUTH_REQUIRED.txt` and expose `WAITING_FOR_KEY` in `status.json`.
4. Detect a changed key file, probe it once, and resume automatically.

An optional `--key-refresh-command` may call an external, user-owned secret
provisioner that atomically replaces the file. It must not scrape the Riot
Developer Portal or store Riot login/2FA cookies. For unattended research,
apply for a Personal API key instead of automating a development-key login.
Keeping a signed-in Whale tab open does not provide a supported refresh API;
the collector intentionally does not attach to Chromium remote debugging or
extract credentials from browser state.

Official references:

- <https://developer.riotgames.com/docs/portal>
- <https://developer.riotgames.com/policies/general>

The former `Lol_project/.env` key was found in plaintext during the audit and
must be revoked/reset before this agent is started. Do not run
`Lol_project/scripts/debug_riot_key.py`; it prints the full key.

## Data and state layout

Use a new root. Do not mix 2026 files with the existing 2025 corpus.

```text
D:\LOL_Project\data\raw\2026_current\
  kr\
    detail\KR_....json
    timeline\KR_....json
    quarantine\
  _collector_state\
    collector.sqlite3
    collector.log
    status.json
    AUTH_REQUIRED.txt       # only while authentication is paused
```

The SQLite database records rank/tier/LP snapshots, per-PUUID incremental
cursors, match provenance, attempts, exclusions, raw API/public patch labels,
and SHA-256 checksums. Raw JSON is written to `.part`, validated, flushed, and
atomically promoted. A missing timeline is never written as a fake JSON file.

Public patch labels are `26.x`, while Match-V5 `gameVersion` in 2026 uses
`16.x`. The raw build string is the source of truth; both labels are stored.

## Setup

Install the acquisition dependency from the versioned `LOL_teamfight` repo:

```powershell
python -m pip install -e ".[acquisition]"
```

Copy `config/riot_api.env.example` to a private path outside the repository,
for example `C:\Users\<user>\.secrets\riot.env`, and put the newly reset key
there. Do not pass the key itself as a command-line argument.

Probe authentication without downloading data:

```powershell
python scripts\collect_current_season.py `
  --key-file C:\Users\<user>\.secrets\riot.env `
  --output-root D:\LOL_Project\data\raw\2026_current `
  --probe-only
```

Run a deliberately small first cycle:

```powershell
python scripts\collect_current_season.py `
  --key-file C:\Users\<user>\.secrets\riot.env `
  --output-root D:\LOL_Project\data\raw\2026_current `
  --min-api-patch 16.13 `
  --players-per-cycle 5 `
  --matches-per-cycle 10 `
  --once
```

After checking `status.json`, SQLite counts, JSON schema, queue `420`, map `11`,
and detail/timeline pair integrity, start the continuous agent by omitting
`--once`. It refreshes Challenger/Grandmaster/Master snapshots hourly, scans a
rotating player batch with a persisted high-water cursor, and downloads pending
matches through one shared application-level limiter. Each hourly cycle has a
default 50-minute work budget, preventing a large pending queue from delaying
the next rank snapshot and authentication health check for several hours.

For an overnight Master-only run with physical storage guards, use
`--tiers MASTER --max-storage-gb 40 --min-free-gb 150`. Patch-level match and
byte totals are continuously written to `status.json`; raw files remain in one
deduplicated store, while SQLite records both API `16.x` and public `26.x`
labels without duplicating files.

## Replay archive for detector validation

Replay acquisition is deliberately separate from Match-V5 acquisition. The
League Client must be open and logged in; its short-lived local lockfile secret
is read only in memory and is never logged or persisted. Replays are linked to
completed detail/timeline pairs by match ID and all three SHA-256 hashes.

Run a dry check before downloading:

```powershell
python scripts\collect_current_replays.py `
  --source-db D:\LOL_Project\data\raw\2026_current\_collector_state\collector.sqlite3 `
  --output-root D:\LOL_Project\data\replays\2026_current `
  --league-lockfile "D:\Game\Riot Games\League of Legends\lockfile" `
  --dry-run
```

The default archive processes expiry-risk oldest matches first, stops at 500
complete replays or 8 GiB, and keeps at least 10 GiB free. Each League Client
download is copied through a flushed `.part` file, hashed, atomically promoted,
and then removed from the Client replay directory to avoid duplicate storage.
The resumable state is stored under `_replay_state/replay_manifest.sqlite3` and
an atomically refreshed, player-identifier-free `manifest.jsonl` is exported
for analysis.

Do not evaluate detector recall from detector-selected clips alone. Select
full matches from this archive without exposing detector windows to annotators;
use short clips only for adjudication and durable evidence after full-match
annotation.

## Windows Task Scheduler

The installer registers one limited task with an at-logon trigger, a five-minute
watchdog trigger, and restart-on-failure settings. A running instance ignores
the watchdog trigger; if the task is stopped unexpectedly, the next trigger
starts it without losing the SQLite checkpoint. Run it only after the probe and
small-cycle checks pass:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_riot_collector_task.ps1 `
  -ProjectRoot C:\Users\todtj\PycharmProjects\LOL_teamfight `
  -PythonExe C:\Path\To\python.exe `
  -KeyFile C:\Users\todtj\.secrets\riot.env `
  -OutputRoot D:\LOL_Project\data\raw\2026_current `
  -MinApiPatch 16.13 `
  -WatchdogMinutes 5 `
  -StartNow
```

Only one collector may run against a state directory. A file lock prevents a
manual process and the scheduled task from running concurrently and exceeding
the application key limit. The lock also applies to `--once`; stop the scheduled
task before running a manual smoke cycle.

## Failure behavior

- **401/403:** stop API traffic, checkpoint, wait for a different key, probe,
  then resume. The agent never prints a key/prefix/suffix.
- **429:** honor `Retry-After` globally before any worker proceeds.
- **500/503/network:** exponential backoff with jitter, followed by supervisor
  backoff if retries are exhausted.
- **404 timeline:** record state and retry; do not place a sentinel in the valid
  timeline directory.
- **Crash during write:** `.part` remains non-pairable; a later retry validates
  and promotes a complete pair.

## Analysis integration

After the pilot is accepted, point the feature pipeline at the new root:

```powershell
$env:LOL_DETAIL_DIR='D:\LOL_Project\data\raw\2026_current\kr\detail'
$env:LOL_TIMELINE_DIR='D:\LOL_Project\data\raw\2026_current\kr\timeline'
```

Keep feature-cache creation disabled until the known future-information and
coordinate-frame issues are fixed and the 2026 feature schema is frozen.
