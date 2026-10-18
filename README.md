# LOL Teamfight Lab

Research code for League of Legends engagement analysis from public Riot Games API telemetry.

| Folder | Contents |
|---|---|
| `COG/` | Code of the IEEE CoG 2026 paper (tag `v1.0-cog2026`) and the later engagement-definition work: acquisition, detector, features, learners, evaluation scripts, tests. Start with `COG/README.md`. |
| `research_pipeline/`, `common_training_r1/` | Source of the current multi-window outcome-ledger workflow: raw-source audits, ledger producers, PostgreSQL catalog checks, patch-knowledge and terrain tooling, shared-input smoke. Start with `research_pipeline/README.md`. |

This is a source-code release, not a data release. Raw match data, caches, database contents, credentials, split and ledger rows, checkpoints, predictions and generated results are not included. Paths are read from environment variables; see `research_pipeline/README.md`.

The code is released under the MIT License (`LICENSE`). *LOL Teamfight Lab* is not endorsed by Riot Games and does not reflect the views or opinions of Riot Games or anyone officially involved in producing or managing League of Legends. League of Legends and Riot Games are trademarks or registered trademarks of Riot Games, Inc.
