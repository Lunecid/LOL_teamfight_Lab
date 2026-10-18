# Current bounded common-input smoke

This folder contains the shared observed-prefix adapter, tree and hierarchical model implementations, a bounded run card, and report/visualization code.

## Run outline

1. Create an isolated run directory outside the repository and install the packages in `requirements.txt`.
2. Configure the local PostgreSQL connection and data/artifact roots via environment variables; keep credentials in `PGPASSFILE` or another local secret store.
3. Copy `CARD.template.json` to a run-specific `CARD.json`, fill in the four local role sample counts, then review every path and frozen input reference before execution.
4. Run `data.py` to build/validate the bounded input and `train.py` for the bounded fit. Use `python -m unittest` for the included adapter/pipeline checks. Keep local reports and metric exports out of public version control.

This source package contains no recorded smoke counts or performance results. The report-rendering helper is omitted from the public snapshot. STOP is used for epoch selection in the bounded workflow; that role does not establish confirmatory performance or full-scale streaming capacity. Re-run locally after adapting paths and dependencies; this public code snapshot does not bundle private data, ledgers, or run receipts.

Some raw-data audit regression tests also need LOL_REVIEW_MATCH_ID to identify a local fixture; the repository does not contain that identifier or the raw fixture.
