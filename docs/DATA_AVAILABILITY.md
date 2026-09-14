# Code and data availability

Corpus v3.3 of the engagement-outcome study (IEEE Transactions on Games extension).  Written
2026-09-14.  This file backs `docs/tog_manuscript/sec_availability.tex` and answers the review
comment that no anonymised code link or other source had been provided.  Every count below names
the file it was read from and the date it was read.  Path prefixes:

- `FEAT/` = `D:/LOL_Project/fusion_2615/features/`
- `SHARDS/` = `D:/LOL_Project/fusion_2615/corpus_shards_v33/`
- `CACHE/` = `D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13/`

These locations are on the machine that produced the results.  They are not in the repository.

## 1. Summary

| Item | Released | Form and size | Source of the numbers |
|---|---|---|---|
| Code: detector, definition pipeline, labels, features, learners, evaluation and audit scripts, tests | yes, MIT License | repository at the release commit | `LICENSE` |
| Definition specifications (per patch and pooled), event-price table, map anchors, configuration presets | yes, MIT License | `config/fight_boundary/`, `config/game_rules/`, `core/presets.py` | files in the repository |
| Per-engagement feature matrix | yes | 32 shards, 566,452 rows x 7,106 columns, 2,350,323,256 bytes | `SHARDS/manifest.json`; `FEAT/model_comparison_input_audit.json` (`rows_total`, `columns_total`); byte sum of `SHARDS/shard_*.npz` (2026-09-14) |
| Outcome labels | yes | six arrays inside the shards: five with -1 marking a draw, and `y`, the headline label with draws resolved by a seeded coin (section 2.2) | shard keys and `SHARDS/manifest.json` `label` (2026-09-14) |
| Out-of-fold predictions of the headline model | yes | 532,547 rows, 6,209,242 bytes | `FEAT/scale_decomposition_v33_market_event.preds.npz` |
| Test-patch predictions of every learner | yes | 153,816 rows per learner, four files | `FEAT/model_comparison_v33_patch.preds.npz`, `FEAT/deep_tabular_v33_patch_{full,ft,saint}.preds.npz` |
| Match-identifier lists | yes | 210,000 cached matches with their patch; 191,940 behind labelled engagements | `D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13_patch_index.json`; `groups` arrays of the shards |
| Per-match cache (event, metadata and tensor files) | no | 210,000 matches x 3 files | file count of `CACHE/` (2026-09-14) |
| Raw Match-V5 match and timeline responses | no | not held any more | `docs/DATA_MANIFEST.md`, corpora table |
| Merged feature matrices (`*.matrix.npy`) | no | 13,966,440,640 bytes each; rebuilt from the shards | size of `FEAT/scale_decomposition_v33_market_event.matrix.npy` |

The released set is the one in `docs/CLAUDE_TOG_PAPER_PLAN.md` section 9 (code and specifications
under MIT; derived features, labels and out-of-fold predictions; match-identifier lists).  The
per-match cache is left out because it is an event-level extract of the timelines, much closer to
the raw data than the per-engagement tables.

## 2. What is released

### 2.1 Code and specifications

The MIT License (`LICENSE`) covers the whole repository.  The detector constants of corpus v3.3 are
the `v3.3` preset in `core/presets.py` (G 13,700 ms, D 4,264.0 u, R 1,600 u, B 15,000 ms, label
horizon 35 s, label `market_event` with engagement attribution and a 300 g dead zone).  The estimates
behind G and D, with their bootstrap intervals, are in `config/fight_boundary/spec_pooled.json`; the
per-patch estimates are in `spec_15.14.json`, `spec_15.15.json` and `spec_15.16.json`.

Three scripts behind the learner comparison are on branch `codex/engagement-state-value`, not yet on
`feature/fight-boundary-pipeline`: `scripts/run_model_comparison_v33.py` (tree, linear and lead-only
learners; commit 3fb00c3), `scripts/audit_model_comparison_inputs.py` and
`scripts/paired_learner_cis_v33.py` (commit 37bc991).  They must be merged before the release commit
is tagged.

The corpus manifest (`SHARDS/manifest.json`) records the commit the corpus was built from (aac7131,
2026-09-09), the detector constants, the label settings and the SHA-1 of the price table and map
anchors.  The map-anchor hash (108932f8...) matches the file in the repository.  The price-table hash
(d731292b...) is that of the file at commit 5b5f9c8 as checked out on Windows (CRLF line ends).
Commit 006d5e4 changed only the `unit`, `source` and `note` strings of that file; every price is
unchanged (pooled and per-patch tables compared on 2026-09-14), so the current file has another hash
but the same prices.

### 2.2 Derived data

- **Feature matrix.**  One row per engagement, 7,106 columns: 1,015 base features x 7 summary
  statistics plus the age of the last timeline frame.  No column holds a player identifier.  Of the
  7,106 names in `SHARDS/feature_names.json`, none matches `puuid`, `summoner` (other than
  `summoner_spell`), `riot_id`, `game_name`, `tag_line`, `account`, `player_name` or
  `participant_id` (checked 2026-09-14).  The identifier-type columns encode game content only:
  champion, rune style and summoner spell (60 such base columns).
- **Labels.**  `y_market_event` (headline; 532,547 non-draw rows), `y_market_event@window` (541,627),
  `y_market_lex` (538,593), `y_market_lex@window` (541,895) and `y_attention_value_win` (561,721, the
  label of the CoG-era code), each with -1 for a draw (the manifest's `extra_tie_policy` is `drop`).
  Counts of values >= 0 summed over the 32 shards on 2026-09-14.
- **The array `y`.**  Each shard also holds `y`, the label the rows were built with: `market_event`
  under the tie policy `random` (`SHARDS/manifest.json`, `label.row_tie_policy`).  All 566,452 rows
  carry 0 or 1.  On the 532,547 rows where `y_market_event` is not a draw the two arrays agree on every
  row; the other 33,905 rows (the draws) take their value from a deterministic coin, a hash of seed 7,
  the engagement key and the window's events (`_seeded_tie_coin` in `gameplay/labels.py`).  Checked
  over the 32 shards on 2026-09-14.  No reported number uses `y`: the headline, the scale-cut
  re-scoring and the learner comparison use `y_market_event` (the `y_key` or `label_key` fields of
  their result files).  Either the release keeps `y` with this note, or `y` is removed from the
  released shards, in which case their byte sizes no longer match those in `SHARDS/manifest.json`
  (`shards.<i>.bytes`); the choice is open (section 8).
- **Other per-row arrays.**  `groups` (Riot match identifier), `patch`, `engage_ts` (the cutoff, in ms
  from the start of the match; smallest value 120,004), `cluster_blue` and `cluster_red` (participants
  per side; -1 where the count is unknown, in 59 rows, 42 of them with a non-draw `y_market_event`),
  `present_blue` and `present_red` (champions alive within R of the first kill at the cutoff; no
  negative value).  Read from the 32 shards on 2026-09-14; the 42 agree with
  `FEAT/tog_revision/A6-definition-sensitivity/scale_cut_sensitivity_v33.json`
  (`participation.rows_with_negative_count`).
- **Predictions.**  The headline out-of-fold predictions (match-grouped 5-fold cross-validation,
  532,547 rows with label, match, scale class and patch), and the test-patch (15.16) predictions of
  the nine learners of the patch-holdout comparison, 153,816 rows each.  The deep-learner predictions
  come from the published runs and will be replaced when the corrected runs finish (section 8).
  In the prediction files the array named `y` is the evaluated label, `y_market_event` without its
  draws, not the shard array `y` above: for the out-of-fold file the scale-cut artefact records
  `alignment_with_shards.label_equal = true` against the shards' `y_market_event`, and the extended
  input audit passes "tree y equals the shard test patch element-wise" and the same check for
  `deep:full`, `deep:ft` and `deep:saint`.
- **Match identifiers.**  Riot match identifiers (form `KR_<gameId>`) of the 210,000 matches in the
  cache with their patch (15.14: 74,673; 15.15: 74,748; 15.16: 60,579; patch index, counted
  2026-09-14), and of the 191,940 matches with at least one labelled engagement (194,676 matches have
  at least one engagement before draws are removed; unique `groups` over the shards, 2026-09-14).

A match identifier holds no player data, but anyone with a Riot API key can use it to request the
match record, which does identify the players: in the Match-V5 reference
(<https://developer.riotgames.com/api-details/match-v5>, read 2026-09-14) the match metadata lists
the participants' PUUIDs, and each participant entry carries `puuid`, `riotIdGameName` and
`riotIdTagline`.  The list is therefore a pointer to data that Riot serves under its own terms
(section 3), not a copy of that data.

### 2.3 Two ways to reproduce

1. **From the released tables, without API access.**  The headline and learner-comparison AUCs,
   their bootstrap intervals and the paired comparisons can be recomputed, and the models refitted,
   from the shards and prediction files with
   `scripts/run_scale_decomposition.py`, `scripts/run_deep_tabular_baselines.py` and the three
   scripts named in section 2.1.
2. **From the match identifiers, with an API key.**  The records can be requested again from the
   Match-V5 match and timeline endpoints (bibliography keys `riotapi2024matchv5`,
   `riotapi2024timeline`), cached with `python main.py --mode build_cache`, and the whole pipeline
   re-run (README, section Reproducing).  How long Riot keeps serving matches of 2025 patches has not
   been checked, so this path may fail for some or all matches.

## 3. Riot Games API terms

Read on 2026-09-14 from <https://developer.riotgames.com/terms> (the page states it was last updated
on 9 December 2013; bibliography key `riot2013apiterms`, mapped in
`docs/tog_manuscript/references_audit.md` section 2c) and
<https://developer.riotgames.com/policies/general> (last updated 29 May 2025; key
`riot2025devpolicies`).  The following are paraphrases, not legal advice.

- **What the terms cover.**  "Materials" include the API, the keys, the game data the API returns
  and their "copies, portions, extracts and derivatives".  Riot keeps all rights in the Materials,
  including derivatives.
- **What use is licensed.**  Developers may build and distribute applications that use the game data
  in order to show it to their users, or for other purposes set out in Riot's specifications or
  communicated by Riot in writing.
- **What is prohibited.**  Using, distributing or transmitting the game data in a way the terms do
  not authorise.
- **Deletion.**  When a player asks Riot to delete their personal data, Riot passes the request to
  active developers as a list of identifiers.  When the licence ends, the developer must stop using
  and delete all game data it holds.
- **Not addressed.**  A string search of the terms finds no "research" and no "dataset"; neither page
  mentions academic use or public datasets.
- **Policies.**  Products must not de-anonymise players who cannot reasonably be identified from
  visible information.

Consequences for this release:

- The raw match and timeline records are not redistributed.  They carry player identifiers, a public
  copy could not honour deletion requests passed on by Riot, and publishing the records as a dataset
  is not among the licensed uses.
- The per-engagement tables contain no player identifiers.  They are still derivatives in the sense
  of the terms.  The terms name no research use and no dataset, so they neither expressly authorise
  nor expressly forbid sharing such tables; whether a derived table counts as game data distributed
  "in any manner not authorized" is not settled by the text.  The release therefore rests on the
  authors' choice, not on the terms: they limit it to tables without player identifiers, for
  non-commercial research, together with this notice.  The licence of the data files and whether
  Riot's written confirmation is sought before release are open (section 8).

## 4. The raw data are no longer held

`docs/DATA_MANIFEST.md` (corpora table) marks the per-match cache of the main corpus as the sole
copy and states that the raw responses no longer exist.  On 2026-09-14 `CACHE/` held 210,000
matches (210,000 each of `*.meta.json`, `*.events.json` and `*.npz`), and `D:/LOL_Project/data/raw/`
held only the 2026 collections (`2026_current`, `2026_patch_16_14`, `2026_patch_16_14_pilot`), no
2025 data.  The manifest row gives 205,884 matches for the same cache; the file count and the patch
index both give 210,000, so the manifest figure is out of date.

The cache keeps no player identity and no rank.  In a seeded sample of 200 metadata files
(`random.seed(0)`, 2026-09-14) the keys were `anchor_is_norm`, `anchors`, `feature_version`,
`interp_policy`, `match_id`, `patch`, `patch_full`, `role_slots`, `static_meta` and `team_map`; none
of the 200 files contains `puuid` or `queue`.

## 5. Corpus and patch numbering

- **Collection.**  Korean (KR) server.  The collector queried the Master, Grandmaster and Challenger
  ladders (`acquisition/config.py`, `tiers`; `analysis/analysis.py`, `sampling_frame`: a collection
  constraint, per-match tier not retained).  Tier is a property of the collection, not a measured
  variable; neither tier nor queue is stored per match (section 4).
- **Size.**  566,452 engagements; 532,547 engagements with a non-draw `market_event` label
  (15.14: 187,547; 15.15: 191,184; 15.16: 153,816 engagements) in 191,940 matches, from
  `FEAT/scale_decomposition_v33_market_event.json` (`n`, `n_matches`, `patch_distribution`).
- **Patch numbers.**  The API's `gameVersion` numbers 15.14, 15.15 and 15.16 are the patches Riot
  published as 25.14, 25.15 and 25.16, with notes dated 15 July, 29 July and 12 August 2025 (keys
  `riot2025patch2514`, `riot2025patch2515`, `riot2025patch2516`).  The mapping rests on Riot data
  (`references_audit.md`, note N2): Data Dragon lists the champion Yunara (key 804) first at version
  15.14.1 and not at 15.13.1 (`riot2025ddragon1514`); Riot's 25.14 notes introduce Yunara; and champion
  804 appears in 889 of 4,000 sampled corpus matches of version 15.14.  25.15 = 15.15 and
  25.16 = 15.16 follow by succession; no Riot page states the correspondence in words.

## 6. Review anonymity at IEEE Transactions on Games

Pages read on 2026-09-14, and re-read the same day for this revision.  The two
`cis.ieee.org` pages answer an automated fetcher with HTTP 418; requested with a browser user-agent
string they return HTTP 200 and the text summarised below.

| Page | What it says |
|---|---|
| <https://transactions.games/submit/submission-guidelines> (no date on the page) | A notice at the top: from 1 January 2025 the journal reviews double-anonymously, and a manuscript submitted on or after that date must be "fully anonymized".  Author names, affiliations, biographies and funding acknowledgements must not appear, and any other content that could reveal the authors must be removed.  Earlier submissions stay single-anonymous.  The section on extending previously published work asks authors to state the relation to the earlier work in the text and to keep anonymity.  The same page still carries an older paragraph under "Submission Instructions" that describes single-blind review with author names on the manuscript. |
| <https://cis.ieee.org/publications/t-games/tciaig-information-for-authors> (IEEE Computational Intelligence Society; the page quotes an open-access charge for articles submitted in 2026) | A section "Double-Anonymous Review": each published article is reviewed by at least two independent reviewers, and neither side knows the other's identity.  The page names no code, software or data link and no supplementary material. |
| <https://cis.ieee.org/publications/t-games/tciaig-manuscript-submission> | Still says the journal runs single-blind review and that authors list their names and affiliations on the manuscript. |
| <https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/research-reproducibility/> (IEEE Author Center, all IEEE journals) | Encourages sharing data in repositories such as figshare, Zenodo or Dryad and code through Code Ocean.  Says nothing about anonymity. |

Reading.  The two single-blind paragraphs contradict the notice.  The notice names the date from
which it applies and agrees with the society's "Double-Anonymous Review" section, so this work is
prepared for double-anonymous review; the contradiction can be raised with the editorial office
together with item 2 below.
None of the four pages mentions code links, software repositories or data-availability statements
for a manuscript under review; the guidelines page mentions supplementary material only as
excluded from page limits.  Hence:

1. The manuscript under review names no repository, account or URL that belongs to the authors.
2. Reviewers get the code as an anonymised copy made by `scripts/prepare_anonymous_release.py`
   (section 7), either as supplementary material or behind an anonymised link.  Which form the
   editorial office accepts is to be confirmed with the Editor-in-Chief, whose contact is on the
   guidelines page.
3. The earlier conference submission of this work was not published.  References to it (its
   submission number, its reviews, the conference and year) can identify the authors to anyone who
   handled it, so the review copy and the manuscript should leave them out.
4. After acceptance the public repository, the release tag and an archival DOI replace the review
   copy.

## 7. Anonymised review copy

`scripts/prepare_anonymous_release.py` works only on this machine.  It runs only read-only git
subcommands (`archive --format=tar`, `rev-parse`, `log`, `config --get`, `remote get-url`; any other
subcommand, and the `--remote`, `--exec` and `--output` options, raise before git is called), opens
no network connection, never pushes and never uploads.  It:

1. writes the files tracked at one commit into a new directory outside the repository (no `.git`,
   no history), leaving out paths given with `--exclude`;
2. replaces the author and committer names and e-mail addresses from the history, the git
   configuration, the copyright holder in `LICENSE`, the GitHub owner and repository name from the
   `origin` remote, the local project folder name, the home-directory account name and absolute user
   paths, and any `--author`, `--identifier` or `--identifiers-file` entries, with neutral tokens;
3. scans the result and reports residual identifiers (text and binary files) and lines for a manual
   decision: parts of names, every line that held a replaced name or citation key (a co-author who
   never committed is not in the git metadata), references to an earlier submission, and matches of
   any `--review-pattern`.

Exit status 0 means nothing was found; 1, a known identifier survived; 3, no known identifier
survived but lines need a manual decision; 2, a usage error.

The report printed on standard output lists the identifiers that were collected and replaced, in
clear text.  Keep it outside the output directory and never share it with the copy.

Command for the review copy (run by the authors, on the release commit):

```bash
python scripts/prepare_anonymous_release.py --commit <release-tag> \
    --out ../anon_release/<release-tag> --author "<each co-author>" \
    --review-pattern 'CoG\s*2026' --exclude <paths decided in the manual review> \
    > ../anon_release/<release-tag>.report.txt
```

Preview of 2026-09-14 on commit b5e30fe, the head of `feature/fight-boundary-pipeline` at the time
(`--review-pattern 'CoG\s*2026'`, no `--author`; output
`C:/Users/todtj/PycharmProjects/anon_release_preview/head-b5e30fe`, report next to it as
`head-b5e30fe.report.txt`).  The directory lies outside the repository, and `git rev-parse` finds no
working tree at it or at its parent:

- 258 files (257 tracked plus the anonymisation note; 257 UTF-8 text, 1 other); 41 replacements in 16
  files (project name 22, user path 9, Korean user-folder path segment 4, repository URL 2, person
  name 2, copyright holder 1, citation key 1).
- Residual known identifiers: 0 in text files, 0 in non-text files.  Exit status 3.
- A second, independent byte search of the copy (UTF-8 and UTF-16) for the handle, the account
  name, the project folder name, the author and committer e-mail addresses and their domain, and
  the surnames and given names in `LICENSE` and the committed README found only the co-author name
  of the next item.  The remaining hits of "Users\" are the placeholder `C:\Users\<user>` in
  `docs/ACQUISITION_2026.md`.  The Korean user-folder words that are left (21 in 5 files, per the
  report) are ordinary words in Korean prose, not path segments.
- 80 lines for a manual decision:
  - 3 lines in the committed `README.md` next to a replaced author name.  They name a co-author who
    is absent from the git metadata, and the title of the earlier conference paper.  The rewritten
    README (uncommitted on 2026-09-14) no longer contains them; with `--author` the co-author's name
    is replaced in any case.
  - 24 hits of the earlier-submission patterns in 19 files.  19 are real: submission numbers and
    meta-review mentions in code comments (`gameplay/labels.py` and several `scripts/*_v33.py`),
    planning documents, `docs/AUDIT.md` (which also names a second conference paper number) and the
    header of the manuscript's `main.tex`.  5 are false positives: the paper number of a cited article
    in `definition_refs.bib` and `references_audit.md`, and three batch-size settings quoted from the
    TabNet paper ("paper 512", "paper 4096", "paper 16384") in `scripts/run_deep_tabular_baselines.py`
    and `tests/test_deep_tabular_baselines.py`.
  - 53 matches of `CoG\s*2026` in 28 files, including the `cog2026` preset name and its tests,
    `docs/CoG2026_Paper.md`, `docs/AUDIT.md` and the manuscript sources.
- Uncommitted files (among them this file, `README.md`, `sec_availability.tex` and the script itself)
  are not in the preview, because the script exports a commit.  They were checked separately by
  applying the script's own rules (`collect_identifiers`, `build_rules`, `apply_rules`,
  `find_hits`) to their text: no known identifier survives in any of the four.  Review hits:
  `README.md` 8 (`CoG 2026` and the `cog2026` preset and tag names in the opening paragraph, the
  configuration note and History), this file 8 (its own description of the review hits in the
  list above), `sec_availability.tex` 0, the script 0.

The script was changed on 2026-09-14 after the first preview, in two ways that do not alter the
export or the replacements.  (1) The line-context check now skips lines that already held a neutral
token before replacement; previously the script's own token constants would have listed four of
its lines once it is committed.  (2) Comments and help texts no longer spell out examples that its
own patterns match (a submission number, the project folder name, a venue and year), so the script
lists none of its own lines for review.  `head-b5e30fe-v2` is the same run with the changed script;
apart from the output path its report is identical to that of `head-b5e30fe`.  The same run on commit
1073858 (`head-1073858-w5`, and earlier `head-1073858-final`) gave 256 files and otherwise identical
counts.  Commit ac13e86, which became the branch head later that day (`head-ac13e86`), gave 260 files,
the same 41 replacements, 0 residual identifiers and 81 review lines: the one extra line is the
`CoG 2026` in the header comment of the newly committed `sec_background.tex`.  Earlier previews in the same folder (`head-dc243cf`, from a previous version of the script,
and `head-eb5056a`) can be deleted by the authors.

## 8. Open items

- Anonymised review link or supplementary archive, in the form the editorial office accepts
  (`\pending{anon-url}` in `sec_availability.tex`).
- Public repository URL, release tag and archival DOI after review (`\pending{release-doi}`).
- Licence of the released data files, and whether to ask Riot for written confirmation that the
  per-engagement tables may be shared for research (`\pending{data-licence}`).
- Merge the three learner-comparison scripts of section 2.1 into the release branch.
- Decide, for the review copy, what to do with the 80 manual-review lines of section 7 (53
  `CoG\s*2026` matches, 24 earlier-submission hits of which 19 are real, 3 README lines), plus the
  review hits of the files still uncommitted on 2026-09-14 (section 7).
- Keep or remove the array `y` in the released shards (section 2.2); if it is removed, delete the
  sentence on it in `sec_availability.tex`.
- Bibliography (not edited by this item): the entries `riotapi2024matchv5` and `riotapi2024timeline`
  give URLs on the page `https://developer.riotgames.com/apis` (fragments `#match-v5` and
  `#match-v5/GET_getTimeline`) with an access date of 11 September 2026.  That page returns a
  script-rendered shell to a plain HTTP client; the Match-V5 fields cited in section 2.2 were read at
  `https://developer.riotgames.com/api-details/match-v5` on 2026-09-14.  The bibliography owner
  should align URL and access date.
- `docs/DATA_MANIFEST.md`, v3.3 corpus row, lists the five label columns but not the array `y`.
- Deep-learner rows: the published TabNet predictions were trained with the sparsity term
  subtracted from the loss, and the published SAINT evaluation batches let attention reach other
  engagements of the same match (`scripts/run_deep_tabular_baselines.py` docstring).  The corrected,
  capacity-matched and tuned runs are in progress; their prediction files replace the published ones
  in the release.
- `docs/DATA_MANIFEST.md` gives 205,884 matches for the main cache; the count is 210,000 (section 4).
