# Rule-anchored constants R and B: sources and patch-note check

First checked 2026-09-11; revised 2026-09-14 (item W3-definition) for the ToG manuscript section
`docs/tog_manuscript/sec_definition.tex`. Second pass 2026-09-14, after the W3 fact-check: the OpenDota history in
section 4.1 was re-read from the GitHub REST API and raw files (3b6a7cd rename, 3f9237b addition to odota/parser,
570cfc0 removal from odota/core, constants at e2e4432 and on odota/parser master de9d6b2) and agrees with the table;
no other change to this file.
Third pass 2026-09-14 (item F1-cross-file-fixes): section 4 gains the turret, plate and elemental-drake rule values in force
in 15.14-15.16 (before patch 26.1), because the label's price variants and `docs/tog_manuscript/sec_background.tex` rely on
them and the current wiki tables show post-26.1 values; section 4.1 (OpenDota history) is unchanged.
Corpus: Match-V5 game versions 15.14, 15.15, 15.16 (Riot's displayed patch numbers 25.14, 25.15, 25.16; the
mapping is in section 3).
Detector values: `core/presets.py` preset `v3.3` and `D:/LOL_Project/fusion_2615/corpus_shards_v33/manifest.json`
(`TF2_VALIDITY_RADIUS 1600.0`, `TF2_ENGAGE_PRE_KILL_MS 15000`).
Citation keys: `docs/references/definition_refs.bib` (bib section 10); per-entry evidence in
`docs/tog_manuscript/references_audit.md` sections 2a, 2c and 3.10.

Verdict wording for the manuscript: **"no change listed"** in the patch notes. Nothing here shows that the rules
were *verified unchanged*; patch notes can omit changes, and hotfixes announced elsewhere were not searched.

---

## 1. The two rules

| Constant | Detector key (v3.3) | Value | Game rule | Source (bib key; wiki revision) | What the page says (paraphrase) | Patch history on the page |
|---|---|---|---|---|---|---|
| R | `TF2_VALIDITY_RADIUS` | 1,600 u | champion-kill experience sharing radius | https://wiki.leagueoflegends.com/en-us/Experience_(champion) (`lolwiki2026experience`; revision 4053165, 2026-08-22) | Experience from a dying champion goes to enemy champions within 1,600 units of the death location; minions use 1,500 units. | Lists V25.S1.1 (minion range 1,400 to 1,500) and V4.11 (lane minion range 1,250 to 1,400). No entry for the champion-kill radius. |
| B | `TF2_ENGAGE_PRE_KILL_MS` | 15 s | assist (kill-credit) window | https://wiki.leagueoflegends.com/en-us/Assist (`lolwiki2026assist`; revision 4016680, 2026-05-12) | On Summoner's Rift a contribution counts for 15 seconds (20 s on Howling Abyss); a new contribution restarts the timer. | No patch-history section. |
| B (corroboration) | | 15 s | kill-credit eligibility | https://wiki.leagueoflegends.com/en-us/Kill (`lolwiki2026kill`; revision 4053216, 2026-08-23) | Mentions the 15 s Summoner's Rift window for credit. | No patch entry for this window. |
| B (corroboration) | | 15 s | OpenDota team-fight start offset | odota/core `processors/processTeamfights.js` at commit e2e4432 (`opendota2018processteamfights`) | A fight opened by a hero death at time t starts at t - 15; it closes once 15 pass without a hero death; fights with fewer than 3 deaths are dropped. | See section 4 for the file's history. |

Values and patch-history statements for the three wiki pages were first read on 2026-09-11 and re-read on the
cited revisions on 2026-09-14 by item W1b (`references_audit.md` 2c, rows for this file's lines 17-19); they agree.
The wiki gives current values and does not date them to 25.14-25.16. The patch-note check in section 2 is what
ties them to the corpus patches, and it can only say that no change was listed.

## 2. Patch-note check (official notes plus wiki patch pages)

Question asked of every page: does it list a change to (a) the champion-kill experience sharing radius or
experience distribution, (b) the assist or kill-credit window, (c) champion kill gold, bounty or shutdown rules?

| Patch (game version) | Official notes (bib key) | Published | Mid-patch section on the official page | Wiki patch page (bib key) | Wiki hotfixes | (a) experience radius | (b) assist window | (c) kill gold / bounty |
|---|---|---|---|---|---|---|---|---|
| 25.14 (15.14) | https://www.leagueoflegends.com/en-us/news/game-updates/patch-25-14-notes/ (`riot2025patch2514`) | 2025-07-15 | none | https://wiki.leagueoflegends.com/en-us/V25.14 (`lolwiki2026v2514`; release 2025-07-16) | 2025-07-21 (Riven bug; Battle of Koeshin story-mode requirement) | no change listed | no change listed | **Change listed:** champion bounty accrual from farming 1 per 17.5 g to 1 per 20 g, and bounty suppression bounds widened by 25 %. This is bounty gold, not R or B. The label reads kill gold from each event's `bounty` + `shutdownBounty`, so no label constant depends on it. |
| 25.15 (15.15) | https://www.leagueoflegends.com/en-us/news/game-updates/patch-25-15-notes/ (`riot2025patch2515`) | 2025-07-29 | none | https://wiki.leagueoflegends.com/en-us/V25.15 (`lolwiki2026v2515`; release 2025-07-30) | none listed | no change listed | no change listed | no change listed |
| 25.16 (15.16) | https://www.leagueoflegends.com/en-us/news/game-updates/patch-25-16-notes/ (`riot2025patch2516`) | 2025-08-12 | none | https://wiki.leagueoflegends.com/en-us/V25.16 (`lolwiki2026v2516`; release 2025-08-13) | 2025-08-13 (Kled, Singed), 2025-08-14 (Sylas) | no change listed | no change listed | no change listed |

**Method and its limits.**

- 2026-09-11: each of the six pages was read with an automated fetch that summarises the page text. For all six
  it reported no mention of an experience range, an assist window or kill credit; the only related entry was the
  25.14 bounty change.
- 2026-09-14 (item W1b, recorded in `references_audit.md` 3.10): the three official notes were downloaded with a
  plain HTTP client and string-searched in full. The 25.14 notes contain no "assist", no "credit" and no experience
  range or radius; the 25.15 and 25.16 notes contain none of these and no "bounty". The hotfix lists come from the
  wiki version pages.
- **A person's reading of the six pages is still not on record.** Before submission a person should open them and
  search for "experience", "assist", "credit" and "bounty". Hotfixes posted only on social media or developer blogs
  were not searched.

## 3. Mapping displayed patch numbers to game versions

The manuscript relies on: **Riot's displayed patches 25.14, 25.15 and 25.16 are the releases that the Match-V5
`gameVersion` field records as 15.14, 15.15 and 15.16.**

- Two numbering schemes exist: Riot's Patch 25.S1.1 notes (`riot2025patch25s11`) say the client now displays the
  seasonal patch number, with a setting to show the old one, and the wiki V25.S1.1 and V25.09 articles give the old
  numbers 15.1 and 15.9 in their first sentence. No Riot page read states 25.14 = 15.14 in words.
- **Evidence for 25.14 = 15.14 (Riot data plus the corpus), checked 2026-09-14 by this item:**
  - Data Dragon champion data (`riot2025ddragon1514`): https://ddragon.leagueoflegends.com/cdn/15.13.1/data/en_US/champion.json
    lists 170 champions without Yunara; https://ddragon.leagueoflegends.com/cdn/15.14.1/data/en_US/champion.json lists
    171 with Yunara (key 804). The version list (https://ddragon.leagueoflegends.com/api/versions.json) runs
    15.13.1, 15.14.1, 15.15.1, 15.16.1, 15.17.1 without gaps.
  - Riot's Patch 25.14 notes introduce Yunara (`references_audit.md` 3.10, row `riot2025patch2514`).
  - Corpus: champion 804 appears in 108 of 500 matches sampled (`random.Random(1).sample`) from the 74,673 match ids
    with value "15.14" in `D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13_patch_index.json` (champion ids
    from `static_meta.champion_by_pid` of each `.meta.json`). Item W1b's independent draw found 889 of 4,000.
- 25.15 = 15.15 and 25.16 = 15.16 follow by succession (wiki Prev/Next links; Data Dragon versions in turn).
- The third-party pages cited on 2026-09-11 (gameleap.com, escorenews.com) are not allowed sources and are no longer
  used.
- **Status: VERIFIED for 25.14 = 15.14 from Riot data and the corpus; 25.15 and 25.16 by succession.**

## 4. Other game-rule and source values cited in the section

| Value | Used for | Source (bib key; revision) | Patch dating |
|---|---|---|---|
| Champion sight radius 1,350 u | nearby game quantity (R context sentence) | https://wiki.leagueoflegends.com/en-us/Sight (`lolwiki2026sight`; revision 4035710) | Current value. **Corrected 2026-09-14:** the page's patch history has one entry (V13.22, fog-of-war attacker reveal), none for this radius. **Not patch-dated.** |
| Zac E (Elastic Slingshot) range 1,200 / 1,350 / 1,500 / 1,650 / 1,800 u | nearby game quantity (R context sentence) | https://wiki.leagueoflegends.com/en-us/Zac (`lolwiki2026zac`; revision 4057435) | Page history: no range change between V25.10 and V25.20 (latest change V26.15, cooldown only). **Dated for the corpus.** The values are drawn from a data template (Template:Data Zac/Elastic Slingshot, revision 4046591). **Corrected 2026-09-14:** neither Zac page calls it the "longest non-ultimate engage", and the manuscript no longer says so ("a non-ultimate engage ability"). |
| Out of combat 5 s after the last combat action | nearby game quantity (B context sentence) | https://wiki.leagueoflegends.com/en-us/Combat_status, reached from the redirect Combat (`lolwiki2026combatstatus`; revision 4058480) | The page says *most* out-of-combat effects trigger after 5 s; no history. **Not patch-dated.** |
| Base champion kill bounty 300 g (levels 1-6; 310-420 g at levels 7-18) | motivates the 300 g label dead zone | https://wiki.leagueoflegends.com/en-us/Champion_gold_bounties (`lolwiki2026bounties`; revision 4040646) and Riot's Patch 25.09 notes (`riot2025patch2509`) | **Corrected 2026-09-14:** the page's patch history does not start at V26.03; it has the V25.09 entry that set the current base and the V25.14 accrual entry. Riot's 25.09 notes confirm the base change, and Riot's notes for 25.10-25.16 list no base-kill-gold change (`references_audit.md` N4). **"No change listed" for 15.14-15.16**, the same standard as R and B. |
| OpenDota team fight starts 15 s before the first death | corroborates B | odota/core `processors/processTeamfights.js` at e2e4432 (`opendota2018processteamfights`) | Source code, not a game rule; history below. |
| Elemental drake kill gold 25 g | "the game's rule" of the label variant `market_event_dragon_rule` (`gameplay/labels.py` `DRAGON_RULE_TEAM_GOLD = 25.0`); Table `tab:constants`, row "Price perturbation" | https://wiki.leagueoflegends.com/en-us/Dragon_pit, reached from the redirect Elemental drake (`lolwiki2026dragonpit`; revision 4000799) and Riot's Patch 26.1 notes (`riot2026patch261`) | **Dated to 15.14-15.16 (added 2026-09-14).** The page's patch history sets kill gold to 25 g in the V9.23 November 25th hotfix (from 100 g) and raises it to 75 g in V26.01; no history line between them mentions gold, bounty or reward; the 2025 entries V25.S1.2, V25.05, V25.12 and V25.19 are fixes (`references_audit.md` N9, from verbatim readings of the wikitext; a WebFetch reading by this item on 2026-09-14 agrees on the two kill-gold lines and on the absence of gold lines).  Riot's 26.1 notes give the same before-value ("Kill gold: 25 => 75"), and Riot's 25.14, 25.15 and 25.16 notes contain no "kill gold", "plate", "local gold" or "global gold" and "drake" only in a Mel bug fix (25.16) (plain HTTP download and string search, 2026-09-14).  So 25 g is the 15.14-15.16 value at the **"no change listed"** standard.  Whether the killer alone or every team member receives it is not stated on any page read (`references_audit.md` N9); the variant `market_event_dragon_rule_per_member` covers the other reading.  The current wiki value, 75 g, is post-26.1 and must not be used for the corpus. |
| Turret plating and turret gold | reading the fitted plate and turret prices (`config/game_rules/event_prices.json` pooled table: plates 120 g, outer turret 540 g, inner turret 730 g, inhibitor turret 510 g, nexus turret 185 g; regression-estimated team gold, not rule values) and `sec_background.tex` Table I row "Turret, plating" | https://wiki.leagueoflegends.com/en-us/Turret (`lolwiki2026turret`; revision 4050003) and Riot's Patch 26.1 notes (`riot2026patch261`) | **Dated to 15.14-15.16 (added 2026-09-14).** Before 26.1 only outer turrets had plates, five per turret, and plates fell off at 14 minutes (26.1 notes: "Plates are now permanent instead of falling off at 14 minutes"; inner and inhibitor turrets gained plates in 26.1).  Values replaced by 26.1 (notes' before-values): outer turret local gold 250 g, plate gold 125 g, maximum local gold 875 g, minimum 250 g; inner turret local gold 425/675 g; inhibitor turret local gold 375 g.  The Turret history dates them: V13.20 plate gold 125 g (from 175 g); V13.23 inner turret local gold 425 g (mid lane) and 675 g (side lanes), inhibitor turret 375 g, global gold of all three 25 g; between V14.1 and V25.20 the only turret-gold lines are the first-turret bonus (V14.11 raised to 300 g, V25.S1.1 removed; `references_audit.md` N8, verbatim readings), so **no first-turret bonus existed in 15.14-15.16** (restored in 26.1).  Riot's 25.14-25.16 notes mention turrets only in bug fixes and contain no plate or turret gold.  So these are the 15.14-15.16 values at the **"no change listed"** standard (`references_audit.md` N5, N8).  The current wiki table (outer and inner local gold 0, plate gold 120 g, plates on inner and inhibitor turrets, first-turret bonus 300 g) is post-26.1.  The global gold of outer and nexus turrets (50 g on the current page) is not dated to a patch.  Corpus check: every `TURRET_PLATE_DESTROYED` event of 900 sampled matches lies at an outer-turret position, at most 5 per lane, none after 14:00.03 (`sec_background.tex` header, checks C3-C4). |

### 4.1 History of OpenDota's team-fight processor (corrected 2026-09-14)

The 2026-09-11 version of this file, and `docs/DEFINITION_EVIDENCE.md` section 15, said the file was checked "before
its 2023-12 removal". That is wrong: the file was renamed in 2023-12 (`.js` to `.mjs`, commit 3b6a7cd), removed from
odota/core in 2024-01 (commit 570cfc0), and the same logic, with unchanged constants, lives on in odota/parser.
`docs/DEFINITION_EVIDENCE.md` section 15 now carries the same correction (marked "corrected 2026-09-14"). Each commit below was read through the GitHub REST API
(`https://api.github.com/repos/odota/<repo>/commits/<sha>`) and each constant in the raw file at that commit, on
2026-09-14.

| Repository and commit | Date (UTC) | Message | Change to the team-fight processor | Constants in the file at that point |
|---|---|---|---|---|
| odota/core e2e4432 | 2018-10-21 | update eslint rules (#1770) | the version pinned by the bib entry | `teamfightCooldown = 15`; `start: e.time - teamfightCooldown`; closes when `e.time - currTeamfight.last_death >= teamfightCooldown`; `filter(tf => tf.deaths >= 3)` |
| odota/core 3b6a7cd | 2023-12-02 | codemod processors to esm | `processors/processTeamfights.js` **renamed** to `processors/processTeamfights.mjs` | same four rules |
| odota/parser 3f9237b | 2024-01-06 | add blob parsing | `processors/processTeamfights.mjs` **added** to odota/parser | same four rules (current master) |
| odota/core 570cfc0 | 2024-01-07 | switch to new parser model | `processors/processTeamfights.mjs` **removed** from odota/core | -- |
| odota/parser 8ee9b58 | 2025-12-27 | implement processors in Java | `.mjs` changed in one line (rounding of death positions); Java port added in `src/main/java/opendota/CreateParsedDataBlob.java` | `.mjs`: same four rules; Java `processTeamfights`: `int teamfightCooldown = 15`, `currTeamfight.start = e.time - teamfightCooldown`, closes at `e.time - currTeamfight.last_death >= teamfightCooldown`, `filter(tf -> tf.deaths >= 3)` |
| odota/parser master de9d6b2 | 2026-08-25 | (latest commit on 2026-09-14) | -- | `processors/processTeamfights.mjs` still present with the same four rules |

Limits, as in `references_audit.md` 2a: (1) the repositories cannot show what OpenDota's servers ran when Tot et al.
fetched their labels, and Tot et al. name no version; (2) the files do not state the unit of `e.time` (seconds is
inferred from the parser's field names); (3) Tot et al. do not give the offset, so the manuscript cites
`tot2021camera` for "the labels Tot et al. used" and `opendota2018processteamfights` for the 15 s offset, never
`tot2021camera` alone.

## 5. Items previously not confirmed

- **Wiki "Teamfight" glossary entry. Resolved 2026-09-14.** On 2026-09-11 neither the rendered Terminology page
  nor its raw wikitext, as fetched, showed a "Teamfight" entry. Item W1b found the entry in section T of revision
  4057849 (2026-08-28), which predates 2026-09-11; that day's fetch probably saw a truncated text. The manuscript now
  paraphrases the entry and cites `lolwiki2026terminology`.
- Anchoring R and B to game rules does not show that a fight's spatial or temporal extent equals these values. The
  section reports how much the gate moves the corpus (`D:/LOL_Project/fusion_2615/features/fight_boundary/presence_sensitivity.json`:
  qualifying share of kill episodes from 6.5 % to 41.1 % over the R x B grid) and leaves the predictive re-run under
  v3.3, including an M = 3 point, as `\pending{presence_gate_v33}`.
