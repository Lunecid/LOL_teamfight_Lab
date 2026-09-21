# References audit: docs/references/definition_refs.bib

Item W1-references.  First written 2026-09-11; revised 2026-09-14 after a fact-check.  This file is the
per-entry table that the header of `docs/references/definition_refs.bib` points to.

History.  An earlier agent rewrote the bib file, marked every entry VERIFIED, and was cut off before
writing this table.  The 2026-09-11 pass re-checked every entry against the source named in its row and
changed the six entries whose evidence was weak or came from a disallowed source.  The 2026-09-14 pass
answered the fact-check: it replaced the non-publisher sources behind five entries, added a dated
OpenDota source for the 15 s offset, and re-ran the parse, key and `bibtex` checks (section 4 lists the
changes).  A first web-source pass on 2026-09-14 (item W1b-web-sources) added entries (bib section 10:
League of Legends Wiki pages, Riot Games patch notes, the Riot API terms and policies, Riot's Data Dragon
data) for the `\pending{cite}` placeholders that name web pages.  That pass was cut off after appending
39 entries and before writing this file's mapping.  A second web-source pass the same day (same item)
re-checked all 39 against their sources, corrected three evidence comments and extended five, added
`riot2025patch2509` (40 entries in section 10), and wrote section 2c (placeholder-to-key mapping),
section 3.10 (per-entry table) and section 4.1 (changes).  The former section 2c is now 2d, and the former
sections 4.1 and 4.2 are now 4.2 and 4.3.  A third web-source pass the same day (same item) answered a
fact-check of the second: it corrected the `lolwiki2025summonersrift` comment, added the turret and
elemental-drake rewards in force before patch 26.1 to six entry comments and to 2c (rows for
sec_limitations.tex:150-153 and 165-169, notes N8 and N9), and refreshed sections 1 and 2.  No printed
field, key or entry changed in that pass (section 4.1, items 7-11).

Allowed sources (task rule 4): publisher pages and publisher PDFs, DBLP, arXiv, DOI resolvers, and, for a
web source, the source page itself.  Terms used below:

- "Crossref" means the metadata the publisher deposited for the DOI, read at
  `https://api.crossref.org/works/<DOI>`.
- "DBLP" means the dblp record for the DOI, read through the dblp SPARQL endpoint
  `https://sparql.dblp.org/sparql`.  The dblp.org web search serves a bot check to scripts, and this pass
  did not try to get past it.
- A publisher page that is no longer online is cited through its Internet Archive capture, with the
  capture timestamp.
- Source code is cited from the publisher's own repository at a pinned commit, read through the GitHub
  REST API.

The bib file keeps the evidence for each entry in the comment block directly above it, in the form
`% [VERIFIED yyyy-mm-dd] <source> -- what it confirmed` or `% [UNVERIFIED] <reason>`.

## 1. Triage of the file (re-run 2026-09-14 after the third web-source pass)

| Check | Result |
|---|---|
| Entries parsed | 105 (65 earlier + 40 web sources), each with a type and a key; 105 at-signs in the file, all line-initial, so no stray entry starts |
| Brace balance (entry level) | balanced in all 105 |
| Duplicate keys (exact and case-insensitive) | none |
| Entries without a VERIFIED/UNVERIFIED marker in the comment block directly above | none |
| The 65 earlier entries | unchanged: `git diff --numstat` against HEAD (commit 09b89c2; the bib file was last committed in 1009ac8) reports 744 added lines and 0 deleted lines, all in the header comment and section 10 |
| Stray at-sign in a comment line, or percent sign inside an entry | none |
| Encoding and line endings | ASCII only (0 bytes above 127); LF only (0 CR bytes) |
| `bibtex` with `IEEEtran.bst` 1.14 (MiKTeX) over all entries (`\nocite{*}`) | 105 `\bibitem`s, `warning$ -- 0` |
| `pdflatex` of a test document with `cite`, `url` and `hyperref` (as in `main.tex`) and `\nocite{*}` | no errors; every `\url` in the `note` fields of section 10 typesets |

The web-source passes did not re-run the duplicate-field and required-field checks of the earlier pass;
`bibtex` reports neither problem.  The checker scripts and the test files are in the session scratchpad,
not in the repository.  Git warns that it will convert the file's LF endings to CRLF when it next touches
the working copy (Windows `core.autocrlf`); the file on disk is LF.

IEEEtran sets `misc` titles in sentence case, so every title of the new entries is braced whole to keep
names such as "Patch 25.S1.1 Notes" and "Summoner's Rift" as printed.

Two features of the IEEEtran rendering of section 10 were examined and left as they are.  (1) IEEEtran
continues the sentence after `howpublished`, so `format.note` lower-cases the first letter of a note
("League of Legends Wiki, Aug. 2026, glossary entry ...", "revision 3968911 of ...", "byline: Riot
Riru", "previous version:").  This is the style's own behaviour; a note whose value starts with a brace
would keep its capital, but that would print a capital after a comma.  (2) IEEEtran prints a dash in place
of an author name repeated from the previous reference (`riot2025ddragon1514` and the other Riot Games
entries under `\nocite{*}`).  In the manuscript this happens only when two Riot Games references are
adjacent in citation order; the dash can be switched off with an `IEEEtranBSTCTL` entry in `main.tex`,
which this item does not own.  `bibtex` reports 0 warnings.

## 2. Citation-key sanity

Every key cited in `docs/tog_manuscript/*.tex` and in `docs/references/definition_section_draft.tex` exists
in the bib file.  The check stripped comments, then collected the keys of every `\cite...{...}` and
`\nocite{...}`.  Other agents were editing the section files during these passes: the line numbers in
`sec_limitations.tex` moved between two runs on 2026-09-14, and `sec_intro.tex` and `sec_related.tex`
appeared between the first and second web-source passes.  The table below is the third web-source pass's
run (2026-09-14), repeated at the end of that pass because `sec_intro.tex` changed during it.  Since the
second pass the `sec_related.tex` and `sec_intro.tex` line numbers have moved, `chitayat2023beyond` is no
longer cited in `sec_related.tex` (a comment at line 184 says it was dropped for space), and
`sec_intro.tex` now cites two web-source keys.  Re-run the check after later section
edits.

| Cited key | Cited in (file line) | In bib |
|---|---|---|
| arik2021tabnet | sec_learners 105, 199; sec_limitations 365; sec_related 242 | yes |
| berman2014mapping | sec_definition 283; draft 152 | yes |
| bisberg2022 | sec_related 78 | yes |
| block2018narrative | sec_intro 92; sec_limitations 499; sec_related 200 | yes |
| campello2013hdbscan | sec_definition 91; draft 58 | yes |
| charleer2018dashboards | sec_intro 94, 96; sec_limitations 500; sec_related 220 | yes |
| chitayat2023beyond | sec_definition 249; sec_limitations 472; draft 134 | yes |
| costa2021 | sec_related 78 | yes |
| efron1993bootstrap | sec_intro 264 | yes |
| gorishniy2021revisiting | sec_learners 106, 174, 185; sec_limitations 359; sec_related 255 | yes |
| grinsztajn2022tree | sec_related 253 | yes |
| guo2017calibration | sec_related 304 | yes |
| halfaker2015session | sec_definition 49; sec_related 172; draft 41 | yes |
| hitar2023 | sec_related 78 | yes |
| hodge2021win | sec_intro 95; sec_limitations 486; sec_related 80 | yes |
| jacobs2021measurement | sec_definition 35, 427; draft 20, 163 | yes |
| jalovaara2024win | sec_related 96, 292 | yes |
| katona2019time | sec_learners 241; sec_related 104 | yes |
| ke2017lightgbm | sec_intro 156; sec_learners 150; sec_related 260 | yes |
| ke2022 | sec_related 105, 130; draft 15, 30, 106 | yes |
| kim2020confidence | sec_related 306 | yes |
| kokkinakis2020dax | sec_intro 93; sec_limitations 499; sec_related 202 | yes |
| lolwiki2026assist | sec_intro 202 | yes |
| lolwiki2026experience | sec_intro 201 | yes |
| maymin2021smart | sec_intro 299; sec_related 95, 133, 142, 291 | yes |
| mehrzadi2012session | sec_definition 49, 108; draft 41, 65 | yes |
| naeini2015obtaining | sec_intro 304; sec_related 288, 302 | yes |
| opendota2018processteamfights | sec_related 146 | yes |
| pedrassoli2024passive | sec_intro 94; sec_limitations 501; sec_related 204 | yes |
| pedrassoli2024wincondition | sec_intro 97; sec_limitations 501; sec_related 222 | yes |
| platt2000probabilities | sec_related 302 | yes |
| riotapi2024matchv5 | sec_intro 107 | yes |
| riotapi2024timeline | sec_intro 107; sec_related 142; draft 28 | yes |
| schubert2016encounter | sec_background 409; sec_definition 29, 388; sec_intro 91; sec_limitations 79, 498; sec_related 119, 197; draft 15, 36 | yes |
| shwartzziv2022tabular | sec_related 251 | yes |
| silva2018continuous | sec_learners 240; sec_related 79 | yes |
| silverman1981multimodality | sec_definition 51; draft 50 | yes |
| somepalli2021saint | sec_learners 105, 214; sec_limitations 359; sec_related 243 | yes |
| tot2021camera | sec_related 104, 144; draft 16, 112 | yes |
| zaliapin2013clusters | sec_definition 50, 165; draft 43, 92, 146 | yes |
| zaliapin2022perspectives | sec_definition 165, 264; draft 92 | yes |

**41 keys cited; missing keys: none.**  The other 64 entries are not cited yet.  Most of them are waiting
for the `\pending{cite}` placeholders in 2a (literature and API pages) and 2c (web sources).  Two of the
40 web-source keys are cited: `lolwiki2026experience` and `lolwiki2026assist`, in `sec_intro.tex`:201-202,
for the experience-sharing radius and the assist window, which those pages support (2c).  Every web-source
placeholder listed in 2c still shows as `\pending{cite}` until the section owners apply 2c.

### 2a. `\pending{cite}` placeholders that can now become `\cite` (for the section owners)

This item does not edit the section files.  Line numbers are from 2026-09-14.

| File:line | Placeholder names | Replace with | Support |
|---|---|---|---|
| sec_definition.tex:30 | Ke et al., IEEE CoG 2022 | `\cite{ke2022}` | The Ke et al. PDF defines a team fight as an encounter with at least one kill. |
| sec_definition.tex:180 | Ke et al. 2022 | `\cite{ke2022}` | The Ke et al. PDF requires at least two players of the same team on one side. |
| sec_definition.tex:31 | Tot et al., IEEE CoG 2021 | `\cite{tot2021camera}` | Tot et al. (Section IV.A) take team-fight labels for 1,457 matches from the OpenDota API. |
| sec_definition.tex:42 | Riot Games, Match-V5 timeline API documentation | `\cite{riotapi2024timeline}` | Endpoint page. |
| sec_definition.tex:192 | Tot et al. 2021; OpenDota processTeamfights.js | `\cite{tot2021camera,opendota2018processteamfights}`, never `\cite{tot2021camera}` alone | See the note below this table. |
| sec_background.tex:341 | Riot Developer Portal, Match-V5 reference and timeline endpoint | `\cite{riotapi2024matchv5,riotapi2024timeline}` | Endpoint pages. |
| sec_availability.tex:95 | Riot Match-V5 and Timeline API documentation | `\cite{riotapi2024matchv5,riotapi2024timeline}` | Endpoint pages. |
| sec_learners.tex:103 | Grinsztajn et al. 2022; Shwartz-Ziv and Armon 2022 | `\cite{grinsztajn2022tree,shwartzziv2022tabular}` | |
| sec_learners.tex:234 | Cho et al. 2014, gated recurrent unit | `\cite{cho2014learning}` | |
| sec_learners.tex:238 | Vaswani et al. 2017, Transformer | `\cite{vaswani2017attention}` | |
| sec_learners.tex:363 | Efron and Tibshirani 1993, bootstrap | `\cite{efron1993bootstrap}` | |
| sec_learners.tex:392 | Loshchilov and Hutter 2019, AdamW | `\cite{loshchilov2019decoupled}` | |

**sec_definition.tex:192 must not become a Tot-only `\cite`.**  The sentence says that B = 15 s equals the
offset at which the OpenDota team-fight label used by Tot et al. starts before the first death.  Tot et al.
do not give that offset.  They say only that OpenDota's labels are inconsistent about start and end times.
The two citations support different halves of the sentence:

- `tot2021camera` supports "the OpenDota team-fight label used by Tot et al.".
- `opendota2018processteamfights` supports the 15 s offset.  In OpenDota's team-fight processor at commit
  e2e4432, a fight opened by a hero death at time t starts at t - 15 (`teamfightCooldown = 15`).  The
  fight closes once 15 pass without a hero death, and only fights with at least 3 deaths are kept.

Three limits remain, and the section owner should state them or word around them:

1. That commit stood in the repository from 2018-10-21 to 2022-09-20.  The span contains Tot et al.'s CoG
   2021 conference, but the repository cannot show what OpenDota's servers ran when Tot et al. fetched
   their labels, and Tot et al. name no version.
2. The processor file does not state the unit of its time field.  OpenDota's parser fills that field with
   rounded game-time values, so the unit is seconds, but this is inferred from the field names; neither
   file says so.
3. The source comment at sec_definition.tex:193 points to `docs/DEFINITION_EVIDENCE.md` section 15 (line
   745), which says the version checked was the one just before a 2023-12 deletion.
   `docs/references/rule_constants_evidence.md` line 59 says "version before its 2023-12 removal".  The
   history does not support a deletion in 2023-12.  Commit 3b6a7cd renamed the file to
   `processTeamfights.mjs` on 2023-12-02.  Commit 570cfc0 removed it from odota/core on 2024-01-07, and
   the same logic now lives in odota/parser.  The owners of those two documents should correct them.  The
   constants do not change: they are the same before and after both commits.

### 2b. `\pending{cite}` placeholders with no entry

None remain.  The web pages that this list named on 2026-09-11 (League of Legends Wiki pages, Riot Games
patch notes 25.14, 25.15, 25.16 and 26.1, the Riot Games API Terms and Conditions) now have entries in
bib section 10; section 2c maps each placeholder to its keys.  One web source is named only in a comment
and has no entry: the IEEE ToG submission guidelines page (sec_availability.tex:17), which no sentence
cites.

### 2c. Web-source placeholders and the keys that resolve them (item W1b-web-sources)

This item does not edit the section files.  Line numbers were re-checked by `grep` in the third web-source
pass (2026-09-14) and had not moved since the second; other agents were editing the files at the time.
Every key below is in bib section 10 and is marked VERIFIED; section 3.10 lists the entries.  "Replace
with" gives the citation that covers every page the placeholder names; the last column says what the keys
support and what they do not.  Every `\pending{cite}` row can become `\cite` now.  Of the two
`\pending{rulefact}` rows, sec_limitations.tex:167 (turret rewards) can be filled at the "no change
listed" standard, and sec_limitations.tex:153 (drake gold) can be narrowed to the recipient question.  Notes N1 to
N9 follow the table.

Scan.  `\pending{cite}` markers and bare mentions of League of Legends Wiki pages, Riot patch notes, Riot
API terms or policies, Data Dragon and `leagueoflegends.com` / `riotgames.com` URLs were collected from
`docs/tog_manuscript/*.tex` and `docs/references/rule_constants_evidence.md`.  `sec_intro.tex`,
`sec_related.tex`, `sec_learners.tex` and `main.tex` have no web-source placeholder.  `sec_intro.tex`:201-202
already cites `lolwiki2026experience` (a champion's death shares experience within a radius; the page gives
1,600 u) and `lolwiki2026assist` (the assist window; the page gives 15 s on Summoner's Rift); both
citations are supported.  Placeholders for
literature and for the Match-V5 API pages are in 2a.

| File:line | Placeholder names | Replace with | Support and limits |
|---|---|---|---|
| sec_background.tex:259 | LoL Wiki pages Summoner's Rift, Champion, Sight, Movement speed | `\cite{lolwiki2025summonersrift,lolwiki2026champion,lolwiki2026sight,lolwiki2026movementspeed}` | Table I, map and players.  The Summoner's Rift page does not state the team size (N1).  Movement-speed numbers are template-drawn (N3). |
| sec_background.tex:260 | LoL Wiki pages Gold, Experience (champion), Minion, Creep score | `\cite{lolwiki2026gold,lolwiki2026experience,lolwiki2026minion,lolwiki2026farming}` | Creep score redirects to Farming. |
| sec_background.tex:261 | LoL Wiki pages Kill, Death, Champion gold bounties, V25.14 | `\cite{lolwiki2026kill,lolwiki2026death,lolwiki2026bounties,lolwiki2026v2514}` | Add `riot2025patch2509` if the row dates the 300 g base bounty (N4). |
| sec_background.tex:262 | LoL Wiki pages Turret, Inhibitor, Nexus; Riot Games Patch 26.1 notes | `\cite{lolwiki2026turret,lolwiki2026inhibitor,lolwiki2026nexus,riot2026patch261}` | The 26.1 notes give the pre-26.1 plate rule (outer turrets only; plates fall off at 14 minutes) and the pre-26.1 turret gold (N8). |
| sec_background.tex:263-264 | LoL Wiki pages Dragon, Elder Dragon, Aspect of the Dragon, Voidgrub, Rift Herald, Atakhan, Baron Nashor, V25.09 | `\cite{lolwiki2026dragonpit,lolwiki2026elderdragon,lolwiki2026dragonslayer,lolwiki2026voidgrub,lolwiki2026riftherald,lolwiki2026atakhan,lolwiki2026baron,lolwiki2026v2509}` | Redirects: Dragon to Dragon pit, Aspect of the Dragon to Dragon Slayer, Voidgrub to Voidgrub camp.  `riot2025patch2509` also gives the 8:00 Voidgrub and 15:00 Rift Herald spawn times. |
| sec_background.tex:265 | LoL Wiki pages Sight, Ward | `\cite{lolwiki2026sight,lolwiki2025ward}` | |
| sec_background.tex:279 | LoL Wiki: Summoner's Rift | `\cite{lolwiki2025summonersrift}` | Supports the sides, lanes, jungle and river, not "two teams of five players" (N1). |
| sec_background.tex:286 | LoL Wiki: Nexus | `\cite{lolwiki2026nexus}` | The page's invulnerability rule gives the sentence's condition. |
| sec_background.tex:290-291 | LoL Wiki: Gold; Experience (champion) | `\cite{lolwiki2026gold,lolwiki2026experience}` | |
| sec_background.tex:295 | LoL Wiki: Minion | `\cite{lolwiki2026minion}` | |
| sec_background.tex:308 | LoL Wiki: Kill; Death; Champion gold bounties | `\cite{lolwiki2026kill,lolwiki2026death,lolwiki2026bounties,lolwiki2026experience}` | The sentence also says nearby enemy champions share the victim's XP; that is on the Experience (champion) page (the comment at line 301 names it), so its key is added. |
| sec_background.tex:312-313 | LoL Wiki: Sight; Ward | `\cite{lolwiki2026sight,lolwiki2025ward}` | |
| sec_background.tex:325 | LoL Wiki: V25.S1.1, V25.14, V25.15, V25.16 | `\cite{lolwiki2026v25s11,lolwiki2026v2514,lolwiki2026v2515,lolwiki2026v2516,riot2025patch25s11,riot2025ddragon1514}` | The wiki V25.14-V25.16 pages do not give the numbers 15.14-15.16; the correspondence needs the Data Dragon entry and the corpus check (N2).  The three dates are the wiki infobox release dates (N2). |
| sec_background.tex:329-330 | LoL Wiki: Atakhan; Riot Games, Patch 26.1 notes | `\cite{lolwiki2026atakhan,riot2026patch261}` | The 26.1 notes remove Atakhan and make plates permanent instead of falling off at 14 minutes. |
| sec_definition.tex:27 | League of Legends Wiki, Terminology page, teamfight entry | `\cite{lolwiki2026terminology}` | The entry "Teamfight" is in section T of revision 4057849 (2026-08-28), which predates 2026-09-11.  The placeholder's "not re-located on 2026-09-11" is out of date; the rendered page, as fetched, is cut off before section T.  The sentence's paraphrase matches the entry. |
| sec_definition.tex:187 | League of Legends Wiki, Experience (champion) and Assist pages | `\cite{lolwiki2026experience,lolwiki2026assist}` | 1,600 u (champion deaths; minions 1,500 u) and 15 s on Summoner's Rift.  Access dates: N6. |
| sec_definition.tex:198 | Riot Games, Patch 25.14, 25.15 and 25.16 notes | `\cite{riot2025patch2514,riot2025patch2515,riot2025patch2516,lolwiki2026v2514,lolwiki2026v2515,lolwiki2026v2516}` | The sentence also says the hotfixes listed for the patches were checked; those lists are on the wiki version pages, hence their keys.  A string search of the full text of the three notes finds no "assist", no "credit" and no experience range or radius. |
| sec_definition.tex:203 | League of Legends Wiki, Sight, Zac and Combat pages | `\cite{lolwiki2026sight,lolwiki2026zac,lolwiki2026combatstatus}` | Combat redirects to Combat status, which says most (not all) out-of-combat effects start after 5 s.  Zac's range is template-drawn (N3). |
| sec_limitations.tex:150 | League of Legends Wiki, "Elemental drake" page | `\cite{lolwiki2026dragonpit,riot2026patch261}` | Elemental drake redirects to Dragon pit.  Its V26.01 entry raises elemental drake kill gold to 75 from 25, and Riot's 26.1 notes give the same before-value.  The page's patch history also dates the 25: the V9.23 November 25th hotfix set it (back from 100), and no gold entry follows until V26.01; the 2025 entries (V25.S1.2, V25.05, V25.12, V25.19) are fixes.  So the value for 15.14-15.16 is 25 g, "no change listed" (N9).  The sentence at lines 150-151 ("states neither the value in force in 15.14--15.16 nor ...") overstates the gap: only the recipient is unknown. |
| sec_limitations.tex:153 (`\pending{rulefact}`) | Patch-dated gold of an elemental drake kill for 15.14-15.16, and whether it is paid per team member | Partly fillable: value from `lolwiki2026dragonpit` and `riot2026patch261` | Fill the value: 25 g, set in the V9.23 hotfix, with no change listed through 25.16.  The 25.14-25.16 notes list no drake gold change; their only drake hit is a Mel bug fix in 25.16.  Keep a narrowed marker for the recipient only: no page read says whether the killer or every team member receives it (N9). |
| sec_limitations.tex:165-166 | League of Legends Wiki, "Turret" page | `\cite{lolwiki2026turret,riot2026patch261}` | The values in the sentence (outer 50 g and inner 25 g global, no local gold) are the current, post-26.1 table.  "No local gold" holds only from 26.1.  Before 26.1, Riot's 26.1 notes give an outer turret 250 g local gold, 125 g per plate and at most 875 g local gold in total, an inner turret 425/675 g local gold, and an inhibitor turret 375 g.  The Turret history gives the inner value as 425 g mid lane and 675 g top and bottom lanes (N8).  The 25.14-25.16 notes list no turret or plate gold change.  The first-turret bonus mentioned at line 175 did not exist in 15.14-15.16 (N5). |
| sec_limitations.tex:167 (`\pending{rulefact}`) | Patch-dated turret rewards for 15.14-15.16 | Partly fillable: `riot2026patch261`, `lolwiki2026turret` | Fill the local and plate values of N8 as "in force in the last 2025 patch (26.1 notes); set in V13.20/V13.23 for plates, inner and inhibitor turrets; no change listed in the 25.14-25.16 notes".  Inner and inhibitor global gold (25 g) is dated too: V13.23, no change listed since.  The global gold of outer and nexus turrets (50 g on the current page) is not dated to a patch, but no history entry from V13.20 to the newest, V26.03, changes it: "no change listed", as for R.  If the owner wants patch-dated values only, keep a marker for that one value.  The notes' before-values describe the last 2025 patch; they are not dated to 15.14-15.16 by themselves. |
| sec_limitations.tex:168-169 (source comment) | "no patch history for 15.x" | none; correct the comment | Wrong: the Turret page's patch history has V25.S1.1 (first-turret bonus removed) and V26.01 (bonus restored; outer and inner local gold to 0; plates 125 to 120) entries, among other 2025 entries (N5, N8). |
| sec_limitations.tex:171-173 (the sentence after the Turret citation) | The regression ranks the inner turret above the outer one, "the reverse of that listing" | none | The listing it reverses is the post-26.1 table.  Before 26.1 an inner turret paid 425 or 675 g local gold against an outer turret's 250 g, plus up to 625 g from its plates before 14:00 (N8).  The section owner should re-check the comparison against these values. |
| sec_limitations.tex:392-393 | League of Legends Wiki, "Experience (champion)", "Assist" and "Kill" pages | `\cite{lolwiki2026experience,lolwiki2026assist,lolwiki2026kill}` | The three revisions (2026-08-22, 2026-05-12, 2026-08-23) predate 8 September 2026, so both readings the sentence mentions saw the same wikitext (N6). |
| sec_limitations.tex:400-401 | Riot Games, patch 25.14, 25.15 and 25.16 notes | `\cite{riot2025patch2514,riot2025patch2515,riot2025patch2516}` | As for sec_definition.tex:198. |
| sec_availability.tex:73 | Riot Games API Terms and Conditions | `\cite{riot2013apiterms}` | Materials include copies, portions, extracts and derivatives of Game Information; a string search finds no "research" and no "dataset". |
| sec_availability.tex:85 | Riot Games API Terms and Conditions | `\cite{riot2013apiterms}` | Use only as authorised; delete all Game Information on termination; deletion requests passed to developers as identifier lists (GDPR section). |
| sec_availability.tex:12-16 (comment only) | Riot General Policies | `riot2025devpolicies`, if a sentence cites the policies | No sentence cites it.  The comment's "none is in definition_refs.bib yet" is out of date (N7). |
| rule_constants_evidence.md:17-19 (section 1) | Wiki Experience (champion), Assist, Kill | `lolwiki2026experience`, `lolwiki2026assist`, `lolwiki2026kill` | Values and patch-history statements re-read on the cited revisions; they agree with the table. |
| rule_constants_evidence.md:31-33 (section 2) | Riot notes 25.14-25.16; wiki V25.14-V25.16 | `riot2025patch2514`, `riot2025patch2515`, `riot2025patch2516`, `lolwiki2026v2514`, `lolwiki2026v2515`, `lolwiki2026v2516` | That section asks a person to search the six pages.  This item ran an automated string search of the full text of the three downloaded notes instead (results in the entries' comments): no change listed to R or B.  A person's reading is still not on record. |
| rule_constants_evidence.md:42-49 (section 3) | Riot 25.S1.1 notes; third-party pages gameleap.com and escorenews.com | `riot2025patch25s11`, `riot2025ddragon1514` | The two third-party pages are not allowed sources.  The Data Dragon entry and the corpus check (N2) confirm 25.14 = 15.14 from Riot data. |
| rule_constants_evidence.md:55-58 (section 4) | Wiki Sight, Zac, Combat, Champion gold bounties | `lolwiki2026sight`, `lolwiki2026zac`, `lolwiki2026combatstatus`, `lolwiki2026bounties`, `riot2025patch2509` | Two statements there are wrong: the Sight page has one patch-history entry (V13.22, unrelated), and the bounties history does not start at V26.03 (N4).  Line 56 still calls Zac's Elastic Slingshot the "longest non-ultimate engage"; neither Zac page supports that, and sec_definition.tex:200-201 no longer says it ("a non-ultimate engage ability"). |
| sec_definition.tex:377 (source comment, row "Dead zone") | Champion gold bounties, "not patch-dated for 25.14-25.16" | `lolwiki2026bounties`, `riot2025patch2509` | Out of date: the 300 g base (310-420 g at levels 7-18) was set in 25.09, and Riot's notes for 25.10-25.16 list no base-kill-gold change (N4).  Riot's 25.09 notes say base kill gold rises by 10 g per level from level 7 to 18, to at most 420 g. |
| rule_constants_evidence.md:63-67 (section 5) | Wiki Terminology "Teamfight" entry (`\pending{cite}`) | `lolwiki2026terminology` | The entry exists in revision 4057849 of 2026-08-28, so it was on the page on 2026-09-11; that day's fetch probably saw a truncated text. |

Notes for the section owners.

- **N1 (team size).**  sec_background.tex:278 and sec_intro.tex:88 (line 67 at the second pass) say two
  teams of five players.  No key
  in the bib file supports the number five, and `lolwiki2025summonersrift` must not be cited for it.  That
  page's Environment section speaks of two teams of champions without a number, and its lead says only
  that the map is the first and most popular one, used for Classic queues and esports.  The rendered page,
  as fetched, mentions five-a-side only in game-mode names (5v5).  Cite the ten participants of each
  Match-V5 match record, or reword.  **Applied 2026-09-14 (item F1-cross-file-fixes):** both sentences of
  sec_background.tex that give the team size (the Table I row "Team, side, champion, role" and the first
  sentence of subsection sec:background-game) now cite `riotapi2024matchv5` for the ten participants of each
  match record; `lolwiki2025summonersrift` is cited there only for the sides and `lolwiki2026nexus` only for the
  win condition.  The API page documents the participant list and its `teamId` but gives no count; the count
  is the corpus's (check C1 in the sec_background.tex header), as the `riotapi2024matchv5` row of 3.4 says.
- **N2 (patch numbers and dates).**  The wiki V25.S1.1 and V25.09 articles give the old numbers 15.1 and
  15.9 in their first sentence; the V25.14-V25.16 articles give no old number.  25.14 = 15.14 rests on
  Riot data: Data Dragon has no Yunara at 15.13.1 and Yunara (key 804) at 15.14.1; Riot's Patch 25.14
  notes introduce Yunara; and champion 804 appears in 889 of 4,000 corpus matches of game version 15.14
  (`random.seed(0)`, `random.sample` over the 74,673 ids with value 15.14 in
  `D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13_patch_index.json`; champion ids from
  `static_meta.champion_by_pid` in each `.meta.json`; re-run 2026-09-14).  25.15 = 15.15 and 25.16 = 15.16
  follow by succession (wiki Prev/Next links; Data Dragon versions 15.14.1, 15.15.1, 15.16.1 in turn); no
  Riot page says so in words.  sec_background.tex:324 gives release dates 16 July, 30 July and
  13 August 2025: those are the wiki infobox dates.  Riot published the notes on 15 July, 29 July and
  12 August 2025 (18:00 UTC).
- **N3 (template-drawn values).**  The movement-speed range (325-355) and Zac's Elastic Slingshot range
  (1,200-1,800) are not in the cited pages' own wikitext; the pages draw them from champion data
  templates.  A permanent link fixes the page text but not those values.  For Zac, the entry comment names
  the template revision (Template:Data Zac/Elastic Slingshot, revision 4046591 of 2026-07-28) and the patch
  history subpage (Zac/Patch history, revision 4046566 of 2026-07-28).
- **N4 (base bounty dating).**  The base bounty of 300 g at levels 1-6 (310-420 g at levels 7-18) was set
  in patch 25.09: the wiki bounties history (V25.09) and Riot's Patch 25.09 notes (`riot2025patch2509`)
  agree.  Riot's notes for 25.10 to 25.16 list no change to base kill gold (string search, see the entry).
  So the "Dead zone" source comment at sec_definition.tex:377 ("not patch-dated") can become "set in
  25.09; no change listed in the notes for 25.10-25.16".  The comment at sec_background.tex:139-140 says
  the page's /History subpage lists no change after V14.21.  That subpage (revision 3921339 of 2025-06-29)
  was not read, but the page's own patch history (revision 4040646) has the V25.09 base change and the
  V25.14 accrual change, so the comment should cite that history instead.  The row text at sec_background.tex:148-149 says 25.14
  slowed the accrual of shutdown gold; the notes change only the bounty accrued from farming (1 per 17.5
  gold to 1 per 20 gold) and widen the suppression thresholds by 25 percent.
- **N5 (first-turret bonus).**  The 300 g first-turret bonus did not exist in 15.14-15.16.  Riot's Patch
  25.S1.1 notes replace the First Blood and First Turret gold bonuses with Feats of Strength, whose rewards
  are boot upgrades; Riot's Patch 26.1 notes remove Feats of Strength and say the first turret destroyed
  once again grants 300 g; the wiki Turret history has both entries, and no note or wiki entry read in
  between restores the bonus.  sec_limitations.tex:174-176 contrasts the negative first-turret coefficient
  with the first-turret bonus on the current wiki page.  That bonus is a post-26.1 value, so the sentence
  should not imply that the corpus patches paid it.
- **N6 (access dates).**  The placeholders say "retrieved 2026-09-11" or "accessed 11 Sep. 2026"; the
  entries print "Accessed: Sep. 14, 2026" and a revision.  Every cited wiki revision is dated on or before
  2026-09-03, so readings on 8 and 11 September saw the same page text; the prose dates can stay.
- **N7 (stale source comments).**  sec_background.tex:15-16 and sec_availability.tex:12 say no wiki or Riot
  source is in the bib file; that is no longer true.  sec_limitations.tex:168-169 says the Turret page has
  no patch history for 15.x; it has (N8).
- **N8 (turret rewards before 26.1).**  The sentence at sec_limitations.tex:164-167 gives the current
  table, in which turrets pay no local gold.  That table dates from patch 26.1.  Riot's Patch 26.1 notes
  (`riot2026patch261`, re-downloaded 2026-09-14 and string-searched) list these before-values: an outer
  turret had 250 g local gold (now 0), 125 g per plate (now 120 g) and at most 875 g local gold in total
  (now 600 g); an inner turret had 425/675 g local gold and an inhibitor turret 375 g (both now 600 g
  spread across five plates).  The notes give no global turret gold.  The Turret page's patch history
  (`lolwiki2026turret`, revision 4050003, each block copied verbatim) dates most of these values.  V13.20
  set plate gold to 125 g.  V13.23 set the local gold of the mid-lane inner turret to 425 g, of the
  side-lane inner turrets to 675 g and of the inhibitor turret to 375 g, and the global gold of all three
  to 25 g.  V26.01 set outer and inner local gold to 0.  Between V14.1 and V25.20 the only turret-gold
  lines are the first-turret bonus (V14.11 raised it to 300 g, V25.S1.1 removed it).  A string search of
  Riot's 25.14, 25.15 and 25.16 notes finds "turret" only in bug fixes, "plate" only in the item name
  Experimental Hexplate, and no "local gold" or "global gold".  So the values for 15.14-15.16 are:
  - outer turret: 250 g local and 125 g per plate, at most 875 g local (the 26.1 before-values, which
    describe the last patch of 2025);
  - inner turret: 425 g (mid lane) or 675 g (top and bottom lanes) local, 25 g global;
  - inhibitor turret: 375 g local, 25 g global.

  All are "no change listed" values, not values verified patch by patch.  The 50 g global gold of outer
  and nexus turrets is not dated to a patch; no history entry from V13.20 to the newest, V26.03, changes
  it, so it is "no change listed" in the sense used for R.  The sentence at sec_limitations.tex:171-173
  compares the regression with the current listing, in which an outer turret pays more than an inner one.
  Before 26.1 an inner turret's local gold (425 or 675 g) exceeded an outer turret's 250 g, although an
  outer turret's plates could add up to 625 g before 14:00.  The comparison should be re-checked against
  these values.
- **N9 (elemental-drake kill gold).**  The Dragon pit page's patch history (`lolwiki2026dragonpit`,
  revision 4000799) has three elemental-drake kill-gold lines, copied verbatim: V8.23 raised it from 25 g
  to 100 g, the V9.23 November 25th hotfix lowered it back to 25 g, and V26.01 raised it to 75 g.  No
  history line between V9.23 and V26.01 mentions gold, bounty or reward.  The four 2025 entries
  (V25.S1.2, V25.05, V25.12, V25.19) are fixes.  The V26.01 line "global gold reduced to 150 from 250"
  belongs to the Elder Dragon.  Riot's 26.1 notes give the same before-value (kill gold 25 => 75), and
  Riot's 25.14-25.16 notes list no drake gold change.  So the value for 15.14-15.16 is 25 g, "no change
  listed", the same standard as R, B and the base bounty.  Only the recipient is still unknown: the
  page's Rewards section says only that slaying a dragon grants experience to allies.  The sentence at
  sec_limitations.tex:150-151 and its `\pending{rulefact}` should be narrowed to the recipient question.

### 2d. Differences from the superseded CoG bibliography (`paper/refer.tex`, gitignored)

- `paper/refer.tex` cites the Hodge et al. ToG paper as `hodge2021`; the bib file and the ToG sections use
  `hodge2021win`.  No other key differs.
- `pedrassoli2024wincondition` (the Dota 2 win-condition paper, `paper/refer.tex` lines 45-50): the bibitem
  there prints "F. O. Block" and "Art. 314".  Neither detail is in the entry.  No source read gives Florian
  Block a middle initial.  The article number appears only on the authors' accepted manuscript, which is
  not an allowed source.  Crossref and DBLP give pages 1-22 but no article number, and the ACM Digital
  Library page served a Cloudflare check on 2026-09-14.
- Page ranges that `paper/refer.tex` gives for NeurIPS 2017 papers are not used, because the proceedings
  records have none (`ke2017lightgbm`, `vaswani2017attention`, `hamilton2017inductive`).

## 3. Per-entry table

Status is VERIFIED for all 105 entries (65 in 3.1-3.9, 40 web sources in 3.10).  "Crossref" in the source column abbreviates
`https://api.crossref.org/works/` followed by the entry's DOI.  The date in the status column is the last
date on which the named source was read for that entry.

### 3.1 Engagement definition: boundaries, clustering, measurement

| key | work | status | source URL | notes |
|---|---|---|---|---|
| zaliapin2013clusters | Zaliapin, Ben-Zion 2013, Earthquake clusters in southern California I, JGR Solid Earth 118(6):2847-2864 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1002/jgrb.50179 | |
| zaliapin2022perspectives | Zaliapin, Ben-Zion, Perspectives on clustering and declustering of earthquakes, SRL 93(1):386-401 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1785/0220210127 | Online 2021-09-22, issue January 2022; year is the issue year. |
| halfaker2015session | Halfaker et al. 2015, User session identification based on strong regularities in inter-activity time, WWW, 410-418 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1145/2736277.2741117 | Author list follows the current ACM record; the local arXiv copy prints an earlier name for the sixth author. |
| mehrzadi2012session | Mehrzadi, Feitelson 2012, On extracting session data from activity logs, SYSTOR, 1-7 | VERIFIED 2026-09-14 | https://api.crossref.org/works/10.1145/2367589.2367592 | Place (Haifa, Israel) from the Crossref event field, which reads "Haifa Israel" (re-read 2026-09-14). The first page of the local PDF `MehrzadiFeitelson2012_SessionData_SYSTOR.pdf` prints the same place. The bib comment now says the same. |
| catledge1995browsing | Catledge, Pitkow 1995, Characterizing browsing strategies in the World-Wide Web, Comput. Netw. ISDN Syst. 27(6):1065-1073 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1016/0169-7552(95)00043-7 | |
| kleinberg2002bursty | Kleinberg 2002, Bursty and hierarchical structure in streams, KDD, 91-101 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1145/775047.775061 | |
| ester1996dbscan | Ester, Kriegel, Sander, Xu 1996, A density-based algorithm for discovering clusters in large spatial databases with noise, KDD-96, 226-231 | VERIFIED 2026-09-11 | https://aaai.org/papers/kdd96-037-a-density-based-algorithm-for-discovering-clusters-in-large-spatial-databases-with-noise/ ; https://cdn.aaai.org/KDD/1996/KDD96-037.pdf | Pages printed on the first and last pages of the PDF. Place and publisher removed on 2026-09-11 (their only source was an OSTI record). |
| campello2013hdbscan | Campello, Moulavi, Sander 2013, Density-based clustering based on hierarchical density estimates, PAKDD (LNCS), 160-172 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1007/978-3-642-37456-2_14 | LNCS volume number is not in the record and is omitted. |
| silverman1981multimodality | Silverman 1981, Using kernel density estimates to investigate multimodality, JRSS B 43(1):97-99 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1111/j.2517-6161.1981.tb01155.x | |
| hartigan1985dip | Hartigan, Hartigan 1985, The dip test of unimodality, Ann. Statist. 13(1):70-84 | VERIFIED 2026-09-11 | https://projecteuclid.org/journals/annals-of-statistics/volume-13/issue-1/The-Dip-Test-of-Unimodality/10.1214/aos/1176346577.full ; Crossref | The Crossref record has no pages; pages from Project Euclid (readable with WebFetch; plain HTTP clients get a bot check). |
| otsu1979threshold | Otsu 1979, A threshold selection method from gray-level histograms, IEEE TSMC 9(1):62-66 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1109/TSMC.1979.4310076 | |
| wiltschko2015mapping | Wiltschko et al. 2015, Mapping sub-second structure in mouse behavior, Neuron 88(6):1121-1135 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1016/j.neuron.2015.11.031 | |
| berman2014mapping | Berman, Choi, Bialek, Shaevitz 2014, Mapping the stereotyped behaviour of freely moving fruit flies, J. R. Soc. Interface 11(99), art. 20140672 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1098/rsif.2014.0672 | The record puts the article number in its page field. |
| kulldorff2001prospective | Kulldorff 2001, Prospective time periodic geographical disease surveillance using a scan statistic, JRSS A 164(1):61-72 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1111/1467-985X.00186 | |
| jacobs2021measurement | Jacobs, Wallach 2021, Measurement and fairness, FAccT, 375-385 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1145/3442188.3445901 | |
| cronbach1955construct | Cronbach, Meehl 1955, Construct validity in psychological tests, Psychol. Bull. 52(4):281-302 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1037/h0040957 | |
| mcqueen2014screens | McQueen, Wiens, Guttag 2014, Automatically recognizing on-ball screens, MIT Sloan Sports Analytics Conference | VERIFIED 2026-09-11 | https://www.sloansportsconference.com/research-papers/automatically-recognizing-on-ball-screens | Conference's own page (HTTP 200 again on 2026-09-14). It lists the three authors; the year comes from the page's 2014-conference link and the linked PDF's file name. No pages exist. |

### 3.2 Esports analytics: encounters, team fights, outcome prediction, patches

| key | work | status | source URL | notes |
|---|---|---|---|---|
| schubert2016encounter | Schubert, Drachen, Mahlmann 2016, Esports analytics through encounter detection, MIT Sloan Sports Analytics Conference, Research Papers Competition | VERIFIED 2026-09-14 | https://web.archive.org/web/20170626141008/http://www.sloansportsconference.com/content/esports-analytics-through-encounter-detection/ ; https://web.archive.org/web/20171017210220/http://www.sloansportsconference.com/wp-content/uploads/2016/02/1458.pdf | CoG R1 list. **Changed on 2026-09-14.** The fact-check was right that the Lund University record is not an allowed source. A guessed page under `/research-papers/` returns 404, and the old page URL now redirects to the conference home page. The conference's own paper page survives as an Internet Archive capture (2017-06-26), with title, three authors, abstract and a link to the conference-hosted PDF (capture 2017-10-17). That PDF has 18 pages. Page 1 prints the title, track "Other Sports", paper number 1458 and the authors in three columns: Schubert (left), Drachen (centre), Mahlmann (right). Every footer reads "2016 Research Papers Competition". The entry keeps the printed order; the conference page lists Mahlmann second. Booktitle changed to name the conference and the competition, and the url now points to the archived PDF. The Lund record and the local PDF (a Lund cover page plus the same paper) agree but are no longer the evidence. |
| ke2022 | Ke et al. 2022, DOTA 2 match prediction through deep learning team fight models, IEEE CoG, 96-103 | VERIFIED 2026-09-14 | https://api.crossref.org/works/10.1109/CoG51982.2022.9893647 | The key is fixed by `definition_section_draft.tex` and `paper/refer.tex`. Re-read 2026-09-14: the Crossref record lists the same ten authors in the same order; they also match the printed first page of `D:/LOL_Project/references/Ke2022_TeamFightModels_CoG.pdf`. |
| tot2021camera | Tot et al. 2021, What are you looking at? Team fight prediction through player camera, IEEE CoG, 1-8 | VERIFIED 2026-09-14 | https://api.crossref.org/works/10.1109/CoG52621.2021.9619038 | The key is fixed by the draft and `paper/refer.tex`. Re-read 2026-09-14: the Crossref record lists the same 14 authors in the same order, spelling "Oluseji" Olarewaju and Ben "Kirmann" as the printed first page does. The same people appear as Oluseyi Olarewaju and Ben Kirman on `kokkinakis2020dax`; each entry keeps its own record's spelling. Section IV.A of the paper takes team-fight labels for 1,457 patch-7.27 matches from the OpenDota API. The paper does not give the labels' start offset (see 2a). |
| opendota2018processteamfights | OpenDota, `processTeamfights.js`, team-fight processor of odota/core at commit e2e4432 (2018-10-21) | VERIFIED 2026-09-14 | https://github.com/odota/core/blob/e2e44328032e5ce81879f2a6ec3ee73e9120e736/processors/processTeamfights.js ; https://api.github.com/repos/odota/core/commits?path=processors/processTeamfights.js | **Added on 2026-09-14** so that sec_definition.tex:192 has a dated source for the 15 s offset. Source code has no DOI or proceedings record, so the publisher's own repository at a pinned commit is the source. At that commit: line 10 sets `teamfightCooldown = 15`; line 17 sets `start = e.time - teamfightCooldown` for the hero death that opens a fight; line 45 closes a fight at the first interval event 15 or more after the last hero death; line 57 keeps fights with `deaths >= 3`. There is no spatial condition. `createParsedDataBlob.js` at the same commit (line 102) stores the result as `parsedData.teamfights`. The file's history has 24 commits. The next change (b6e810a, 2022-09-20) only reformats it. The three rules are identical at 5ed1f21 (2016-03-10), a1d40ea (2023-12-01) and 3b6a7cd (2023-12-02, renamed to `.mjs`), and in odota/parser `processors/processTeamfights.mjs` on master (read 2026-09-14). Commit 570cfc0 removed the file from odota/core on 2024-01-07. The unit of `e.time` is not stated in the file; OpenDota's parser at 44ec63a (2020-06-10, `Parse.java` lines 353, 469) rounds the combat-log timestamp and `m_fGameTime`, so seconds is inferred from the field names. The key year is the commit year. The repository cannot show what OpenDota's servers ran. |
| costa2021 | Costa, Mantovani, Monteiro Souza, Xexeo 2021, Feature analysis to League of Legends victory prediction on the picks and bans phase, IEEE CoG, 1-5 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1109/CoG52621.2021.9619019 | Third author changed on 2026-09-11 to the record's name split (family name "Monteiro Souza"). |
| hitar2023 | Hitar-Garcia, Moran-Fernandez, Bolon-Canedo 2023, Machine learning methods for predicting League of Legends game outcome, IEEE ToG 15(2):171-181 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1109/TG.2022.3153086 | |
| hodge2021win | Hodge et al. 2021, Win prediction in multiplayer esports: live professional match prediction, IEEE ToG 13(4):368-379 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1109/TG.2019.2948469 | Same work as `hodge2021` in `paper/refer.tex`. |
| bisberg2022 | Bisberg, Ferrara 2022, GCN-WP: semi-supervised graph convolutional networks for win prediction in esports, IEEE CoG, 449-456 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1109/CoG51982.2022.9893671 | |
| katona2019time | Katona et al. 2019, Time to die: death prediction in Dota 2 using deep learning, IEEE CoG, 1-8 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1109/CIG.2019.8847997 | The DOI keeps the "CIG" prefix under which the publisher registered it. |
| silva2018continuous | Silva, Pappa, Chaimowicz 2018, Continuous outcome prediction of League of Legends competitive matches using recurrent neural networks, SBGames 2018, 639-642 | VERIFIED 2026-09-14 | https://www.sbgames.org/sbgames2018/files/papers/ComputacaoShort/188226.pdf | **Re-read on 2026-09-14** (HTTP 200, 4-page PDF), after the fact-checker had received HTTP 429 earlier that day. Page 1 gives the title and the three authors. Every page header reads "SBC - Proceedings of SBGames 2018 - ISSN: 2179-2259" and "Computing Track - Short Papers". Every footer reads "XVII SBGames - Foz do Iguacu - PR - Brazil, October 29th - November 1st, 2018" with page numbers 639, 640, 641 and 642. The entry is unchanged apart from its marker date. |
| chitayat2023beyond | Pedrassoli Chitayat, Block, Walker, Drachen 2023, Beyond the meta: leveraging game design parameters for patch-agnostic esport analitics, AIIDE 19(1):116-125 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1609/aiide.v19i1.27507 ; https://ojs.aaai.org/index.php/AIIDE/article/view/27507 | The published title spells "Analitics" and is reproduced as published. |

### 3.3 Esports audience experience and narrative tools (CoG reviewer R1's list)

All six works on R1's list have entries: `schubert2016encounter` (3.2) and the five below.

| key | work | status | source URL | notes |
|---|---|---|---|---|
| block2018narrative | Block et al. 2018, Narrative bytes: data-driven content production in esports, ACM TVX, 29-41 | VERIFIED 2026-09-14 | https://api.crossref.org/works/10.1145/3210825.3210833 ; https://dblp.org/rec/conf/tvx/BlockHHSDUDC18 | R1 list. The Crossref record stores "Data-Driven Content Production in Esports" in its subtitle field. DBLP gives the same title, pages and eight authors in the same order. **Changed on 2026-09-14:** "Victoria J. Hodge" now follows DBLP (Crossref has no initial), which also matches `hodge2021win`. |
| kokkinakis2020dax | Kokkinakis et al. 2020, DAX: data-driven audience experiences in esports, ACM IMX, 94-105 | VERIFIED 2026-09-14 | https://api.crossref.org/works/10.1145/3391614.3393659 ; https://dblp.org/rec/conf/tvx/KokkinakisDNOPR20 | R1 list. **Evidence replaced on 2026-09-14.** The 17 author names now rest on DBLP and Crossref, which agree in order and spelling, not on the White Rose accepted manuscript. The Crossref record duplicates "Pedrassoli" in one given name; DBLP does not. Place from the Crossref event field ("Cornella, Barcelona Spain"). The entry's fields are unchanged; the manuscript agrees with them. |
| pedrassoli2024passive | Pedrassoli Chitayat et al. 2024, From passive viewer to active fan: towards the design and large-scale evaluation of interactive audience experiences in esports and beyond, ACM IMX, 94-107 | VERIFIED 2026-09-14 | https://dl.acm.org/doi/10.1145/3639701.3656318 ; https://api.crossref.org/works/10.1145/3639701.3656318 ; https://dblp.org/rec/conf/tvx/ChitayatCBDWDMY24 | R1 list. **Changed on 2026-09-14.** The ACM Digital Library page (read once before the site began serving a Cloudflare check), Crossref and DBLP all list the eight authors as Alistair Coates, James Alfred Walker and Mark Mcconachie. The entry now follows them. The White Rose accepted manuscript, which the 2026-09-11 entry followed, prints "Alastair Coates", "James Walker" and "Mark McConachie"; it is not an allowed source. "Mcconachie" is probably a case-folding artefact of the metadata, but no allowed source shows "McConachie". Place from the Crossref event field ("Stockholm Sweden"). |
| charleer2018dashboards | Charleer et al. 2018, Real-time dashboards to support eSports spectating, CHI PLAY, 59-71 | VERIFIED 2026-09-14 | https://api.crossref.org/works/10.1145/3242671.3242680 ; https://dl.acm.org/doi/10.1145/3242671.3242680 ; https://dblp.org/rec/conf/chiplay/CharleerGGCLV18 | R1 list. The ACM Digital Library page and DBLP (read 2026-09-14) give the same six authors in the same order and pages 59-71. |
| pedrassoli2024wincondition | Pedrassoli Chitayat, Block, Walker, Drachen 2024, How could they win? An exploration of win condition for esports narratives in Dota 2, PACM HCI 8(CHI PLAY), 1-22 | VERIFIED 2026-09-14 | https://api.crossref.org/works/10.1145/3677079 ; https://dblp.org/rec/journals/pacmhci/ChitayatBWD24 | R1 list; the Dota 2 win-condition paper of `paper/refer.tex` lines 45-50. **Changed on 2026-09-14:** the author forms now rest on Crossref and DBLP, which agree (James A. / James Alfred Walker). The `note = {Art. no. 314}` field was removed because its only source was the White Rose accepted manuscript; see 2c. Published 2024-10-14 (Crossref), hence `month = oct`. |

### 3.4 Data source

| key | work | status | source URL | notes |
|---|---|---|---|---|
| riotapi2024timeline | Riot Games, MATCH-V5 API: get a match timeline by match id | VERIFIED 2026-09-14 | https://developer.riotgames.com/api-details/match-v5 | The key is fixed by `definition_section_draft.tex` (from `paper/refer.tex`). The page lists `GET /lol/match/v5/matches/{matchId}/timeline` returning `TimelineDto`. The documentation is undated: "2024" in the key is not a version date, and the entry gives an access date. **Corrected 2026-09-14 (item F1):** the portal URL the entry used to give (`https://developer.riotgames.com/apis#match-v5/GET_getTimeline`) is rendered by script; a plain HTTP client gets about 1,500 characters of navigation text and no field.  What was read is the static page in the source column (plain HTTP GET, HTTP 200, 167,797 bytes): `TimelineDto`, `InfoTimeLineDto` (`frameInterval` typed long with no value, `participants`, `frames`), `FramesTimeLineDto` (`events`, `participantFrames`, `timestamp`), `EventsTimeLineDto` (`timestamp`, `realTimestamp`, `type`) and `ParticipantFrameDto` (position, gold, level, CS, champion and damage statistics).  It gives no frame interval value and no per-event-type fields.  The entry's url and note now name that page and the access date 2026-09-14. |
| riotapi2024matchv5 | Riot Games, MATCH-V5 API: get a match by match id | VERIFIED 2026-09-14 | https://developer.riotgames.com/api-details/match-v5 | Lists `GET /lol/match/v5/matches/{matchId}`, which returns `MatchDto`. Undated; carries an access date. **Corrected 2026-09-14 (item F1):** the portal URL the entry used to give (`https://developer.riotgames.com/apis#match-v5`) is rendered by script and yields no field to a plain HTTP client.  What was read is the static page in the source column: `MetadataDto.participants` ("A list of participant PUUIDs"), `InfoDto.participants` (`List[ParticipantDto]`), `ParticipantDto.teamId`, `teamPosition` and `individualPosition`, `TeamDto.teamId` and `win`.  The page states no participant count (its `ParticipantFramesDto` key reads "1-9"), so "ten participants, five per team" rests on this record structure plus the corpus count (sec_background.tex check C1).  The entry's url and note now name the page read and the access date 2026-09-14. |

### 3.5 Minute-resolution telemetry and interpolation

| key | work | status | source URL | notes |
|---|---|---|---|---|
| che2018grud | Che et al. 2018, Recurrent neural networks for multivariate time series with missing values, Sci. Rep. 8, art. 6085 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1038/s41598-018-24271-9 | |
| horne2007brownian | Horne et al. 2007, Analyzing animal movements using Brownian bridges, Ecology 88(9):2354-2363 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1890/06-0957.1 | |
| shukla2019interpolation | Shukla, Marlin 2019, Interpolation-prediction networks for irregularly sampled time series, ICLR | VERIFIED 2026-09-11 | https://arxiv.org/abs/1909.07782 ; https://iclr.cc/Downloads/2019 | The arXiv comment names ICLR; the ICLR 2019 list links it as /virtual/2019/poster/1058. ICLR papers have no page numbers. |

### 3.6 Learners and the tree-vs-deep-learning question on tabular data

| key | work | status | source URL | notes |
|---|---|---|---|---|
| ke2017lightgbm | Ke et al. 2017, LightGBM: a highly efficient gradient boosting decision tree, NeurIPS 30 | VERIFIED 2026-09-11 | https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Bibtex.bib | The proceedings record has an empty page field, so the "3146-3154" in `paper/refer.tex` is not used. |
| gorishniy2021revisiting | Gorishniy, Rubachev, Khrulkov, Babenko 2021, Revisiting deep learning models for tabular data, NeurIPS 34, 18932-18943 | VERIFIED 2026-09-14 | https://proceedings.neurips.cc/paper_files/paper/2021/file/9d86d83f925f2149e9edb0ac3b49229c-Bibtex.bib | Complete. Re-read 2026-09-14: the proceedings BibTeX gives volume 34, pages 18932-18943, Curran Associates; the url resolves. |
| arik2021tabnet | Arik, Pfister 2021, TabNet: attentive interpretable tabular learning, AAAI 35(8):6679-6687 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1609/aaai.v35i8.16826 | |
| somepalli2021saint | Somepalli, Goldblum, Schwarzschild, Bruss, Goldstein 2021, SAINT: improved neural networks for tabular data via row attention and contrastive pre-training, arXiv:2106.01342 | VERIFIED 2026-09-14 (as a preprint) | https://arxiv.org/abs/2106.01342 ; https://dblp.org/rec/journals/corr/abs-2106-01342 | Stays a preprint. Re-read 2026-09-14: arXiv shows one version (v1, 2021-06-02), no journal reference, no comment. The only DBLP record is the CoRR preprint, and a Crossref title search returns no published version. A version was presented as a poster at the NeurIPS 2022 Table Representation Learning workshop (https://neurips.cc/virtual/2022/58531, which lists the authors in a different order). The organisers' call for papers describes that workshop as non-archival. **Added on 2026-09-14:** the arXiv-issued DOI 10.48550/arXiv.2106.01342. |
| grinsztajn2022tree | Grinsztajn, Oyallon, Varoquaux 2022, Why do tree-based models still outperform deep learning on typical tabular data?, NeurIPS 35 Datasets and Benchmarks, 507-520 | VERIFIED 2026-09-14 | https://proceedings.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Bibtex-Datasets_and_Benchmarks.bib ; https://api.crossref.org/works/10.52202/068431-0037 | Re-read 2026-09-14: volume 35, pages 507-520, DOI 10.52202/068431-0037. |
| shwartzziv2022tabular | Shwartz-Ziv, Armon 2022, Tabular data: deep learning is not all you need, Information Fusion 81:84-90 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1016/j.inffus.2021.11.011 | |
| lundberg2020local | Lundberg et al. 2020, From local explanations to global understanding with explainable AI for trees, Nat. Mach. Intell. 2(1):56-67 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1038/s42256-019-0138-9 | TreeSHAP reference. |

### 3.7 Calibration, resampling, stacking, partition agreement

| key | work | status | source URL | notes |
|---|---|---|---|---|
| platt2000probabilities | Platt, Probabilities for SV machines, in Smola et al. (eds.), Advances in Large-Margin Classifiers, MIT Press, 61-74 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.7551/mitpress/1113.003.0008 ; https://api.crossref.org/works/10.7551/mitpress/1113.001.0001 | This is the task list's "Platt 1999". The publisher's records give the year 2000 and the chapter title above, so the key uses 2000. The work is commonly cited under a longer title that begins "Probabilistic outputs for support vector machines". direct.mit.edu served a bot page. |
| zadrozny2002transforming | Zadrozny, Elkan 2002, Transforming classifier scores into accurate multiclass probability estimates, KDD, 694-699 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1145/775047.775151 | |
| guo2017calibration | Guo, Pleiss, Sun, Weinberger 2017, On calibration of modern neural networks, ICML, PMLR 70:1321-1330 | VERIFIED 2026-09-11 | https://proceedings.mlr.press/v70/guo17a.html | |
| naeini2015obtaining | Naeini, Cooper, Hauskrecht 2015, Obtaining well calibrated probabilities using Bayesian binning, AAAI 29(1):2901-2907 | VERIFIED 2026-09-11 | https://ojs.aaai.org/index.php/AAAI/article/view/9602 ; https://ojs.aaai.org/index.php/AAAI/article/download/9602/9461 | Page numbers and "Gregory F. Cooper" come from the publisher PDF (first page 2901, last page 2907); on 2026-09-11 they were re-sourced there from Europe PMC, which is not an allowed source. Neither the AAAI page nor Crossref shows pages. |
| efron1993bootstrap | Efron, Tibshirani 1993, An Introduction to the Bootstrap, Chapman & Hall | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1007/978-1-4899-4541-9 ; https://www.cambridge.org/core/journals/psychometrika/article/abs/b-efron-and-r-j-tibshirani-1993-an-introduction-to-the-bootstrap-new-york-chapman-hall-xvi-436-pp-isbn-04120423t2-5000/D7CC806A82BF7AA651C2BBF745D58017 | The DOI is Springer's registration for the book. Its record (re-read 2026-09-14) names the publisher "Springer US, Boston, MA" and lists the 1993 print ISBN 9780412042317 alongside an electronic ISBN. The imprint "New York: Chapman & Hall" comes from the title of a Psychometrika book review on Cambridge Core, a publisher page, but for the review, not for the book. This is the weakest imprint evidence in the file; the author may prefer to cite the Springer registration as is. |
| wolpert1992stacked | Wolpert 1992, Stacked generalization, Neural Networks 5(2):241-259 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1016/S0893-6080(05)80023-1 | |
| hubert1985comparing | Hubert, Arabie 1985, Comparing partitions, J. Classification 2(1):193-218 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1007/BF01908075 | |

### 3.8 Neural components and training procedures used by the deep learners

| key | work | status | source URL | notes |
|---|---|---|---|---|
| cho2014learning | Cho et al. 2014, Learning phrase representations using RNN encoder-decoder for statistical machine translation, EMNLP, 1724-1734 | VERIFIED 2026-09-11 | https://aclanthology.org/D14-1179/ ; https://api.crossref.org/works/10.3115/v1/D14-1179 | Month (October) and place (Doha, Qatar) come from the Anthology BibTeX. |
| vaswani2017attention | Vaswani et al. 2017, Attention is all you need, NeurIPS 30 | VERIFIED 2026-09-11 | https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Bibtex.bib | The record has no pages, so the "5998-6008" in `paper/refer.tex` is not used. |
| hamilton2017inductive | Hamilton, Ying, Leskovec 2017, Inductive representation learning on large graphs, NeurIPS 30 | VERIFIED 2026-09-11 | https://proceedings.neurips.cc/paper_files/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Bibtex.bib | Carried over from `paper/refer.tex` and not cited in the ToG sections. The record has no pages, so the "1024-1034" in `paper/refer.tex` is not used. |
| loshchilov2019decoupled | Loshchilov, Hutter 2019, Decoupled weight decay regularization (AdamW), ICLR | VERIFIED 2026-09-11 | https://arxiv.org/abs/1711.05101 ; https://iclr.cc/Downloads/2019 | The arXiv comment says it was published at ICLR 2019; the ICLR 2019 list links it as /virtual/2019/poster/935. |
| loshchilov2017sgdr | Loshchilov, Hutter 2017, SGDR: stochastic gradient descent with warm restarts, ICLR | VERIFIED 2026-09-11 | https://arxiv.org/abs/1608.03983 ; https://iclr.cc/archive/www/doku.php%3Fid=iclr2017:conference_posters.html | Evidence is the ICLR 2017 archived poster list (changed on 2026-09-11 from a search-listing sighting). That list shortens the title to "SGDR: Stochastic Gradient Descent with Restarts"; the entry keeps the arXiv title. |
| yun2019cutmix | Yun et al. 2019, CutMix: regularization strategy to train strong classifiers with localizable features, ICCV, 6022-6031 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1109/ICCV.2019.00612 | IEEE pagination; the CVF open-access copy is paginated differently. |
| zhang2018mixup | Zhang, Cisse, Dauphin, Lopez-Paz 2018, mixup: beyond empirical risk minimization, ICLR | VERIFIED 2026-09-11 | https://iclr.cc/virtual/2018/poster/177 ; https://arxiv.org/abs/1710.09412 | The arXiv v2 comment identifies that version as the ICLR camera-ready. |
| chen2020simple | Chen, Kornblith, Norouzi, Hinton 2020, A simple framework for contrastive learning of visual representations (SimCLR), ICML, PMLR 119:1597-1607 | VERIFIED 2026-09-11 | https://proceedings.mlr.press/v119/chen20j.html | |
| martins2016sparsemax | Martins, Astudillo 2016, From softmax to sparsemax, ICML, PMLR 48:1614-1623 | VERIFIED 2026-09-11 | https://proceedings.mlr.press/v48/martins16.html | |

### 3.9 Win probability and state value (engagement-state-value worktree)

| key | work | status | source URL | notes |
|---|---|---|---|---|
| maymin2021smart | Maymin 2021, Smart kills and worthless deaths: eSports analytics for League of Legends, JQAS 17(1):11-27 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1515/jqas-2019-0096 | This is the "Maymin (2021)" win-probability paper of the worktree docs (`CLAUDE_NEXT_STAGE_HANDOFF.md` l.61, `CLAUDE_V3_BUCKET_RETRAIN_REPORT.md` l.11, which follows its section 3.2 sampling). Published online 2020-09-21, issue 2021. No citation key for it existed anywhere in the repositories (re-searched 2026-09-14), so `maymin2021smart` was chosen. |
| jalovaara2024win | Jalovaara 2024, Win probability estimation for strategic decision-making in esports, master's thesis, Aalto University, School of Science | VERIFIED 2026-09-14 | https://aaltodoc.aalto.fi/server/api/core/items/c1f92b5b-66f2-4742-b746-7d1a1b4d093e ; https://sal.aalto.fi/publications/pdf-files/theses/mas/tjal24a_public.pdf | This is the "Jalovaara (2024)" bucket-sampling work of the worktree docs (`CLAUDE_V3_BUCKET_RETRAIN_REPORT.md` l.15; `CLAUDE_NEXT_STAGE_HANDOFF.md` l.75). The thesis's section 5.1 (p. 27) divides each match into 5-minute intervals and samples one game state per interval. Source type: Aaltodoc is Aalto University's own publication archive, and the university issues the thesis, so the record is the publisher's page, unlike the Lund record once used for Schubert et al. Re-read 2026-09-14: issued 2024-08-26, 38 pages, School of Science, Master's Programme in Mathematics and Operations Research, URN:NBN:fi:aalto-202411217333. Not peer reviewed. No key existed, so `jalovaara2024win` was chosen. |
| kim2020confidence | Kim, Lee, Chung 2020, A confidence-calibrated MOBA game winner predictor, IEEE CoG, 622-625 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.1109/CoG47356.2020.9231878 | Named in the worktree docs. |
| ferreira2025quantifying | Ferreira, Faganeli Pucer 2025, Quantifying player death impact in League of Legends, SCORES'25, 51-54 | VERIFIED 2026-09-11 | https://api.crossref.org/works/10.51939/scores25.12 | Student research symposium; weaker evidence than a journal or main-track paper. The record also lists the university as an author, which is omitted. |

### 3.10 Game rules, patch notes and data terms (web sources; bib section 10)

Wiki pages: `https://wiki.leagueoflegends.com/en-us/<page>`, no byline, cited at the latest revision on
2026-09-14 with its permanent link `Special:PermanentLink/<revision>` in the `note` field; month and year
are those of that revision.  All 31 revision ids and timestamps below were re-read on 2026-09-14 in one
MediaWiki API query and match the entries.  "Not patch-dated" means the page gives a current value and no
patch-history entry or version article dates it to 25.14-25.16 (game versions 15.14-15.16).  Riot pages
were downloaded with a plain HTTP client (HTTP 200); title, byline and publication time come from each
page's own metadata (`og:title`, JSON-LD `datePublished` and `author`).  Patch-number mapping: see N2 in
2c and the header of bib section 10.

| key | page | status | source URL | revision or date; notes |
|---|---|---|---|---|
| lolwiki2026terminology | League of Legends Wiki, Terminology (glossary entries "Ace" and "Teamfight") | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Terminology | Revision 4057849, 2026-08-28.  Teamfight entry in section T (wikitext, action=parse section 21); the wiki search API finds the definition's wording on this page only.  **Extended 2026-09-14 (item F1):** the key is also cited for "ace" (sec_background.tex, Table I).  Section A (action=parse section 2, read through WebFetch) has the Ace entry with two senses: all champions of a team defeated at the same time, and killing the last living champion of the opposing team.  The bib note now names both entries; no other term is cited to this key.  Not patch-dated. |
| lolwiki2025summonersrift | Summoner's Rift | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Summoner's_Rift | Revision 3968911, 2025-11-21.  Sides, lanes, jungle, river, six drake elements.  **Comment corrected in the second web-source pass:** the page does not state the team size (N1).  **Corrected again in the third pass:** the two-teams-of-champions phrase is in the Environment section (section 1), not the lead (section 0). |
| lolwiki2026champion | Champion | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Champion | Revision 4026288, 2026-06-09.  Player-controlled characters with their own abilities and attributes. |
| lolwiki2026sight | Sight | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Sight | Revision 4035710, 2026-06-25.  1,350 u for champions and turrets; 900 u for wards.  **Comment corrected:** the patch history has one entry (V13.22, fog-of-war attacker reveal), none for these radii.  Not patch-dated. |
| lolwiki2026movementspeed | Movement speed | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Movement_speed | Revision 4047640, 2026-07-29.  325-355 on the rendered page; the numbers are template-drawn (N3).  Not patch-dated. |
| lolwiki2026gold | Gold | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Gold | Revision 4039269, 2026-07-03.  Uses and sources of gold. |
| lolwiki2026experience | Experience (champion) | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Experience_(champion) | Revision 4053165, 2026-08-22.  1,600 u for champion deaths, 1,500 u for minions, level cap 18.  The only range entries in the patch history concern minions (V25.S1.1, V4.11).  R is "no change listed", not dated. |
| lolwiki2026minion | Minion | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Minion | Revision 4051665, 2026-08-13; reached from the redirect Minion (League of Legends).  First wave at 0:30 is post-26.1 (Riot 26.1 notes: 1:05 to 30 s). |
| lolwiki2026farming | Farming | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Farming | Revision 3989487, 2026-02-01; reached from the redirect Creep score.  1 per minion, 4 per full camp. |
| lolwiki2026kill | Kill | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Kill | Revision 4053216, 2026-08-23.  Definition; 15 s credit on Summoner's Rift (20 s Howling Abyss); bounty to the killer, assist gold split.  No patch entry for the 15 s window. |
| lolwiki2026assist | Assist | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Assist | Revision 4016680, 2026-05-12.  15 s on Summoner's Rift, 20 s on Howling Abyss; no patch-history section.  B is "no change listed", not dated. |
| lolwiki2026death | Death | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Death | Revision 4051174, 2026-08-11.  Respawn at the fountain; base wait 10 s (level 1) to 52.5 s (level 18); time factor after 15 minutes. |
| lolwiki2026bounties | Champion gold bounties | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Champion_gold_bounties | Revision 4040646, 2026-07-06.  Base 300 g at levels 1-6, 310-420 g at 7-18; V25.14 farming accrual change.  **Comment extended:** V25.09 set the current base, which `riot2025patch2509` confirms (N4). |
| lolwiki2026turret | Turret | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Turret | Revision 4050003, 2026-08-04.  Counts, current rewards, five plates.  **Comment corrected:** the patch history does have V25.x and V26.x entries; V25.S1.1 removes and V26.01 restores the 300 g first-turret bonus (N5).  **Comment extended in the third pass:** pre-26.1 local and plate gold from `riot2026patch261`; V13.20 and V13.23 date the plate, inner and inhibitor values; no turret-gold line in V14.1-V26.03 apart from the first-turret bonus; outer and nexus global gold not dated but no change listed (N8). |
| lolwiki2026inhibitor | Inhibitor | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Inhibitor | Revision 4047299, 2026-07-29.  Super minions; 5-minute respawn, dated to V4.20 by the page. |
| lolwiki2026nexus | Nexus | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Nexus | Revision 3991087, 2026-02-11.  Win condition; invulnerability rule; spawns minions. |
| lolwiki2026dragonpit | Dragon pit | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Dragon_pit | Revision 4000799, 2026-03-18; reached from the redirects Elemental drake and Dragon.  5:00 first drake, 5-minute respawn, Dragon Soul on the fourth; V26.01 kill gold 75 from 25 (recipient not stated).  **Comment extended in the third pass:** kill gold 25 from the V9.23 November 25th hotfix to V26.01, no gold entry in 2025 (N9). |
| lolwiki2026elderdragon | Elder Dragon | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Elder_Dragon | Revision 4002526, 2026-03-26.  Spawns after a team's fourth drake; 6-minute respawn; 150 s buff to living teammates. |
| lolwiki2026dragonslayer | Dragon Slayer (section Aspect of the Dragon) | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Dragon_Slayer | Revision 4056301, 2026-08-25; reached from the redirect Aspect of the Dragon.  Burn; execute below 20 percent of maximum health; the only 2025 entry is V25.S1.3 (soul values). |
| lolwiki2026voidgrub | Voidgrub camp | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Voidgrub_camp | Revision 4015021, 2026-05-02; reached from the redirect Voidgrub.  Three grubs once at 8:00 (V25.09); V25.14 and V25.16 entries are fixes. |
| lolwiki2026riftherald | Rift Herald | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Rift_Herald | Revision 4026493, 2026-06-09.  Once per game at 15:00 (V25.09); Eye of the Herald. |
| lolwiki2026atakhan | Atakhan | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Atakhan | Revision 3982741, 2026-01-09.  Added V25.S1.1, removed V26.01; 20:00 spawn on the side with more combat by 14:00; V25.09 left only the Thornbound form. |
| lolwiki2026baron | Baron Nashor | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Baron_Nashor | Revision 4053266, 2026-08-23.  6-minute respawn; Hand of Baron 180 s; initial spawn 25:00 from V25.S1.1 to V26.01; V25.15 and V25.16 health changes only. |
| lolwiki2025ward | Ward | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Ward | Revision 3906127, 2025-06-05.  Wards remove fog of war; control ward visible, reveals and disables wards, limit one placed; the only 2025 entry is V25.S1.1 (two ward skins removed). |
| lolwiki2026zac | Zac | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Zac | Revision 4057435, 2026-08-27.  E range 1,200-1,800 by rank, no range change V25.10-V25.20.  **Comment extended:** range and history are transcluded (N3); the pages do not call it the longest non-ultimate engage. |
| lolwiki2026combatstatus | Combat status | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/Combat_status | Revision 4058480, 2026-09-03; reached from the redirect Combat.  Most out-of-combat effects start 5 s after combat.  No patch history; not patch-dated. |
| lolwiki2026v25s11 | V25.S1.1 (version article) | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/V25.S1.1 | Revision 4017989, 2026-05-14.  First sentence numbers the patch 15.1; release January 9, 2025; Atakhan introduced. |
| lolwiki2026v2509 | V25.09 (version article) | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/V25.09 | Revision 4057851, 2026-08-29.  First sentence numbers the patch 15.9; release April 30, 2025; Voidgrub 8:00, Rift Herald 15:00. |
| lolwiki2026v2514 | V25.14 (version article) | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/V25.14 | Revision 4016266, 2026-05-10.  Release July 16, 2025; Yunara; July 21 hotfix; bounty accrual entry.  Gives no 15.14 number. |
| lolwiki2026v2515 | V25.15 (version article) | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/V25.15 | Revision 4021808, 2026-05-21.  Release July 30, 2025; Prev V25.14, Next V25.16; no hotfix. |
| lolwiki2026v2516 | V25.16 (version article) | VERIFIED 2026-09-14 | https://wiki.leagueoflegends.com/en-us/V25.16 | Revision 4037479, 2026-06-29.  Release August 13, 2025; Prev V25.15, Next V25.17; August 13 and 14 hotfixes (section headings). |
| riot2025patch25s11 | Riot Games, Patch 25.S1.1 Notes | VERIFIED 2026-09-14 | https://www.leagueoflegends.com/en-us/news/game-updates/patch-25-s1-1-notes/ | Riot Riru, published 2025-01-07T19:00Z.  Client shows the seasonal patch number; Atakhan; **comment extended:** Feats of Strength replace the First Blood and First Turret gold bonuses (N5). |
| riot2025patch2509 | Riot Games, Patch 25.09 Notes | VERIFIED 2026-09-14 | https://www.leagueoflegends.com/en-us/news/game-updates/patch-25-09-notes/ | **New in the second web-source pass.**  Riot Riru and Riot Sakaar, published 2025-04-29T18:00Z.  Base kill gold rises by 10 g per level from 7 to 18 (maximum 420); Void Grubs 6:00 to 8:00, no respawn; Rift Herald at 15:00.  The notes for 25.10-25.16 list no base-kill-gold change (string search; N4). |
| riot2025patch2514 | Riot Games, Patch 25.14 Notes | VERIFIED 2026-09-14 | https://www.leagueoflegends.com/en-us/news/game-updates/patch-25-14-notes/ | Riot Riru, published 2025-07-15T18:00Z.  Yunara; bounty from farming 1 per 17.5 g to 1 per 20 g; suppression thresholds 25 percent wider; no "assist", "credit", experience range or radius.  Third pass: "turret" only in a bug fix; no plate, drake, local or global gold. |
| riot2025patch2515 | Riot Games, Patch 25.15 Notes | VERIFIED 2026-09-14 | https://www.leagueoflegends.com/en-us/news/game-updates/patch-25-15-notes/ | Riot Sakaar, published 2025-07-29T18:00Z.  Baron in-combat regeneration removed; no "assist", "credit", "bounty", experience range or radius.  Third pass: "turret" only in a Heimerdinger bug fix; no plate, drake, local or global gold. |
| riot2025patch2516 | Riot Games, Patch 25.16 Notes | VERIFIED 2026-09-14 | https://www.leagueoflegends.com/en-us/news/game-updates/patch-25-16-notes/ | Riot Riru, published 2025-08-12T18:00Z.  Baron base health 11,500 to 11,800; no "assist", "credit", "bounty", experience range or radius.  Third pass: "turret" only in a bug fix, "plate" only in Experimental Hexplate, "drake" only in a Mel bug fix; no local or global gold. |
| riot2026patch261 | Riot Games, Patch 26.1 Notes | VERIFIED 2026-09-14 | https://www.leagueoflegends.com/en-us/news/game-updates/patch-26-1-notes/ | Riot Riru and Riot Sakaar, published 2026-01-07T19:00Z.  Plates permanent (had fallen off at 14 minutes); outer-turret local gold 250 to 0, plate gold 125 to 120; Atakhan, Blood Roses, Feats of Strength removed; Baron 25:00 to 20:00; minions 1:05 to 30 s; **comment extended:** First Blood (100 g) and first turret (300 g) bonuses restored.  **Extended again in the third pass:** before-values of outer (local 250, plate 125, maximum local 875, minimum local 250), inner (local 425/675) and inhibitor (local 375) turrets; elemental drake kill gold 25 to 75; no global turret gold (N8, N9). |
| riot2013apiterms | Riot Games, API Terms and Conditions | VERIFIED 2026-09-14 | https://developer.riotgames.com/terms | Page states last updated December 9, 2013 (heading misspelt "CONDTIONS").  Materials definition; unauthorised use; deletion on termination; GDPR deletion requests.  No "research", no "dataset". |
| riot2025devpolicies | Riot Games, General Policies (Riot Developer Portal) | VERIFIED 2026-09-14 | https://developer.riotgames.com/policies/general | Page states last updated May 29, 2025.  Named only in a sec_availability.tex comment. |
| riot2025ddragon1514 | Riot Games, Data Dragon champion data, game versions 15.13.1 and 15.14.1 | VERIFIED 2026-09-14 | https://ddragon.leagueoflegends.com/cdn/15.14.1/data/en_US/champion.json | 171 champions with Yunara (key 804) at 15.14.1; 170 without at 15.13.1; version list (499 versions) runs 15.13.1, 15.14.1, 15.15.1, 15.16.1 without gaps.  Undated files: no year field. |

## 4. Changes

### 4.0 Cross-file fixes of 2026-09-14 (item F1-cross-file-fixes)

1. `lolwiki2026terminology`: the note now reads "Glossary entries ``Ace'' and ``Teamfight''", because
   sec_background.tex cites the key for "ace" as well as "teamfight".  The Ace entry was read in section A of
   revision 4057849 (action=parse, section 2); the Teamfight entry was re-read (section 21).  3.10 row updated.
2. `riotapi2024matchv5`, `riotapi2024timeline`: url changed from the script-rendered portal
   (`https://developer.riotgames.com/apis#match-v5...`) to the static page that was actually read,
   `https://developer.riotgames.com/api-details/match-v5`; the note says the portal view is rendered by script;
   access date Sep. 11 -> Sep. 14, 2026; the evidence comments list the DTO fields read and state that the page
   gives no participant count and no frame interval value.  3.4 rows updated.  This answers the mismatch
   flagged in the sec_availability.tex source comment (item W5).
3. N1 marked as applied in sec_background.tex.

### 4.1 Web-source passes of 2026-09-14 (item W1b-web-sources)

1. First web-source pass (cut off): appended bib section 10 with 39 entries (31 wiki pages, Riot patch notes
   25.S1.1, 25.14, 25.15, 25.16 and 26.1, the API terms, the general policies, Data Dragon) and the section
   header with the conventions and the patch-number mapping.
2. Second web-source pass, triage: every one of the 39 entries was re-checked.  Revision ids and timestamps of all 31
   wiki pages match (one api.php query); load-bearing values were re-read on the cited revisions
   (action=parse with oldid), exact wording through the wiki search API (list=search, insource phrase);
   Riot pages and Data Dragon files re-downloaded and string-searched; the corpus Yunara count re-run
   (889 of 4,000).  The printed fields of all 39 entries were correct and are unchanged.
3. Second web-source pass, comments corrected: `lolwiki2025summonersrift` (no team size in the prose),
   `lolwiki2026sight` (one V13.22 patch entry), `lolwiki2026turret` (V25.x and V26.x entries exist; first-turret
   bonus removed in V25.S1.1 and restored in V26.01).
4. Second web-source pass, comments extended: `lolwiki2026movementspeed` and `lolwiki2026zac` (template-drawn values,
   with template and subpage revisions for Zac), `lolwiki2026bounties` (V25.09 base change; the /History
   subpage named in sec_background.tex is a separate page), `riot2025patch25s11` (Feats of Strength
   replace the first-turret gold bonus), `riot2026patch261` (bonuses restored).
5. Second web-source pass, entry added: `riot2025patch2509`, which dates the base champion bounty and gives the
   Voidgrub and Rift Herald spawn times.
6. Second web-source pass, this file: section 1 re-run; section 2 table refreshed (39 keys cited now that
   `sec_intro.tex` and `sec_related.tex` exist); 2b emptied; new 2c (placeholder-to-key mapping with notes
   N1-N7); old 2c renamed 2d; new 3.10; this subsection; section 5 extended.
7. Third web-source pass (answering a fact-check), `lolwiki2025summonersrift` comment corrected.  The
   second pass had said that the lead and the Environment section both speak of two teams of champions.
   The lead (section 0 of revision 3968911) does not; the phrase is in the Environment section.  The
   conclusion is unchanged: the page gives no team size.  The fact-check was right.
8. Third pass, `lolwiki2026turret` and `riot2026patch261` comments extended with the turret rewards before
   26.1.  From the 26.1 notes: outer turret 250 g local, 125 g per plate, at most 875 g local; inner
   turret 425/675 g local; inhibitor turret 375 g local.  From the Turret history: V13.20 and V13.23 date
   the plate, inner and inhibitor values, and nothing changes them before V26.01.  The fact-check was
   right that the row for sec_limitations.tex:165-166 omitted these values and that the comment at
   sec_limitations.tex:168-169 is wrong.  Two findings go beyond the fact-check.  (a) The Turret history
   also dates inner and inhibitor global gold (25 g, V13.23), and no entry from V13.20 to V26.03 changes
   the outer and nexus turrets' 50 g, so every local, plate and global value is at least "no change
   listed".  (b) The 25.16 notes contain a third turret hit, a Voidgrub/Viego bug fix, besides the 25.14
   and 25.15 fixes that the fact-check named; it is not a gold change either.
9. Third pass, `lolwiki2026dragonpit` comment extended.  Kill gold is 25 g from the V9.23 hotfix to
   V26.01, and the page lists no 2025 gold entry.  The fact-check was right.  One addition: the V26.01
   global-gold line (250 to 150) belongs to the Elder Dragon, not to the elemental drakes, so the
   fact-check's "only kill-gold entries" statement holds.  Riot's 26.1 notes give the same before-value,
   so `riot2026patch261` was added to that row.
10. Third pass, `riot2025patch2514`, `riot2025patch2515` and `riot2025patch2516` comments extended with the
    turret, plate and drake string search.
11. Third pass, this file: section 1 re-run (744 added, 0 deleted lines against HEAD; 105 entries;
    `warning$ -- 0`) with a note on the IEEEtran rendering; section 2 table refreshed (line numbers in
    `sec_intro.tex` and `sec_related.tex` moved; 41 keys, two of them web sources); 2c rows for
    sec_limitations.tex:150, 153, 165-166, 167, 168-169 and 171-173, sec_definition.tex:377 and
    rule_constants_evidence.md:56; the scan note on the two web-source citations now in `sec_intro.tex`;
    N1 and N7 extended; new N8 and N9; 3.10 rows updated.

### 4.2 Pass of 2026-09-14 (answering the fact-check)

1. `schubert2016encounter`: evidence moved from the Lund University record (not an allowed source) to the
   conference's own paper page and conference-hosted PDF, both read through Internet Archive captures.
   Booktitle now names the Research Papers Competition, and a url to the archived PDF was added.  The
   author order is unchanged: it is the order printed on the paper.
2. `kokkinakis2020dax`, `pedrassoli2024wincondition`, `block2018narrative`, `charleer2018dashboards`:
   evidence moved to DBLP and Crossref, plus the ACM Digital Library page for Charleer et al.  The White
   Rose accepted manuscripts are no longer used as sources.  Field changes: `block2018narrative` gains
   "Victoria J." (DBLP); `pedrassoli2024wincondition` loses `note = {Art. no. 314}`, whose only source was
   a manuscript.
3. `pedrassoli2024passive`: author forms changed to those of the ACM page, Crossref and DBLP: "Alistair
   Coates", "James Alfred Walker", "Mark Mcconachie" (previously "Alastair", "James", "McConachie", from the
   manuscript).
4. `opendota2018processteamfights`: new entry, the dated OpenDota source for the 15 s offset at
   sec_definition.tex:192 (see 2a).
5. `mehrzadi2012session`: the bib comment and this table now give the same source for the place (the
   Crossref event field, with the local PDF as agreement).
6. `silva2018continuous`: publisher PDF re-read (HTTP 200); marker date 2026-09-14, fields unchanged.
7. `somepalli2021saint`: re-checked arXiv, DBLP and Crossref, which show no peer-reviewed version; added
   the arXiv DOI.
8. `tot2021camera`, `ke2022`, `gorishniy2021revisiting`, `grinsztajn2022tree`, `jalovaara2024win`:
   re-read, fields unchanged; evidence comments extended.
9. Bib header: the conventions for DBLP, archived publisher pages and source code were added.
10. Section 2a: the sec_definition.tex:192 row now requires both citations and records the limits of the
    OpenDota source; the OpenDota item was removed from 2b.

### 4.3 Pass of 2026-09-11

1. `silva2018continuous`: re-verified against the publisher PDF; added pages 639-642, track, place and
   month.
2. `naeini2015obtaining`: page numbers and author initial re-sourced from Europe PMC to the AAAI publisher
   PDF.
3. `ester1996dbscan`: removed `address` and `publisher`, whose only source was an OSTI record.
4. `costa2021`: third author written in the publisher record's name split.
5. `loshchilov2017sgdr`: evidence changed to the ICLR 2017 archived poster list.
6. `jalovaara2024win`: evidence extended to the Aalto repository record; URN added as a note.
7. Evidence comments made more exact for `block2018narrative`, `mcqueen2014screens` and
   `somepalli2021saint`.

## 5. Source-type notes and sources that could not be read

Entries whose evidence includes something other than a publisher page, DBLP, arXiv or a DOI resolver:

- `tot2021camera`: the author spellings were first read on the conference-hosted PDF at ieee-cog.org.  The
  Crossref record now confirms the same list, so the entry no longer depends on that PDF.
- `jalovaara2024win`: university archive, which is the thesis publisher (see its row).
- `opendota2018processteamfights`: the publisher's GitHub repository, the only primary source for source
  code.
- `schubert2016encounter`: Internet Archive captures of the publisher's page and PDF.
- `efron1993bootstrap`: the imprint comes from a review title on a publisher page (see its row).
- Local PDFs under `D:/LOL_Project/references` are cited only as agreement, never as the sole source.
- Section 3.10: every web-source entry rests on the publisher's own page (League of Legends Wiki, Riot
  Games), which task rule 4 allows for web sources.  The League of Legends Wiki is community-edited; each
  entry pins a revision.  `riot2025ddragon1514` is Riot's static data service, a data file rather than a
  page.  The patch-number mapping in N2 also rests on a count over the local corpus cache, which is
  evidence about the corpus, not a bibliographic source.

Limits of the web-source reading (2026-09-14):

- wiki.leagueoflegends.com served a bot-check page ("Please wait", HTTP 403) to a plain HTTP client, and
  the same page to the Browser pane.  No attempt was made to get past it: the pane was closed without
  waiting.  The wiki was read through the host's page fetcher (WebFetch) only.
- That fetcher summarises long pages with a small model, and two readings of the same long wikitext
  disagreed on details.  Only values confirmed by a targeted reading of a short section, by the wiki search
  API (`list=search` with an `insource` phrase, which returns the matching page and a snippet), or by two
  agreeing readings were written into comments.  Values that the manuscript takes from templates (N3) were
  checked on the rendered page or on the template itself.
- Riot's pages (leagueoflegends.com, developer.riotgames.com, ddragon.leagueoflegends.com) were downloaded
  with a plain HTTP client and searched as text, so their statements do not depend on a summary.
- Third pass: the history values in N8 and N9 were taken only from blocks that the fetcher was asked to
  copy verbatim.  These were contiguous ranges of `api.php?action=parse&oldid=<rev>&prop=wikitext&section=<n>`:
  the V26.01 block and the V25.19-to-V9.23 range of Dragon pit; the V25.20-to-V14.19 range, the
  V14.18-to-V13.23 range, the range from the section start to V26.01 (the V26.03 block) and the V13.23,
  V14.19, V13.20 and V26.01 blocks of Turret.  Summary readings
  of the same sections disagreed on details: one attached the Elder Dragon global-gold line to a
  non-existent "V14.3 (Elder Dragon)" heading, and another missed it.  Those readings were not used.
  The absence of gold lines between V9.23 and V26.01 on Dragon pit rests on three readings that agree.
  One is a verbatim heading list with a "NONE" answer for gold, bounty and reward; the other two are
  summaries.  Riot's 25.14-25.16 notes were checked for turret, plate and drake entries only by string
  search, as for R and B; a person's reading of the three notes is still not on record.

Sources that could not be read on 2026-09-14 (fact-check pass):

- dblp.org web search: served an Anubis proof-of-work bot check to scripts.  The dblp SPARQL endpoint was
  used instead.
- dl.acm.org: one IMX 2024 page and the CHI PLAY 2018 page were read; the next requests got a Cloudflare
  check (HTTP 403).  No attempt was made to get past it, so the PACM HCI article number could not be
  confirmed.
- sloansportsconference.com: the 2016 paper page and PDF no longer exist on the live site (redirect to the
  home page; the PDF returns 404).  The Internet Archive captures were used.
- GitHub code search API: requires authentication, so commit history and raw files at pinned commits were
  read instead.

Sources that could not be read on 2026-09-11: dblp.org (bot check), dl.acm.org (HTTP 403), direct.mit.edu
and the aaltodoc item pages (bot or 403 pages; the Aaltodoc REST API worked), Project Euclid (bot page to
plain HTTP; readable with WebFetch), sbgames.org (intermittent), and table-representation-learning.github.io
2022 edition (404).
