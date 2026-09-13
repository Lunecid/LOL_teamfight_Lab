# References audit: docs/references/definition_refs.bib

Item W1-references.  First written 2026-09-11; revised 2026-09-14 after a fact-check.  This file is the
per-entry table that the header of `docs/references/definition_refs.bib` points to.

History.  An earlier agent rewrote the bib file, marked every entry VERIFIED, and was cut off before
writing this table.  The 2026-09-11 pass re-checked every entry against the source named in its row and
changed the six entries whose evidence was weak or came from a disallowed source.  The 2026-09-14 pass
answered the fact-check: it replaced the non-publisher sources behind five entries, added a dated
OpenDota source for the 15 s offset, and re-ran the parse, key and `bibtex` checks (section 4 lists the
changes).

Allowed sources (task rule 4): publisher pages and publisher PDFs, DBLP, arXiv, DOI resolvers.  Terms used
below:

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

## 1. Triage of the file (re-run 2026-09-14 after the last edit)

| Check | Result |
|---|---|
| Entries parsed | 65, each with a type and a key; 65 line-initial at-signs, so no stray entry starts |
| Brace balance (entry and field level) | balanced in all 65 |
| Duplicate keys (exact and case-insensitive) | none |
| Duplicate fields inside an entry | none |
| Required fields per type (article, inproceedings, incollection, book, misc, mastersthesis) | none missing |
| Entries without a VERIFIED/UNVERIFIED marker | none |
| Half-written entries or stray text between entries | none |
| Stray at-sign in a comment line, or unescaped percent sign inside an entry | none |
| `bibtex` with `IEEEtran.bst` 1.14 (MiKTeX) over all entries (`\citation{*}`) | 65 `\bibitem`s, `warning$ -- 0` |

The checker script and the test `.aux` are in the session scratchpad, not in the repository.

## 2. Citation-key sanity

Every key cited in `docs/tog_manuscript/*.tex` and in `docs/references/definition_section_draft.tex` exists
in the bib file.  The check stripped comments, then collected the keys of every `\cite...{...}` and
`\nocite{...}`.  Other agents were editing the section files during this pass: the line numbers in
`sec_limitations.tex` moved between two runs on 2026-09-14.  Re-run the check after later section edits.

| Cited key | Cited in (file line) | In bib |
|---|---|---|
| arik2021tabnet | sec_learners 105, 199; sec_limitations 365 | yes |
| berman2014mapping | sec_definition 283; draft 152 | yes |
| block2018narrative | sec_limitations 499 | yes |
| campello2013hdbscan | sec_definition 91; draft 58 | yes |
| charleer2018dashboards | sec_limitations 500 | yes |
| chitayat2023beyond | sec_definition 249; sec_limitations 472; draft 134 | yes |
| gorishniy2021revisiting | sec_learners 106, 174, 185; sec_limitations 359 | yes |
| halfaker2015session | sec_definition 49; draft 41 | yes |
| hodge2021win | sec_limitations 486 | yes |
| jacobs2021measurement | sec_definition 35, 427; draft 20, 163 | yes |
| katona2019time | sec_learners 241 | yes |
| ke2017lightgbm | sec_learners 150 | yes |
| ke2022 | draft 15, 30, 106 | yes |
| kokkinakis2020dax | sec_limitations 499 | yes |
| mehrzadi2012session | sec_definition 49, 108; draft 41, 65 | yes |
| pedrassoli2024passive | sec_limitations 501 | yes |
| pedrassoli2024wincondition | sec_limitations 501 | yes |
| riotapi2024timeline | draft 28 | yes |
| schubert2016encounter | sec_background 409; sec_definition 29, 388; sec_limitations 79, 498; draft 15, 36 | yes |
| silva2018continuous | sec_learners 240 | yes |
| silverman1981multimodality | sec_definition 51; draft 50 | yes |
| somepalli2021saint | sec_learners 105, 214; sec_limitations 359 | yes |
| tot2021camera | draft 16, 112 | yes |
| zaliapin2013clusters | sec_definition 50, 165; draft 43, 92, 146 | yes |
| zaliapin2022perspectives | sec_definition 165, 264; draft 92 | yes |

**25 keys cited; missing keys: none.**  The other 40 entries are not cited yet.  Most of them are waiting
for the `\pending{cite}` placeholders in 2a.

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

### 2b. `\pending{cite}` placeholders with no entry (outside this item's list)

No entries were added for these.  Each is a web page that has to be fetched and dated, and each game fact
must also be patch-dated, before it can be cited.

- League of Legends Wiki pages: Terminology (teamfight entry), Turret, Experience (champion), Kill, Assist,
  Death, Champion gold bounties, Sight, Ward, Zac, Combat, Summoner's Rift, Champion, Movement speed, Gold,
  Minion, Creep score, Inhibitor, Nexus, Dragon, Elder Dragon, Aspect of the Dragon, Voidgrub, Rift Herald,
  Atakhan, Baron Nashor, and the patch pages V25.S1.1, V25.09, V25.14, V25.15, V25.16
  (sec_background.tex:259-265, :279-329; sec_definition.tex:27, :187, :203; sec_limitations.tex:153, :350).
- Riot Games patch notes 25.14, 25.15, 25.16 (sec_definition.tex:198; sec_limitations.tex:358) and 26.1
  (sec_background.tex:262, :329).
- Riot Games API Terms and Conditions (sec_availability.tex:73, :85).

### 2c. Differences from the superseded CoG bibliography (`paper/refer.tex`, gitignored)

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

Status is VERIFIED for all 65 entries.  "Crossref" in the source column abbreviates
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
| riotapi2024timeline | Riot Games, MATCH-V5 API: get a match timeline by match id | VERIFIED 2026-09-11 | https://developer.riotgames.com/api-details/match-v5 | The key is fixed by `definition_section_draft.tex` (from `paper/refer.tex`). The page lists `GET /lol/match/v5/matches/{matchId}/timeline` returning `TimelineDto`. The documentation is undated: "2024" in the key is not a version date, and the entry gives an access date. |
| riotapi2024matchv5 | Riot Games, MATCH-V5 API: get a match by match id | VERIFIED 2026-09-11 | https://developer.riotgames.com/api-details/match-v5 | Lists `GET /lol/match/v5/matches/{matchId}`, which returns `MatchDto`. Undated; carries an access date. |

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

## 4. Changes

### 4.1 Pass of 2026-09-14 (answering the fact-check)

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

### 4.2 Pass of 2026-09-11

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

Sources that could not be read on 2026-09-14:

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
