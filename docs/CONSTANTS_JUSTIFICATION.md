# Every constant, anchored in game mechanics

> **SUPERSEDED (2026-09-11).** This page documents the v2 detector used for the CoG 2026
> submission. It is not the current engagement definition and must not be cited as one.
> Current definition: `docs/tog_manuscript/sec_definition.tex` (manuscript) and
> `docs/ENGAGEMENT_DEFINITION_V3.md` (compact spec); evidence in `docs/DEFINITION_EVIDENCE.md`
> sections 14-17 and 22.
>
> Withdrawn from this page:
>
> 1. **18 s / 4,000 u as the current kill-cluster gap and cluster diameter.** Both are now
>    derived from 208,141 matches (`config/fight_boundary/spec_pooled.json`): G = 13.7246 s,
>    the antimode of the consecutive-kill-gap density, and D = 4,263.87 u, the distance at
>    which the share of within-G kill pairs that have a champion in common falls to 50 %.
>    The v3.3 corpus was detected with the rounded values 13,700 ms and 4,264.0 u
>    (`D:/LOL_Project/fusion_2615/corpus_shards_v33/manifest.json`).
> 2. **The Data Dragon ability-range coverage argument for R = 1,800 u**: the "89.3 % of
>    ability casts within 1,800 u" below (Data Dragon 16.16.1, a later patch than the
>    15.14-15.16 corpus) and the per-patch 89.1 % in `DEFINITION_EVIDENCE.md` sections 10 and
>    14. The Data Dragon `range` field holds placeholder values for dash and charge abilities
>    (`DEFINITION_EVIDENCE.md` section 15), so a coverage share cannot anchor a radius.
>    R and B are now game-rule anchors, and neither is estimated from the data. R = 1,600 u is
>    the radius within which a dying champion's experience is shared with enemy champions
>    (League of Legends Wiki, "Experience (champion)"; the minion radius, 1,500 u, is a separate
>    value). B = 15 s is the kill/assist credit window on Summoner's Rift (League of Legends
>    Wiki, "Kill"; 20 s on Howling Abyss). Both pages were checked on 2026-09-11. Neither records
>    a change to the 1,600 u radius or the 15 s window, including for the corpus patches
>    15.14-15.16. Earlier citation and the rule brackets: `DEFINITION_EVIDENCE.md` section 15.
> 3. **"AUC spread 0.008"** as the answer to threshold sensitivity. It comes from a 553-match
>    sweep on the v2 detector (`D:/LOL_Project/fusion_2615/features/thresholds/*.json`): six
>    one-factor settings (gap 12/18/24/30 s at 4,000 u; diameter 3,000/5,000 u at 18 s), not
>    a gap x diameter grid, with 2,242-2,783 labelled rows per setting and AUC 0.593-0.601.
>    It says nothing about the v3.3 headline. The G x D sensitivity against that headline has
>    not been produced yet: `\pending{gd_sensitivity_v33}{G x D sensitivity measured against the v3.3 headline}`.
>
> Other rows below that no longer match the released v3.3 configuration: the backtrack is
> B = 15 s, not 10 s (`TF2_ENGAGE_PRE_KILL_MS` = 15000 in the manifest), and the label window
> runs to max(last kill, cutoff + 35 s) (`FIGHT_HORIZON_SEC` = 35; `ENGAGEMENT_DEFINITION_V3.md`
> section 4), not 30 s. The 3,000 u interaction radius is still in use (not overridden in the
> manifest), but its only justification on this page is the same coverage calculation
> (91.4 %), so it has no anchor of its own at present. The named-ability lists below are kept
> for provenance only.

Skill-range distribution measured from Data Dragon patch 16.16.1
(173 champions, 673 spells; `spells[].range` max-rank, `stats.attackrange`):

- Basic attacks: melee <=200 u (51% of champions), ranged median 550, max 650.
- Abilities: median 700, p75 1,000; **89.3% cast within 1,800 u**; per-champion
  longest-range ability median 1,150 (hook/engage class ~1,100-1,300).
- The remaining ~10% are semi-global and global ultimates (2,500-25,000 u).
- Map: 15,000 x 15,000 units; base movement speed ~345-400 u/s.

| constant | value | anchor |
|---|---|---|
| validity radius `TF2_VALIDITY_RADIUS` | 1,800 u | beyond every basic attack and 89.3% of ability casts = the boundary of direct-engagement range; from outside it a champion reaches the fight only via a semi-global ultimate or ~5 s of travel |
| interaction radius `TF2_INTERACTION_RADIUS` | 3,000 u | covers 91.4% of ability casts incl. the mid semi-global tier; permissive by design - an event already proves involvement, the radius only filters cross-map coincidences |
| cluster split `CLUSTER_MAX_DIAMETER` | 4,000 u | two kill groups >=4,000 u apart (~1/4 map diagonal) cannot interact except via true globals -> separate engagements |
| kill-cluster gap `TF2_KILL_CLUSTER_GAP_MS` | 18 s | > measured median teamfight duration (12.4 s) so one fight's kills merge; < early respawn+return time so a reset starts a new engagement |
| backtrack `TF2_ENGAGE_PRE_KILL_MS` | 10 s | approach-and-burst time preceding a first kill (P1 backtrack bounds 5-15 s) |
| min per team `TF2_MIN_PER_TEAM` | 2 | the smallest mutual engagement; 1v1 duels are not team engagements |
| gold dead zone `LABEL_GOLD_DEADZONE` | 300 g | exact game constant: base solo-kill bounty - swings below one kill's value are materially even |
| label horizon `FIGHT_HORIZON_SEC` | 30 s | immediate-conversion window (objective/tower cash-in after a won fight); also CoG observation length |
| grid step `TF2_GRID_STEP_MS` | 5 s | position interpolation resolution between 60 s snapshots and ms kill positions |
| vision radius `VISION_RADIUS` | 1,200 u | ward sight range (~900-1,100) with margin |

Sensitivity already measured: gap 12-30 s x diameter 3,000-5,000 -> AUC spread
0.008; dead zone {150,300,600} reported as a column. The spatial trio is
additionally ordered by role: presence (1,800, strict, causal gate) <
interaction (3,000, permissive, evidence-based) < split (4,000, separator).

## Named anchors per band (patch 16.16.1)

- Longest basic attacks: Ashe 600, Senna 600, Annie 625, Caitlyn 650.
- hook/engage class (inside validity): n=93 spells; e.g. Anivia W 'Crystallize' 1000; Aphelios E 'Weapon Queue System' 1000; Braum Q 'Winter's Bite' 1000; Draven W 'Blood Rush' 1000; Gangplank E 'Powder Keg' 1000; Gragas R 'Explosive Cask' 1000.
- longest direct-engagement casts (validity boundary 1,800): n=16 spells; e.g. Lucian R 'The Culling' 1400; Yasuo R 'Last Breath' 1400; Aphelios Q 'Weapon Abilites' 1450; Jinx W 'Zap!' 1450; Aurelion Sol W 'Astral Flight' 1500; Nidalee Q 'Javelin Toss / Takedown' 1500.
- semi-global tier (between validity and interaction 3,000): n=14 spells; e.g. Renata Glasc R 'Hostile Takeover' 2000; Xayah E 'Bladecaller' 2000; Quinn W 'Heightened Senses' 2100; Tahm Kench E 'Thick Skin' 2400; Akshan R 'Comeuppance' 2500; Ornn R 'Call of the Forge God' 2500.
- between interaction and cluster split 4,000: n=6 spells; e.g. Lux R 'Final Spark' 3340; Bard R 'Tempered Fate' 3400; Caitlyn R 'Ace in the Hole' 3500; Rengar R 'Thrill of the Hunt' 3500; Nocturne R 'Paranoia' 4000; Warwick W 'Blood Hunt' 4000.
- global class (beyond every radius): n=52 spells; e.g. Smolder R 'MMOOOMMMM!' 4200; Kled R 'Chaaaaaaaarge!!!' 4500; Kalista W 'Sentinel' 5000; Xerath R 'Rite of the Arcane' 5000; Ziggs R 'Mega Inferno Bomb' 5000; Akshan W 'Going Rogue' 5500.
