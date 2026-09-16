# Every constant, anchored in game mechanics

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
