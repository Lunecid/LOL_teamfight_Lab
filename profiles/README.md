# Run profiles

A run profile is one JSON file that says what a run of the S4b run kernel is: a training run, an evaluation run or a bit-identity
measurement. The kernel (`c_ept/run`, decision D11) starts a run from a profile file and from nothing else. The loader is
`c_ept.run.profile.load(path)`; it reads the file strictly (`c_ept.run.io.read_json_with_sha`: no duplicate keys, no NaN or infinity, no
symbolic link, at most 16 MiB, the top level is an object), checks every field and returns a frozen `Profile` with the `kind` and the
`sha256` of the file bytes. The run records that hash, so a run names the exact file it was started from.

Rules:

- `kind` is required and is one of `train`, `evaluate`, `measure`.
- A field that the kind does not define is refused (a typo is a refusal, not a silent default). A wrong type or a value outside the
  allowed set or range is refused with a message that names the field. The message never contains a value, an unknown field name, a path
  or a hash.
- An omitted field takes its default. The defaults of a `train` profile are the EPT-T002 run (S model, 25 % of FIT, seed 20261022,
  30 epochs, patience 2, lr 0.001, weight decay 0.01, 10 warm-up steps) and the old behaviour of every Track M switch, so the bare
  profile `{"kind": "train"}` carries the T002 settings. A behaviour changes only when its field is written. The batch size (32) and
  the device are runner constants, not profile fields.
- Whole numbers are JSON integers (`12`, not `12.0`). A flag is `true` or `false` (not `1` or `0`). A real number may be written as
  an integer (`1` is read as `1.0`). `null` is accepted only where the table says so.
- The file must use LF line ends: a file with a carriage return (a CRLF checkout) is refused, so that one content has one hash.
- The hash locks the bytes of the file, not their meaning: the same settings written with other spaces have another hash. Compare
  `Profile.settings` to ask whether two profiles mean the same. The file is read once: the settings, the `sha256` and the line-end test
  all come from the same bytes, so the `sha256` is always the hash of the content that the settings were read from, also when the file is
  changed (or changed and changed back) while or after it is read. The loader then returns the pair of the one read; it does not detect
  such a change, and a later `sha256` comparison with the file on disk shows it.
- A `Profile` cannot be pickled or deep-copied. A child process gets the path and the `sha256` and calls `load` again.

## Fields

### train

| Field | Allowed | Default |
|---|---|---|
| `size` | `"S"`, `"M"` | `"S"` |
| `fraction` | real, above 0 and at most 1 | `0.25` |
| `seed` | integer, 0 up to 2^63 - 1 | `20261022` |
| `epochs` | integer, at least 1 and at most 1000 | `30` |
| `patience` | integer, at least 1 | `2` |
| `lr` | real, at least 0 | `0.001` |
| `weight_decay` | real, at least 0 | `0.01` |
| `warmup` | integer, at least 0 | `10` |
| `kill_law` | `"poisson"`, `"shifted"` | `"poisson"` |
| `k_condition` | `"raw"`, `"binned"` | `"raw"` |
| `decay_groups` | `"all"`, `"split"` | `"all"` |
| `terrain_smooth_weight` | real, at least 0 | `0.0` |
| `swap_mode` | `"average"`, `"random_view"`, `"none"` | `"average"` |
| `eval_average` | `true`, `false` | `true` |
| `on_wall_cap` | `"fail"`, `"best_stop"` | `"fail"` |
| `zone_mark` | `true`, `false` | `true` |
| `field_channels` | integer, at least 0 (`0` is A6) | `8` |
| `column_drop` | `"none"`, `"store"`, `"store_and_subset"` | `"none"` |
| `stats_population` | `"whole_fit"`, `"selected_fit"` | `"whole_fit"` |
| `selection_seed` | integer, 0 up to 2^63 - 1, or `null` | `null` |

Which matches a `fraction` below 1 keeps is decided by a seed. `selection_seed: null` (the default) selects with the training `seed`.
This is the T002 rule (`proxy_matches(s1, share, seed)`): another training seed selects other matches. An integer fixes the selection
for every training seed. This is the T009/matched rule (`select_membership(..., selection_seed=...)`): the S4b profiles with training
seeds 2 and 3 write `"selection_seed": 20261022` so that all seeds train on the same 25 %. `Profile.membership_seed()` returns the
seed to use, `selection_seed` if it is not null, else `seed`.

`Profile.to_train_kwargs()` returns what `TrainConfig` takes, for `TrainConfig(encoder, **kwargs)`: the keys `kill_law`, `k_condition`,
`decay_groups`, `column_drop`, `stats_population`, `terrain_smooth_weight`, `swap_mode`, `eval_average`, `on_wall_cap`, `zone_mark`,
`lr`, `weight_decay`, `warmup_steps` (the profile field `warmup`) and `patience`. This is not a partition of the fields: data
preparation also reads `kill_law`, `k_condition`, `swap_mode`, `column_drop`, `stats_population`, `seed` and `selection_seed`, and K6
maps the remaining fields (`size`, `field_channels`, `fraction`, `epochs`) to the real constructors. `TrainConfig` must carry the same
`column_drop` and `stats_population` as the statistics that the data preparation made, or `train._check_inputs` refuses the run. Every Track M
field of `TrainConfig` is a key here (X1 check in `c_ept/tests/test_run_profile.py`); `swap_mode`, `eval_average`, `on_wall_cap` and
`zone_mark` are keys before their fields exist (M7-M9), so `TrainConfig(encoder, **kwargs)` works only after those tasks have landed.

### evaluate

| Field | Allowed | Default |
|---|---|---|
| `role` | `"STOP"`, `"DEV"`, `"TEST"` | required |
| `models` | non-empty list of distinct, non-empty model names | required |
| `exposure` | `"observed"`, `"nominal"` | `"observed"` |
| `winner_truth` | `"gold"`, `"binned"` | `"gold"` |
| `chunk_views` | integer, at least 1, or `null` | `null` |
| `b0` | `true`, `false` | `false` |
| `strata` | `true`, `false` | `false` |
| `price_sensitivity` | `true`, `false` | `false` |
| `b1_swap_average` | `true`, `false` | `false` |
| `bootstrap_B` | integer, at least 1 and at most 100000 | `2000` |
| `checkpoints` | object: model name to non-empty path text | `{}` |
| `holdout_lock_sha256` | 64 lowercase hexadecimal digits, or `null` (required when `role` is `"TEST"`) | `null` |
| `composed` | `null` (off), or an object with the whole-number keys `sample_matches`, `sample_seed`, `draws`, `mc_batches`, `draw_seed` and optionally `progress_rows` | `null` |

`composed` switches on the composed V_start pass on a sample of STOP matches (S4b-prep K6e, decision D30: A7's V_start CRPS gain and D14's excluded-row share). The pass needs the role `"STOP"` and
`exposure` `"nominal"`. Its parameters have no default, because a later, separately approved cost probe sets them: `sample_matches` (1 to 1,000,000 STOP matches, drawn with `sample_seed`), `draws` (the
Monte Carlo draws of every row, 2 to 1,000,000), `mc_batches` (2 to 10,000; the draws are split into this many equal batches, at least two draws each, and the Monte Carlo standard error is the standard
deviation of the batch means over the square root of the batch count), `draw_seed` (the seed of the Composer; a row draws from the stream `[draw_seed, frame row]`) and `progress_rows` (the rows that one
progress line covers, 1 to 1,000,000, default 16). 합성 V_start 표본 계산은 비용 탐침이 표본 크기와 모의 횟수를 정할 때까지 값을 정하지 않는다. The sample size and the draw counts are never constants of the code.

`checkpoints` is checked for its shape only; the runner reads the files. `holdout_lock_sha256` is the sha256 of the S5 lock file (strict JSON naming the access-ledger
folder and the hash of its `created` entry): the one fixed anchor of the TEST gate (decision D28). It is part of the profile without its `role`, so the DEV rehearsal of a
TEST run carries the same value. The rest of the gate (DEV rehearsal, "TEST never opened" ledger) is the runner's job (K6), not the profile's.

### measure

A measure profile does not repeat the training settings. It names a train profile by relative path and SHA-256, and the runner measures
exactly that configuration, so the P6 measurement cannot drift from the fit that it vouches for.

| Field | Allowed | Default |
|---|---|---|
| `train_profile` | relative path with forward slashes, resolved against the folder of the measure file | required |
| `train_sha256` | 64 lowercase hexadecimal digits | required |
| `steps` | integer, at least 1 and at most 100000 | required |
| `repeat` | integer, at least 2 and at most 10 | `2` |

`load` of a measure profile loads the named file with the same `load` and exposes it as `Profile.train` (`None` for the other kinds).
The named file must be a `train` profile (a reference never nests) and its `sha256` must equal `train_sha256`, else the measure
profile is refused. The path is refused if it is empty, absolute, has a drive letter (any `:`) or a backslash, has an empty, `.` or
`..` segment, has a segment that ends in `.` or a space, or leaves the folder of the measure file through a link.

## Examples

The bare train profile is the EPT-T002 run:

```json
{
  "kind": "train"
}
```

A train profile with Track M switches and a fixed selection of the 25 %:

```json
{
  "kind": "train",
  "epochs": 12,
  "kill_law": "shifted",
  "k_condition": "binned",
  "decay_groups": "split",
  "swap_mode": "random_view",
  "on_wall_cap": "best_stop",
  "selection_seed": 20261022
}
```

An evaluation profile:

```json
{
  "kind": "evaluate",
  "role": "STOP",
  "models": ["b0", "b1", "ept"],
  "checkpoints": {"ept": "path/to/best.pt"}
}
```

A measurement profile for the first example (write the SHA-256 of `train.json` in place of the placeholder):

```json
{
  "kind": "measure",
  "train_profile": "train.json",
  "train_sha256": "<sha256 of train.json>",
  "steps": 3
}
```

## 한국어 안내

프로파일은 실행 한 번(학습, 평가, 측정)을 정하는 JSON 파일 하나이며, 실행 핵심은 이 파일만으로 실행을 시작한다. 파일 바이트의 SHA-256이
잠금이고 실행 기록에 남는다. 모르는 필드와 잘못된 값은 필드 이름을 대며 거부하고, 생략한 필드는 EPT-T002의 옛 값(또는 Track M 스위치의 옛 동작)으로
채워지므로 `{"kind": "train"}` 한 줄이 T002 설정이다. 정수 필드에는 `12`처럼 정수만 쓰고(`12.0` 불가), 논리값은 `true`/`false`만 쓴다. 줄 끝에 CR이 든
파일은 거부한다. 파일은 한 번만 읽으므로 설정, 해시, CR 검사가 같은 바이트에서 나온다. `fraction`이 1보다 작을 때 쓸 경기는 시드로 정한다: `selection_seed`가 `null`이면 학습 시드(T002 규칙), 정수이면 학습 시드와
무관한 고정 선택(T009/matched 규칙)이다. `measure` 프로파일은 설정을 복사하지 않고 train 프로파일 하나를 상대 경로와 해시로 가리킨다. 필드 이름은
`c_ept/tests/test_run_profile.py`의 계약 시험이 고정하므로, M 갈래에서 이름을 바꾸면 그 시험과 이 표를 함께 고쳐야 한다.
