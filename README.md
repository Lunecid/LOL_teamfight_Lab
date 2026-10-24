# LOL Teamfight Lab

English | [한국어](README.ko.md)

Research code for understanding teamfights in League of Legends through public match and timeline data. The work explores engagement outcomes, changes in estimated win probability, and how engagements unfold.

The repository brings together three related projects. Each has its own research question and experiment setup, with code for preparing data, building features, training models, and evaluating predictions. Start with the project closest to what you want to study.

## The projects

| Project | What it studies |
|---|---|
| [EPT — `c_ept/`](c_ept/) | Engagement-process modeling, including input features, model training and evaluation, and computational cost measurement. This is the current development track. |
| [CoG — `COG/`](COG/) | Kill-conditioned engagement outcome prediction under minute-resolution public telemetry. This contains the code from the IEEE Conference on Games research track. |
| [Master / ToG — `master_ver/`](master_ver/) | Changes in estimated win probability around engagements, and what can be predicted from information available before an engagement. |

The projects share a topic, but their engagement definitions and evaluation settings differ. Their models and results should be read within their respective studies.

Supporting code lives in [`research_pipeline/`](research_pipeline/) and [`common_training_r1/`](common_training_r1/). The small library in [`tools/project_runtime/`](tools/project_runtime/) handles source-path checks and process launching for CoG and Master.

[Browse the code by module](MODULES.md). EPT stage runners and reports have their own folders; historical import entry files are kept in `_compat/`.

## Getting started

This is a source-code release; match datasets and trained weights are not included. Running experiments requires your own local inputs and the appropriate research environment. You can install the command-line interfaces and explore their help before preparing those inputs.

From the repository root, use separate, prepared Python environments with `pip`, `setuptools`, and `wheel` already installed. These commands install the interfaces without downloading scientific dependencies.

In the EPT environment:

```powershell
python -m pip install --no-index --no-deps --no-build-isolation -e .
lol-teamfight-ept --help
```

In a separate launcher environment for CoG and Master:

```powershell
python -m pip install --no-index --no-deps --no-build-isolation -e tools/project_runtime -e COG -e master_ver
lol-teamfight-cog --help
lol-teamfight-master --help
```

For EPT research runs, follow the [recorded runtime requirements](environment/s4a-runtime.json) and [configuration example](configs/s4a.example.json). EPT also needs separately supplied legacy resources: set `C_EPT_LEGACY_ROOT` to their absolute directory and check them against [`legacy_pins.json`](c_ept/legacy_pins.json). The public `COG/` folder alone does not supply this resource set.

A lightweight structure check is available without starting an experiment:

```powershell
python -B tools/check_project_boundaries.py
```

## Further reading

- [Working guidelines](docs/WORKING_GUIDELINES_KO.md) — implementation, independent review, execution gates and publication rules; Korean.

- [Project organization and installation](docs/MODULARIZATION_KO.md) — package layout and module boundaries; Korean.
- [CoG methods](COG/README.md) — historical research overview in English; see also the [CoG module guide](COG/MODULARIZATION.md) in Korean.
- [Master module guide](master_ver/MODULARIZATION.md) — source organization and the extracted metric functions; Korean.
- [Shared launcher library](tools/project_runtime/README.md) — installation and how the launchers work.
- [Public release and reproducibility notes](docs/PUBLICATION_20261024.md) — what is included, external prerequisites, and why some comparison tests need the original development history; Korean.
