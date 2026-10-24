# CoG code modules

`analysis/fight_boundaries/` owns boundary geometry, temporal/spatial estimation,
validity-radius lookup, specification serialization, and boundary orchestration.
This is a physical implementation move; scientific definitions and algorithms are unchanged.

| Previous module under `analysis/` | Implementation under `analysis/fight_boundaries/` |
| --- | --- |
| `ability_range.py` | `ability_range.py` |
| `boundary_spec.py` | `boundary_spec.py` |
| `kill_pairs.py` | `kill_pairs.py` |
| `map_regions.py` | `map_regions.py` |
| `spatial_boundary.py` | `spatial_boundary.py` |
| `temporal_boundary.py` | `temporal_boundary.py` |
| `fight_boundary_pipeline.py` | `fight_boundary_pipeline.py` |

The seven previous files are namespace-relative compatibility aliases: Python
resolves each `from .fight_boundaries` target using that shim's `__package__`.
Both source-backed `analysis.<module>` and repository-qualified
`COG.analysis.<module>` imports are supported. Within either root, old and new
imports share one module object, mutable globals and monkeypatches. The two roots
remain separate module objects; importing one never replaces the other's aliases,
globals or class identities.

Implementation modules keep their canonical `__file__` and import specification.
Locally defined functions/classes retain historical pickle roots in their own
namespace: `analysis.<module>` or `COG.analysis.<module>`. Existing payloads are
neither read nor rewritten. Registration derives the legacy name and parent package
from the actual canonical name, registering only the seven aliases within that root.
There are no production `sys.path` changes or import-finder hooks.

The initial migration's hardcoded registration namespace caused independent review
P2 (qualified-import failure and mixed-root alias overwrite). This namespace-derived
registration corrects that defect; independent acceptance requires a separate delta
review.

Scripts, scientific tests, manuscripts, numerical records, inputs and outputs retain
their existing paths. Historical source hashes change with this relocation; this does
not authorize resuming a hash-bound historical run. Network-backed ability-range
lookup, reports, training and experiments are not executed by layout checks.

From `COG/`, with CPU library thread counts set to 1, CUDA hidden, and bytecode disabled:

```powershell
python -m unittest discover -s organization_tests -p test_fight_boundary_layout.py -v
```

The four organization tests compare baseline algorithm AST fingerprints (normalizing
six sibling imports), exact alias structure, and eight fresh-process namespace/order
scenarios: each root alone plus both mixed-root orders, each legacy-first and
canonical-first. They check parent bindings, global isolation, simultaneous
monkeypatches, historical pickle lookups and synthetic dataclass round trips, with
network blocked. Tests alone choose repository-root or CoG-root cwd and add the CoG
search root through subprocess PYTHONPATH for mixed-root probes. Production imports
must leave sys.path and meta_path unchanged.

The tracked fingerprints require neither Git ancestry nor work/ reports. The only
additional implementation AST nodes per module remain an import and call that register
compatibility metadata; this fix changes only that helper, tests and handoff/module
notes. Independent review is separate from these implementer checks.
