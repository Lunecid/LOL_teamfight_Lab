"""Public P1 raw-input projection API.

Exports are lazy so the command entry point can set numerical thread limits
before NumPy is imported.
"""

__all__ = [
    "FULL_INVENTORY_COMPLETE",
    "HistoryPrefix",
    "QUERY_BLOCK_SPECS",
    "REFERENCE_SPECS",
    "REQUIRED_MATCH_FIELDS",
    "STATIC_BLOCK_SPECS",
    "SemanticHistory",
    "VERSION",
    "field_manifest",
    "project_match",
]


def __getattr__(name):
    if name in {"HistoryPrefix", "SemanticHistory", "project_match"}:
        from . import projection
        return getattr(projection, name)
    if name in {
        "FULL_INVENTORY_COMPLETE", "QUERY_BLOCK_SPECS", "REFERENCE_SPECS",
        "REQUIRED_MATCH_FIELDS", "STATIC_BLOCK_SPECS", "VERSION", "field_manifest",
    }:
        from . import schema
        return getattr(schema, name)
    raise AttributeError(name)
