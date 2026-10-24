"""Check declared packaging and direct shared-component imports, without execution.

Only the paths in SOURCE_MODULES are parsed. In particular, protected cohort and
holdout implementations are never opened, recursively scanned, or imported.
This is a bounded structural check, not a transitive dependency/security audit.
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
SHARED_FILES = (
    "c_ept/s4a_runtime.py",
    "c_ept/runtime/__init__.py",
    "c_ept/runtime/resources.py",
    "c_ept/ept/checkpoints.py",
    "c_ept/ept/training_step.py",
    "c_ept/ept/heads.py",
    "c_ept/ept/compose_heads.py",
    "c_ept/compose.py",
)
LAUNCHERS = (
    ("COG", "lol-teamfight-cog", "lol_teamfight_cog"),
    ("master_ver", "lol-teamfight-master", "lol_teamfight_master"),
)
PURE_MODULES = {
    "COG/app/fusion_spec.py": "app.fusion_spec",
    "master_ver/studies/v_production_20261004/metrics.py": "studies.v_production_20261004.metrics",
}
LAUNCHER_MODULES = {
    **{f"{folder}/src/{package}/{filename}.py": f"{package}.{filename}"
       for folder, _, package in LAUNCHERS
       for filename in ("__init__", "__main__", "cli")},
    **{f"tools/project_runtime/src/lol_project_runtime/{filename}.py": f"lol_project_runtime.{filename}"
       for filename in ("__init__", "cli", "launcher")},
}
SOURCE_MODULES = {
    **{path: path[:-3].replace("/", ".") for path in SHARED_FILES},
    **PURE_MODULES,
    **LAUNCHER_MODULES,
}
METADATA_PATHS = ("pyproject.toml", "COG/pyproject.toml", "master_ver/pyproject.toml",
                  "tools/project_runtime/pyproject.toml")


def import_targets(tree: ast.AST, module: str):
    """Yield static imports and literal import_module/__import__ calls at any depth."""
    package = module.split(".")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                prefix = package[:len(package) - node.level + 1]
                base = ".".join([*prefix, *([base] if base else [])])
            yield node.lineno, base
            for alias in node.names:
                yield node.lineno, f"{base}.{alias.name}"
        elif isinstance(node, ast.Call) and node.args:
            name = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else "")
            arg = node.args[0]
            if name in {"import_module", "__import__"} and isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                target = arg.value
                if target.startswith("."):
                    level = len(target) - len(target.lstrip("."))
                    target = ".".join([*package[:len(package) - level + 1], target.lstrip(".")])
                yield node.lineno, target


def check_imports(source: str, module: str, *, allowed_roots=None) -> list[str]:
    errors = []
    for line, target in import_targets(ast.parse(source), module):
        parts = target.split(".")
        if any(part.startswith("run_s") or part in {"train", "s4a_operator"} for part in parts):
            errors.append(f"{module}:{line}: shared component imports command module {target}")
        if allowed_roots is not None and parts[0] not in allowed_roots:
            errors.append(f"{module}:{line}: dependency outside allowlist: {target}")
    return errors


def check_packaging(metadata: dict, *, name: str, package: str, entry: str | None,
                    where: str, subpackages: bool = False, dependencies=None) -> list[str]:
    errors = []
    project = metadata.get("project", {})
    settings = metadata.get("tool", {}).get("setuptools", {})
    finder = settings.get("packages", {}).get("find", {})
    expected = [package, f"{package}.*"] if subpackages else [package]
    for valid, message in (
        (project.get("name") == name, "distribution name"),
        (project.get("scripts", {}) == ({name: entry} if entry else {}), "unique operator entrypoint"),
        (settings.get("py-modules") == [], "no generic top-level modules"),
        (finder.get("where") == [where], "package source directory"),
        (finder.get("include") == expected, "explicit package allowlist"),
        (finder.get("namespaces") is False, "namespace discovery disabled"),
    ):
        if not valid:
            errors.append(f"{name}: invalid {message}")
    if dependencies is not None and project.get("dependencies") != dependencies:
        errors.append(f"{name}: invalid runtime dependencies")
    if name == "lol-project-runtime" and project.get("version") != "0.1.0":
        errors.append(f"{name}: version differs from launcher dependency pin")
    return errors


def check_project(root: Path = ROOT) -> list[str]:
    errors = []
    specs = [(root / "pyproject.toml", "lol-teamfight-ept", "c_ept",
              "c_ept.s4a_operator:main", ".", True, [])]
    specs.extend((root / folder / "pyproject.toml", name, package,
                  f"{package}.cli:main", "src", False, ["lol-project-runtime==0.1.0"])
                 for folder, name, package in LAUNCHERS)
    specs.append((root / "tools/project_runtime/pyproject.toml", "lol-project-runtime",
                  "lol_project_runtime", None, "src", False, []))
    for path, name, package, entry, where, subpackages, dependencies in specs:
        try:
            metadata = tomllib.loads(path.read_text(encoding="utf-8-sig"))
            errors.extend(check_packaging(metadata, name=name, package=package,
                                          entry=entry, where=where, subpackages=subpackages,
                                          dependencies=dependencies))
        except (OSError, ValueError) as exc:
            errors.append(f"{path.relative_to(root)}: {exc}")
    for relative, module in SOURCE_MODULES.items():
        try:
            source = (root / relative).read_text(encoding="utf-8-sig")
            allowed = None
            if relative in LAUNCHER_MODULES:
                # Source selection is data, never an import of research bodies.
                allowed = sys.stdlib_module_names | {module.split(".")[0], "lol_project_runtime"}
            elif relative in PURE_MODULES:
                allowed = sys.stdlib_module_names | ({"numpy"} if relative.startswith("master_ver/") else set())
            errors.extend(check_imports(source, module, allowed_roots=allowed))
        except (OSError, SyntaxError, UnicodeError) as exc:
            errors.append(f"{relative}: {exc}")
    return errors


def main() -> int:
    errors = check_project()
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Project boundaries OK: {len(METADATA_PATHS)} packaging declarations, {len(SOURCE_MODULES)} allowlisted AST files; no runtime imports executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
