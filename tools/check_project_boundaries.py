"""Check declared packaging and direct shared-component imports, without execution.

Only the paths in SOURCE_MODULES are parsed. In particular, protected cohort and
holdout implementations are never opened, recursively scanned, or imported.
This is a bounded structural check, not a transitive dependency/security audit.

S4b-prep K7 (D11, Q8) adds check_tree: every module under c_ept/ is parsed (never
imported), with the protected split modules excluded by name before any directory
listing entry is opened. Rules, applied to the imports of each module:
  * strict (c_ept/run/ and STRICT_FILES): no private (underscore) name or module of another
    package and no command module (c_ept/workflows/run_*.py and s4a_operator.py, tools/*,
    c_ept/comparison/operator.py, c_ept/s4a_operator.py). c_ept/run has no exception; a
    STRICT_FILES test file may keep an import on purpose in the allowlist section
    frozen_reference (a frozen stack it is compared against) or white_box (a private name
    inside c_ept.run.*, the kernel's own seam), each entry with a reason;
  * the rest of c_ept/: a private cross-package import must be listed in
    tools/boundary_allowlist.json (sections library and tests). Every section can only
    shrink: it may not exceed its ALLOWLIST_MAX_ENTRIES ceiling, a stale entry is an
    error, and no new violation passes.
Standard-library underscore modules (_winapi, os._exit, ...) are outside the rule.
Not checked: getattr(module, "_x") with a literal name, and anything built at run time.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
SHARED_FILES = (
    "c_ept/workflows/s4a_runtime.py",
    "c_ept/runtime/__init__.py",
    "c_ept/runtime/resources.py",
    "c_ept/ept/checkpoints.py",
    "c_ept/ept/training_step.py",
    "c_ept/ept/heads.py",
    "c_ept/ept/compose_heads.py",
    "c_ept/valuation/compose.py",
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


# ---------------------------------------------------------------------------
# K7 (D11, Q8): whole-tree check of c_ept/ with an allowlist that only shrinks.
#
# Symbols: m = a module, pkg(m) = the package that contains m (m itself if m is a
# package), P(t) = pkg of the part of an import target t before its first private
# component. A target t = "a.b.<_x>..." is a private cross-package import from an
# importer i when  pkg(i) != P(t)  and  pkg(i) is not inside the private component.
# 해석: `_`로 시작하는 이름은 그 이름이 놓인 패키지 안에서만 쓰라는 약속이다.
# 다른 패키지가 그 이름을 가져오면 패키지 사이의 계약이 사라지므로 위반으로 센다.
# 표준 라이브러리(_winapi 등)는 우리 계약이 아니므로 PROJECT_ROOTS 밖은 세지 않는다.
# ---------------------------------------------------------------------------
PROJECT_ROOTS = frozenset({"c_ept", "research_pipeline", "tools"})
# 보호된 분할 모듈: 이름으로 먼저 제외하며 열지도 파싱하지도 않는다.
PROTECTED_FILES = frozenset({"c_ept/cohort.py", "research_pipeline/cohort_policy.py"})
PROTECTED_DIRS = ("research_pipeline/holdout_r3",)
PROTECTED_MODULES = frozenset({"c_ept.cohort", "research_pipeline.cohort_policy",
                               "research_pipeline.holdout_r3"})
# Q8 "새로 고치는 파일": c_ept/run/ 전체와, S4b 준비에서 새로 만든 시험 파일은 엄격 규칙
# (밑줄 import 금지 + 명령 모듈 import 금지)을 받는다. 시험 파일의 예외는 허용 목록의
# 두 구역뿐이고(이유 필수), c_ept/run/ 은 예외가 없다.
#   frozen_reference: 비교 대상인 동결 스택(run_s3 등)을 일부러 가져오는 시험
#   white_box: 커널 자신(c_ept.run.*)의 밑줄 이음매를 직접 검사하는 시험
STRICT_PREFIXES = ("c_ept/run/",)
STRICT_FILES = frozenset({f"c_ept/tests/test_run_{name}.py"
                          for name in ("identity", "ledger", "supervisor", "guards", "io", "profile", "runners", "evaluate", "gate0", "numeric", "measure", "cli")}
                         | {"c_ept/tests/run_stage_bodies.py", "c_ept/tests/run_runner_bodies.py", "c_ept/tests/run_evaluate_bodies.py", "c_ept/tests/run_gate0_bodies.py",
                            "c_ept/tests/run_measure_bodies.py"})
TESTS_PREFIX = "c_ept/tests/"
SECTIONS = ("library", "tests", "frozen_reference", "white_box")
REASON_SECTIONS = ("frozen_reference", "white_box")
KERNEL_PREFIX = "c_ept.run."
ALLOWLIST_VERSION = 3
ALLOWLIST_FILE = "tools/boundary_allowlist.json"
# 구역별 상한(저장된 기준선). 위반을 고칠 때만 이 수와 목록을 함께 낮춘다. 예외는 한 번뿐이다(리드 판정 INT1: D25 합치기 뒤의 통합 기준선이며 계획의 X1 항목에 적혀 있다):
# tests를 101에서 109로 올렸다. Track M 시험 파일 test_ept_train, test_kcondition, test_killlaw가 가진 밑줄 import 8개이고, 사유는
# "S4b-prep Track M white-box test, integration baseline D25"이다(tests 구역 항목에는 사유 칸이 없어 여기와 커밋 메시지에 적는다).
# library, frozen_reference, white_box는 그대로이고, 이 기준선부터 모든 구역은 줄기만 한다.
ALLOWLIST_MAX_ENTRIES = {"library": 35, "tests": 109, "frozen_reference": 3, "white_box": 9}
_ENTRY_IMPORTER = re.compile(r"^c_ept/[A-Za-z0-9_]+(/[A-Za-z0-9_]+)*\.py$")
_ENTRY_TARGET = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$")


def is_protected(relative: str) -> bool:
    """True for a protected split path, decided by name only (no filesystem access)."""
    rel = relative.replace("\\", "/").strip("/").lower()
    return rel in PROTECTED_FILES or any(rel == d or rel.startswith(d + "/") for d in PROTECTED_DIRS)


def is_strict(relative: str) -> bool:
    return relative.startswith(STRICT_PREFIXES) or relative in STRICT_FILES


def section_of(importer: str, target: str) -> str:
    """The only allowlist section that may list (importer, target); "" when none may (c_ept/run).

    A strict test file keeps a private name inside c_ept.run.* in white_box (its own kernel's
    seam) and anything else (a frozen stack) in frozen_reference.
    """
    if importer.startswith(STRICT_PREFIXES):
        return ""
    if importer in STRICT_FILES:
        if not importer.startswith(TESTS_PREFIX):
            return ""
        return "white_box" if target.startswith(KERNEL_PREFIX) else "frozen_reference"
    return "tests" if importer.startswith(TESTS_PREFIX) else "library"


def _is_link(path: str) -> bool:
    """Symlink or (Windows) junction, decided from lstat only; nothing is opened."""
    is_junction = getattr(os.path, "isjunction", None)
    return os.path.islink(path) or bool(is_junction and is_junction(path))


def walk_package(root: Path = ROOT, links: list | None = None) -> list[str]:
    """Sorted posix paths of the .py files under c_ept/, protected paths pruned by name.

    os.walk only lists directory entries; a protected file or folder is dropped from the
    listing before any caller can open it. A symlink or junction is never followed or read
    (a protected module could hide behind a harmless link name); it is appended to `links`.
    """
    found = []
    for folder, dirs, files in os.walk(root / "c_ept", followlinks=False):
        relative_folder = Path(folder).relative_to(root).as_posix()
        kept = []
        for d in sorted(dirs):
            relative = f"{relative_folder}/{d}"
            if d == "__pycache__" or is_protected(relative):
                continue
            if _is_link(os.path.join(folder, d)):
                if links is not None:
                    links.append(relative)
                continue
            kept.append(d)
        dirs[:] = kept
        for name in files:
            relative = f"{relative_folder}/{name}"
            if not name.endswith(".py") or is_protected(relative):
                continue
            if _is_link(os.path.join(folder, name)):
                if links is not None:
                    links.append(relative)
                continue
            found.append(relative)
    return sorted(found)


def _package_of(relative: str) -> str:
    """Dotted package that contains the file (a package's __init__ belongs to the package)."""
    return ".".join(relative.split("/")[:-1])


def _is_private(part: str) -> bool:
    return part.startswith("_") and not (part.startswith("__") and part.endswith("__"))


def private_target(target: str, importer_package: str, modules: frozenset, packages: frozenset):
    """The private cross-package target (cut after the first private component), else None.

    Cutting keeps one key per private name: `m._f.cache_clear` and `m._f` are the same
    violation, so a harmless edit after the private name does not read as a new one.
    """
    parts = target.split(".")
    if parts[0] not in PROJECT_ROOTS:
        return None
    first = next((i for i in range(1, len(parts)) if _is_private(parts[i])), None)
    if first is None:
        return None
    prefix = ".".join(parts[:first])
    if parts[0] == "c_ept" and prefix not in modules and prefix not in PROTECTED_MODULES:
        return None  # e.g. a private attribute of a class: not a package-level name
    owner = prefix if prefix in packages else prefix.rpartition(".")[0]
    inner = f"{prefix}.{parts[first]}"
    if importer_package == owner or importer_package == inner or importer_package.startswith(inner + "."):
        return None
    return ".".join(parts[:first + 1])


def command_import(target: str) -> bool:
    """Whether `target` is a command module that strict files must not import.

    Exactly (Q8 and the lead ruling): c_ept.workflows.run_*, c_ept.workflows.s4a_operator,
    tools.*, c_ept.comparison.operator and the package entry c_ept.s4a_operator. Library modules
    such as c_ept.ept.train are not command modules.
    """
    parts = target.split(".")
    return (parts[0] == "tools"
            or (parts[:2] == ["c_ept", "workflows"] and len(parts) > 2
                and (parts[2].startswith("run_") or parts[2] == "s4a_operator"))
            or parts[:3] == ["c_ept", "comparison", "operator"]
            or parts[:2] == ["c_ept", "s4a_operator"])


def _resolve(node: ast.ImportFrom, package: list) -> str:
    base = node.module or ""
    if node.level:
        prefix = package[:len(package) - node.level + 1]
        base = ".".join([*prefix, *([base] if base else [])])
    return base


def _literal_call_target(node: ast.Call, package: list):
    """The literal module of importlib.import_module('x') / __import__('x'), as import_targets reads it."""
    if not node.args:
        return None
    name = node.func.id if isinstance(node.func, ast.Name) else (
        node.func.attr if isinstance(node.func, ast.Attribute) else "")
    arg = node.args[0]
    if name not in {"import_module", "__import__"} or not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
        return None
    target = arg.value
    if target.startswith("."):
        level = len(target) - len(target.lstrip("."))
        target = ".".join([*package[:len(package) - level + 1], target.lstrip(".")])
    return target


def scan_tree_imports(tree: ast.AST, relative: str, modules: frozenset, packages: frozenset):
    """Yield (kind, line, target) with kind 'private' or 'command' for one parsed module.

    The alias map is flow-insensitive and file-global (a rebinding of an imported module name
    can give a false positive); a structural check, not data-flow analysis.
    """
    module = relative[:-3].replace("/", ".")
    package = module.split(".")[:-1]
    importer_package = _package_of(relative)

    def private(target):
        return private_target(target, importer_package, modules, packages)

    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
                if private(alias.name):
                    yield "private", node.lineno, private(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(node, package)
            if base and private(base):
                yield "private", node.lineno, private(base)  # the module path itself is private
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                full = f"{base}.{alias.name}"
                if full in modules:
                    aliases[alias.asname or alias.name] = full
                if base and private(full):
                    yield "private", node.lineno, private(full)
        elif isinstance(node, ast.Call):
            literal = _literal_call_target(node, package)
            if literal and private(literal):
                yield "private", node.lineno, private(literal)
    inner = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and id(node) not in inner:
            chain = []
            value = node
            while isinstance(value, ast.Attribute):
                chain.append(value.attr)
                value = value.value
            base = None
            if isinstance(value, ast.Name) and value.id in aliases:
                base = aliases[value.id]
            elif isinstance(value, ast.Call):
                base = _literal_call_target(value, package)
            if base and private(".".join([base, *reversed(chain)])):
                yield "private", node.lineno, private(".".join([base, *reversed(chain)]))
    if is_strict(relative):
        for line, target in import_targets(tree, module):
            if command_import(target):
                yield "command", line, target


def _reject_duplicates(pairs):
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate JSON key")
    return dict(pairs)


def _reject_constant(name):
    raise ValueError(f"non-finite JSON constant {name}")


def load_allowlist(path: Path) -> dict:
    """Strict read of {version: 3, library, tests, frozen_reference, white_box}.

    Returns {section: [(importer, target) ...]}; frozen_reference and white_box tuples carry a
    third item, the reason. No duplicate keys, no NaN, no extra keys (a line number is refused),
    each section sorted by (importer, target) and unique.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant)
    if not isinstance(data, dict) or set(data) != {"version", *SECTIONS} or data["version"] != ALLOWLIST_VERSION:
        raise ValueError(f"expected exactly {{version: {ALLOWLIST_VERSION}, {', '.join(SECTIONS)}}}")
    sections = {}
    for name in SECTIONS:
        with_reason = name in REASON_SECTIONS
        keys = {"importer", "target", "reason"} if with_reason else {"importer", "target"}
        entries = []
        for item in data[name] if isinstance(data[name], list) else [None]:
            if not (isinstance(item, dict) and set(item) == keys
                    and all(isinstance(item[key], str) for key in keys)
                    and _ENTRY_IMPORTER.match(item["importer"]) and _ENTRY_TARGET.match(item["target"])
                    and (not with_reason or item["reason"].strip())):
                raise ValueError(f"{name}: bad entry {item!r}: only importer file, imported name "
                                 f"(and a reason in {' and '.join(REASON_SECTIONS)}) are allowed")
            entries.append((item["importer"], item["target"], *([item["reason"]] if with_reason else [])))
        keyed = [entry[:2] for entry in entries]
        if keyed != sorted(set(keyed)):
            raise ValueError(f"{name}: entries must be sorted by importer and target, and unique")
        sections[name] = entries
    return sections


def write_allowlist(path: Path, sections: dict) -> None:
    payload = {"version": ALLOWLIST_VERSION}
    for name in SECTIONS:
        ordered = sorted(sections.get(name, ()), key=lambda entry: entry[:2])
        keys = ("importer", "target", "reason") if name in REASON_SECTIONS else ("importer", "target")
        payload[name] = [dict(zip(keys, entry)) for entry in ordered]
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, indent=1, ensure_ascii=True) + "\n")


def scan_package(root: Path = ROOT):
    """Return (files, violations, errors); violations are (relative, line, kind, target)."""
    links = []
    files = walk_package(root, links)
    packages = frozenset(_package_of(f) for f in files if f.endswith("/__init__.py"))
    modules = frozenset({f[:-3].replace("/", ".") for f in files if not f.endswith("/__init__.py")} | packages)
    violations = []
    errors = [f"{link}: symlink or junction under c_ept/ refused (not followed, not read)" for link in sorted(links)]
    for relative in files:
        try:
            source = (root / relative).read_text(encoding="utf-8-sig")
            tree = ast.parse(source, filename=relative)
        except (OSError, SyntaxError, UnicodeError) as exc:
            errors.append(f"{relative}: {exc}")
            continue
        violations.extend((relative, line, kind, target)
                          for kind, line, target in scan_tree_imports(tree, relative, modules, packages))
    return files, violations, errors


def check_tree(root: Path = ROOT, *, allowlist_path: Path | None = None, max_entries: dict | None = None) -> list[str]:
    """Strict rule for c_ept/run/ and STRICT_FILES, shrink-only allowlist for the rest of c_ept/.

    Sections: library and tests list private cross-package imports of non-strict modules;
    frozen_reference lists the (private or command) imports a strict test file keeps on purpose
    to compare against a frozen stack; white_box lists the private names inside c_ept.run.* that a
    strict test file of the kernel reaches (the kernel's own seams). Both carry a reason, and c_ept/run
    itself is never listed. Every section has its own ceiling in max_entries.
    """
    allowlist_path = Path(allowlist_path) if allowlist_path else root / ALLOWLIST_FILE
    limits = ALLOWLIST_MAX_ENTRIES if max_entries is None else max_entries
    try:
        sections = load_allowlist(allowlist_path)
    except (OSError, ValueError) as exc:
        return [f"boundary allowlist {allowlist_path.name}: {exc}"]
    _, violations, errors = scan_package(root)
    allowed = {}
    for name in SECTIONS:
        entries = sections[name]
        if len(entries) > limits[name]:
            errors.append(f"boundary allowlist section {name} has {len(entries)} entries, "
                          f"larger than the committed baseline {limits[name]}")
        for entry in entries:
            proper = section_of(entry[0], entry[1])
            if proper != name:
                why = (f"strict file: only a c_ept/tests strict file may be listed, in {proper}"
                       if is_strict(entry[0]) and proper else
                       "strict file: c_ept/run is never listed" if is_strict(entry[0]) else
                       f"belongs in section {proper}")
                errors.append(f"boundary allowlist: {entry[0]} -> {entry[1]} cannot be listed in {name} ({why})")
            else:
                allowed[entry[:2]] = name
    seen = set()
    for relative, line, kind, target in violations:
        if is_strict(relative):
            what = "imports command module" if kind == "command" else "private import across packages"
            if allowed.get((relative, target)) in set(REASON_SECTIONS):
                seen.add((relative, target))
                continue
            errors.append(f"{relative}:{line}: strict rule: {what} {target}")
        elif kind == "private":
            seen.add((relative, target))
            if allowed.get((relative, target)) not in {"library", "tests"}:
                errors.append(f"{relative}:{line}: private import {target} is not in the allowlist")
    for pair in sorted(allowed):
        if pair not in seen:
            errors.append(f"boundary allowlist: stale entry {pair[0]} -> {pair[1]}; remove it and lower the baseline")
    return errors


def main() -> int:
    errors = check_project()
    errors.extend(check_tree())
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    files = walk_package()
    sections = load_allowlist(ROOT / ALLOWLIST_FILE)
    counts = ", ".join(f"{name} {len(sections[name])}/{ALLOWLIST_MAX_ENTRIES[name]}" for name in SECTIONS)
    print(f"Project boundaries OK: {len(METADATA_PATHS)} packaging declarations, {len(SOURCE_MODULES)} allowlisted AST files; no runtime imports executed. "
          f"Tree: {len(files)} c_ept modules parsed (protected paths excluded by name), allowlisted imports (count/baseline): {counts}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
