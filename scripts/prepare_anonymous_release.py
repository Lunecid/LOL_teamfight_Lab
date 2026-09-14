#!/usr/bin/env python3
"""Prepare a local, anonymised copy of this repository for double-anonymous review.

LOCAL ONLY.  The script reads the local git object database and writes plain files into a
directory on this machine.  It never pushes, never creates or changes a repository, never
uploads anything and opens no network connection.  Its git calls are limited to the read-only
subcommands in ``_READ_ONLY_GIT``; any other subcommand raises before it runs.  What happens to
the output (for example, attaching it to a submission as supplementary material) is left to the
authors.

Steps
  1. Export the files tracked at ``--commit`` (default ``HEAD``) with ``git archive``.  Only the
     file tree is written: no ``.git`` directory and no commit history, because the history
     carries author names and e-mail addresses.  Uncommitted edits are therefore not included.
     ``--exclude GLOB`` (repeatable) leaves matching paths out of the copy, for example
     documents that describe an earlier review of the same work.
  2. In every UTF-8 text file, replace identifiers with neutral tokens (``ANONYMOUS_AUTHOR``,
     ``ANONYMOUS_HANDLE``, ``ANONYMOUS_USER``, ``ANONYMOUS``, ``anonymous@example.org``,
     ``example.org``, ``ANONYMOUS_REPOSITORY_URL``, ``anonymous-project``, ``<USER_HOME>``,
     ``<USER_FOLDER>``, and ``anon<year>`` for citation keys).
  3. Remove the copyright holder's name from LICENSE / COPYING files.
  4. Scan the exported tree (text and non-text files) for residual identifiers and print a
     report.  Strings that may identify the authors but are not replaced automatically are listed
     for a manual check: parts of names, every line that held a replaced name or citation key
     (a co-author who never committed to the repository is not in the git metadata, so a name
     next to a replaced one survives), references to an earlier submission of the same work
     (its submission number, the summary review of its programme committee), other e-mail
     addresses and GitHub URLs.  A double-
     anonymous venue may treat a reference to the authors' own earlier, unpublished submission
     as identifying, so those lines need a decision by the authors.

Identifiers are collected at run time, so nothing identifying is written into this file and the
file itself can ship inside the anonymised copy:
  * ``git config --get user.name`` and ``user.email``;
  * author and committer names and e-mail addresses in the history of ``--commit`` (assistant,
    bot and hosting-service entries such as ``noreply@anthropic.com`` are skipped);
  * the owner and repository name in the ``origin`` remote URL;
  * the copyright holder named in LICENSE;
  * the local account name (the last component of the home directory);
  * ``--author "Given Family"`` (repeatable; also matches "Family, Given", "Family Given",
    "G. Family", "Family, G." and citation keys such as ``family2026``), ``--identifier TEXT``
    (repeatable, e.g. an affiliation) and ``--identifiers-file PATH`` (one entry per line,
    ``author:`` prefix for a person, ``#`` starts a comment; keep that file outside the repository).

The local project folder name is replaced like the repository name (underscore and hyphen
spellings only, so that ordinary phrases are left alone); ``--keep-project-name`` keeps both and
lists them for review instead.

Exit status: 0 when the scan finds nothing, 1 when a known identifier survives (the report lists
every hit), 3 when no known identifier survives but items are listed for manual review, 2 on a
usage error.  Only 0 means that the copy needs no further look before it is shared.

    python scripts/prepare_anonymous_release.py --commit HEAD \\
        --out ../anon_release_preview/head --author "Given Family"
"""
from __future__ import annotations

import argparse
import fnmatch
import io
import re
import subprocess
import sys
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

# git subcommands this script may run, and the only first argument each may take (None = any).
# All of them only read the object database or the configuration.
_READ_ONLY_GIT: Dict[str, Optional[Tuple[str, ...]]] = {
    "archive": ("--format=tar",),
    "rev-parse": None,
    "log": None,
    "config": ("--get",),
    "remote": ("get-url",),
}

TOKEN_AUTHOR = "ANONYMOUS_AUTHOR"
TOKEN_HANDLE = "ANONYMOUS_HANDLE"
TOKEN_USER = "ANONYMOUS_USER"
TOKEN_OTHER = "ANONYMOUS"
TOKEN_EMAIL = "anonymous@example.org"
TOKEN_DOMAIN = "example.org"
TOKEN_REPO_URL = "ANONYMOUS_REPOSITORY_URL"
TOKEN_PROJECT = "anonymous-project"
TOKEN_HOME = "<USER_HOME>"
TOKEN_FOLDER = "<USER_FOLDER>"
TOKEN_CITEKEY = "anon"
NOTE_NAME = "ANONYMIZATION_NOTE.txt"
NOTE_TEXT = """This directory is an anonymised export of one commit of a git repository, prepared for
double-anonymous peer review.  The commit history is not included.  Author names, account
handles, e-mail addresses, the repository URL and name, and absolute user paths were replaced
by neutral tokens:

  ANONYMOUS_AUTHOR, ANONYMOUS_HANDLE, ANONYMOUS_USER, ANONYMOUS, anonymous@example.org,
  example.org, ANONYMOUS_REPOSITORY_URL, anonymous-project, <USER_HOME>, <USER_FOLDER>,
  anon<year> (citation keys)
"""

# Commit metadata that identifies an assistant, a bot or a hosting service rather than a person.
_NON_PERSON_EMAIL = re.compile(r"noreply@anthropic\.com|noreply@github\.com|\[bot\]|github-actions", re.I)
_NON_PERSON_NAME = re.compile(r"^(?:claude|codex|copilot|github|dependabot|the authors?|contributors)\b", re.I)
_GENERIC_DOMAINS = {
    "gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "live.com", "icloud.com",
    "naver.com", "daum.net", "hanmail.net", "users.noreply.github.com", "github.com",
    "anthropic.com", "example.org", "example.com",
}
_GENERIC_LOCAL_PARTS = {"noreply", "no-reply", "info", "admin", "contact", "mail", "users"}
_GENERIC_ACCOUNTS = {"user", "users", "admin", "administrator", "owner", "runner", "ubuntu", "root", "home"}

# Korean display names of Windows user folders (Desktop, Downloads, Videos, Documents, Pictures,
# Music).  They are replaced only where they form a path segment; as ordinary words they stay.
_KOREAN_USER_FOLDERS = ("바탕 화면", "바탕화면", "다운로드", "동영상", "문서", "사진", "음악")
_KF = "|".join(re.escape(k) for k in _KOREAN_USER_FOLDERS)
KOREAN_FOLDER_RE = re.compile(rf"(?<=[\\/])(?:{_KF})|(?<!\w)(?:{_KF})(?=[\\/])")
KOREAN_WORD_RE = re.compile(rf"(?:{_KF})")
WIN_HOME_RE = re.compile(r"\b[A-Za-z]:(?:\\{1,2}|/)Users(?:\\{1,2}|/)[^\\/\s\"'`<>|:*?]+", re.I)
POSIX_HOME_RE = re.compile(r"(?<![\w.:/~-])(?:/[A-Za-z](?=/))?/(?:Users|home)/[^/\s\"'`<>]+")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
GITHUB_URL_RE = re.compile(r"github\.com[:/]+[\w.-]+(?:/[\w.-]+)?", re.I)
REMOTE_RE = re.compile(r"github\.com[:/]+([^/\s]+)/([^/\s]+?)(?:\.git)?/?$", re.I)
COPYRIGHT_RE = re.compile(
    r"^([ \t]*Copyright[ \t]*(?:\([cC]\)|©)?[ \t]*\d{4}(?:[ \t]*[-–,][ \t]*\d{4})*)[ \t]+(\S[^\n]*?)[ \t]*$",
    re.I | re.M,
)
LICENSE_NAME_RE = re.compile(r"^(?:licen[cs]e|copying)(?:\.[\w.-]+)?$", re.I)
# References to an earlier submission of the same work: the word submission or paper followed by a
# number, a venue abbreviation and year followed by '#' and a number, and the words for a summary
# review and its author (meta reviewer, hyphenated or not).  Review only.  The comment spells no
# example out, so that this file does not list itself for review.
PRIOR_SUBMISSION_RES: Tuple["re.Pattern[str]", ...] = (
    re.compile(r"\b(?:submission|paper)\s*(?:no\.?\s*|number\s*|id\s*|#\s*)?\d{2,5}\b", re.I),
    re.compile(r"\b[a-z]{2,8}\s*(?:'?\d{2}|\d{4})\s*#\s*\d{2,5}\b", re.I),
    re.compile(r"\bmeta-?review(?:er|s)?\b", re.I),
)


# ----------------------------------------------------------------------------- git (read only)
def _git(repo: Path, *args: str, optional: bool = False) -> bytes:
    sub = args[0] if args else ""
    if sub not in _READ_ONLY_GIT:
        raise PermissionError(f"refusing to run 'git {sub}': only {sorted(_READ_ONLY_GIT)} are allowed")
    first = _READ_ONLY_GIT[sub]
    if first is not None and (len(args) < 2 or args[1] not in first):
        raise PermissionError(f"refusing to run 'git {' '.join(args)}': 'git {sub}' is allowed only with {first}")
    if any(a.startswith(("--remote", "--exec", "--output", "-o")) for a in args[1:]):
        # "git archive --remote" would contact a server and "--output" would write a file.
        raise PermissionError(f"refusing to run 'git {' '.join(args)}': remote, exec and output options are not allowed")
    proc = subprocess.run(["git", "-C", str(repo), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        if optional:
            return b""
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.decode('utf-8', 'replace').strip()}")
    return proc.stdout


def _git_text(repo: Path, *args: str, optional: bool = False) -> str:
    return _git(repo, *args, optional=optional).decode("utf-8", "replace").strip()


def is_excluded(rel: str, patterns: Sequence[str]) -> bool:
    """True if the POSIX path ``rel`` or one of its parent directories matches a glob."""
    parts = rel.strip("/").split("/")
    prefixes = ["/".join(parts[: i + 1]) for i in range(len(parts))]
    return any(fnmatch.fnmatchcase(p, pat.strip("/")) for pat in patterns for p in prefixes)


def export_commit(repo: Path, sha: str, out: Path, exclude: Sequence[str] = ()) -> List[str]:
    """Write the file tree of ``sha`` into ``out`` (no history, no .git); return excluded files."""
    data = _git(repo, "archive", "--format=tar", sha)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
        members, skipped = [], []
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if member.name.startswith(("/", "\\")) or ".." in parts or (parts and ":" in parts[0]):
                raise RuntimeError(f"unsafe path in archive: {member.name}")
            if exclude and is_excluded(member.name, exclude):
                if member.isfile():
                    skipped.append(member.name)
                continue
            members.append(member)
        if sys.version_info >= (3, 12):
            tar.extractall(out, members=members, filter="data")
        else:  # pragma: no cover - older interpreters
            tar.extractall(out, members=members)
    return sorted(skipped)


# ----------------------------------------------------------------------------- identifiers
@dataclass
class Identifiers:
    persons: Set[str] = field(default_factory=set)
    handles: Set[str] = field(default_factory=set)
    repos: Set[str] = field(default_factory=set)
    emails: Set[str] = field(default_factory=set)
    domains: Set[str] = field(default_factory=set)
    users: Set[str] = field(default_factory=set)
    others: Set[str] = field(default_factory=set)
    project_folder: str = ""
    sources: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))


def person_variants(name: str) -> List[str]:
    parts = name.split()
    variants = {name}
    if len(parts) >= 2:
        given, family = " ".join(parts[:-1]), parts[-1]
        variants.update({f"{family}, {given}", f"{family} {given}", f"{given[0]}. {family}", f"{family}, {given[0]}."})
    return sorted(variants, key=len, reverse=True)


def project_variants(repos: Set[str]) -> List[str]:
    variants: Set[str] = set()
    for name in repos:
        if len(name) >= 6:
            variants.update({name, name.replace("_", " "), name.replace("_", "-"),
                             name.replace("-", "_"), name.replace("-", " ")})
    return sorted(variants, key=len, reverse=True)


def license_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and LICENSE_NAME_RE.match(p.name))


def collect_identifiers(repo: Path, sha: str, out: Path, authors: Sequence[str], others: Sequence[str]) -> Identifiers:
    ids = Identifiers()

    def add_person(name: str, source: str, *, trusted: bool = False) -> None:
        name = " ".join(name.split())
        if len(name) >= 3 and (trusted or not _NON_PERSON_NAME.search(name)):
            ids.persons.add(name)
            ids.sources[source].add(name)

    def add_email(addr: str, source: str) -> None:
        addr = addr.strip()
        if "@" not in addr or _NON_PERSON_EMAIL.search(addr):
            return
        ids.emails.add(addr)
        ids.sources[source].add(addr)
        local, _, domain = addr.partition("@")
        if len(local) >= 5 and local.lower() not in _GENERIC_LOCAL_PARTS:
            ids.users.add(local)
        if domain.lower() not in _GENERIC_DOMAINS:
            ids.domains.add(domain.lower())

    add_person(_git_text(repo, "config", "--get", "user.name", optional=True), "git config")
    add_email(_git_text(repo, "config", "--get", "user.email", optional=True), "git config")
    for line in _git_text(repo, "log", "--format=%an%x00%ae%x00%cn%x00%ce", sha).splitlines():
        fields = line.split("\x00")
        if len(fields) != 4:
            continue
        for name, addr in ((fields[0], fields[1]), (fields[2], fields[3])):
            if _NON_PERSON_EMAIL.search(addr):
                continue
            add_person(name, "git history")
            add_email(addr, "git history")

    match = REMOTE_RE.search(_git_text(repo, "remote", "get-url", "origin", optional=True))
    if match:
        ids.handles.add(match.group(1))
        ids.repos.add(match.group(2))
        ids.sources["origin remote"].update(match.groups())

    for lic in license_files(out):
        for m in COPYRIGHT_RE.finditer(lic.read_text(encoding="utf-8", errors="replace")):
            for holder in re.split(r"\s*(?:,|&|\band\b)\s*", m.group(2)):
                add_person(holder, lic.relative_to(out).as_posix())

    account = Path.home().name
    if len(account) >= 4 and account.lower() not in _GENERIC_ACCOUNTS:
        ids.users.add(account)
        ids.sources["home directory"].add(account)

    for name in authors:
        add_person(name, "--author", trusted=True)
    for text in others:
        if text.strip():
            ids.others.add(text.strip())
            ids.sources["--identifier"].add(text.strip())
    ids.project_folder = Path(_git_text(repo, "rev-parse", "--show-toplevel")).name
    return ids


# ----------------------------------------------------------------------------- rules
@dataclass(frozen=True)
class Rule:
    category: str
    label: str
    pattern: "re.Pattern[str]"
    token: str


def _literal(text: str, *, word: bool) -> "re.Pattern[str]":
    body = re.escape(text)
    if word and re.match(r"\w", text):
        body = r"(?<!\w)" + body
    if word and re.search(r"\w$", text):
        body = body + r"(?!\w)"
    return re.compile(body, re.IGNORECASE)


def build_rules(ids: Identifiers, keep_project_name: bool) -> Tuple[List[Rule], List[Rule]]:
    """Return (rules applied in order, review-only rules)."""
    replace: List[Rule] = []
    review: List[Rule] = []
    for owner in sorted(ids.handles, key=len, reverse=True):
        replace.append(Rule("repository URL", f"github.com/{owner}/...", re.compile(
            rf"(?:https?://|git@)?(?:www\.)?github\.com[:/]+{re.escape(owner)}(?:/[\w.-]+)?", re.I), TOKEN_REPO_URL))
    for addr in sorted(ids.emails, key=len, reverse=True):
        replace.append(Rule("e-mail", addr, _literal(addr, word=False), TOKEN_EMAIL))
    for dom in sorted(ids.domains, key=len, reverse=True):
        d = re.escape(dom)
        replace.append(Rule("e-mail", f"*@{dom}", re.compile(rf"[\w.+-]+@(?:[\w-]+\.)*{d}(?![\w-])", re.I), TOKEN_EMAIL))
        replace.append(Rule("institution domain", dom, re.compile(rf"(?<![\w-])(?:[\w-]+\.)*{d}(?![\w-])", re.I), TOKEN_DOMAIN))
    for variant in sorted({v for p in ids.persons for v in person_variants(p)}, key=len, reverse=True):
        replace.append(Rule("person name", variant, _literal(variant, word=True), TOKEN_AUTHOR))
    families = {p.split()[-1] for p in ids.persons if len(p.split()) >= 2 and len(p.split()[-1]) >= 3}
    for fam in sorted(families, key=len, reverse=True):
        replace.append(Rule("citation key", f"{fam}<year>", re.compile(rf"(?<!\w){re.escape(fam)}(?=\d{{4}})", re.I), TOKEN_CITEKEY))
    for handle in sorted(ids.handles, key=len, reverse=True):
        replace.append(Rule("handle", handle, _literal(handle, word=False), TOKEN_HANDLE))
    slugs = set(project_variants(ids.repos))
    folder = ids.project_folder
    if len(folder) >= 6:
        # Underscore and hyphen spellings only: with a space, a folder name such as "chess_openings"
        # would also match ordinary topic phrases ("chess openings in the literature").
        slugs.update({folder, folder.replace("_", "-"), folder.replace("-", "_")})
    for slug in sorted(slugs, key=len, reverse=True):
        rule = Rule("project name", slug, re.compile(rf"(?<![\w-]){re.escape(slug)}(?![\w-])", re.I), TOKEN_PROJECT)
        (review if keep_project_name else replace).append(rule)
    replace.append(Rule("user path", "<drive>:\\Users\\<name>", WIN_HOME_RE, TOKEN_HOME))
    replace.append(Rule("user path", "/Users/<name>, /home/<name>", POSIX_HOME_RE, TOKEN_HOME))
    replace.append(Rule("Korean user-folder path segment", " | ".join(_KOREAN_USER_FOLDERS), KOREAN_FOLDER_RE, TOKEN_FOLDER))
    for user in sorted(ids.users, key=len, reverse=True):
        replace.append(Rule("account name", user, _literal(user, word=False), TOKEN_USER))
    for text in sorted(ids.others, key=len, reverse=True):
        replace.append(Rule("other identifier", text, _literal(text, word=True), TOKEN_OTHER))

    parts = {tok for p in ids.persons if len(p.split()) >= 2 for tok in p.split() if len(tok) >= 4}
    for tok in sorted(parts, key=len, reverse=True):
        review.append(Rule("name part", tok, _literal(tok, word=True), ""))
    if 4 <= len(folder) < 6:
        review.append(Rule("local project folder name", folder, _literal(folder, word=False), ""))
    for pattern in PRIOR_SUBMISSION_RES:
        review.append(Rule("earlier-submission reference", pattern.pattern, pattern, ""))
    return replace, review


# Lines that held a replaced person name or citation key: co-authors and titles sit there.
CONTEXT_RULE = Rule("line with a replaced author name (check co-authors and title)", TOKEN_AUTHOR,
                    re.compile(rf"{TOKEN_AUTHOR}|(?<!\w){TOKEN_CITEKEY}\d{{4}}"), "")


def apply_rules(text: str, rules: Sequence[Rule]) -> Tuple[str, Counter]:
    counts: Counter = Counter()
    for rule in rules:
        text, n = rule.pattern.subn(rule.token, text)
        if n:
            counts[rule.category] += n
    return text, counts


def find_hits(text: str, rules: Sequence[Rule]) -> List[Tuple[int, Rule, str]]:
    hits = []
    for rule in rules:
        for m in rule.pattern.finditer(text):
            hits.append((text.count("\n", 0, m.start()) + 1, rule, m.group(0)))
    return hits


def binary_literals(ids: Identifiers) -> List[str]:
    lits = set(ids.emails) | set(ids.handles) | set(ids.users) | set(ids.others) | set(ids.domains)
    lits |= {v for p in ids.persons for v in person_variants(p)}
    lits |= {f"{k}/" for k in _KOREAN_USER_FOLDERS} | {f"{k}\\" for k in _KOREAN_USER_FOLDERS}
    return sorted(lits, key=len, reverse=True)


def scan_bytes(data: bytes, literals: Sequence[str]) -> Counter:
    low = data.lower()
    found: Counter = Counter()
    for lit in literals:
        for encoding in ("utf-8", "utf-16-le"):
            n = low.count(lit.lower().encode(encoding))
            if n:
                found[lit] += n
    return found


def as_text(data: bytes) -> Optional[str]:
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


# ----------------------------------------------------------------------------- main
def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(
        description="Local anonymised export of one commit for double-anonymous review. "
                    "Never pushes, creates repositories or uploads.")
    ap.add_argument("--out", type=Path, required=True,
                    help="output directory outside the repository; must not exist or must be empty")
    ap.add_argument("--commit", default="HEAD", help="commit, tag or branch to export (default: HEAD)")
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1],
                    help="repository to read (default: the repository containing this script)")
    ap.add_argument("--author", action="append", default=[], metavar="NAME",
                    help="person name to replace, e.g. a co-author absent from git metadata (repeatable)")
    ap.add_argument("--identifier", action="append", default=[], metavar="TEXT",
                    help="any other string to replace, e.g. an affiliation (repeatable)")
    ap.add_argument("--identifiers-file", type=Path, default=None,
                    help="one entry per line; 'author:' prefix marks a person; '#' starts a comment")
    ap.add_argument("--keep-project-name", action="store_true",
                    help="do not replace the repository and project folder names (they are then listed for review)")
    ap.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                    help="leave tracked paths matching this glob (or inside a matching directory) out of the copy, "
                         "e.g. 'docs/earlier_venue_*' (repeatable)")
    ap.add_argument("--review-pattern", action="append", default=[], metavar="REGEX",
                    help="extra case-insensitive regular expression whose hits are listed for manual review, "
                         "not replaced, e.g. a venue and year (repeatable)")
    ap.add_argument("--max-lines", type=int, default=60, help="hits printed per report section (default 60)")
    args = ap.parse_args(argv)

    try:
        top = Path(_git_text(args.repo, "rev-parse", "--show-toplevel")).resolve()
        sha = _git_text(args.repo, "rev-parse", "--verify", f"{args.commit}^{{commit}}")
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = args.out.resolve()
    if out == top or top in out.parents:
        print(f"error: --out must lie outside the repository ({top}) so the copy cannot be committed by accident",
              file=sys.stderr)
        return 2
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        print(f"error: {out} exists and is not an empty directory; choose a new path (this script never deletes files)",
              file=sys.stderr)
        return 2
    authors, others = list(args.author), list(args.identifier)
    if args.identifiers_file is not None:
        for raw in args.identifiers_file.read_text(encoding="utf-8").splitlines():
            entry = raw.split("#", 1)[0].strip()
            if not entry:
                continue
            if entry.lower().startswith("author:"):
                authors.append(entry[len("author:"):].strip())
            else:
                others.append(entry)

    try:
        extra_review = [Rule("--review-pattern", text, re.compile(text, re.I), "") for text in args.review_pattern]
    except re.error as exc:
        print(f"error: --review-pattern is not a valid regular expression: {exc}", file=sys.stderr)
        return 2

    # A copy inside some other working tree could still be committed there; say so.
    probe = out
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    other_top = _git_text(probe, "rev-parse", "--show-toplevel", optional=True)
    warnings: List[str] = []
    if other_top:
        warnings.append(f"--out lies inside another git working tree ({other_top}); do not add it there")

    out.mkdir(parents=True, exist_ok=True)
    excluded = export_commit(args.repo, sha, out, args.exclude)
    ids = collect_identifiers(args.repo, sha, out, authors, others)
    replace_rules, review_rules = build_rules(ids, args.keep_project_name)
    review_rules += extra_review

    changed: Dict[str, Counter] = {}
    totals: Counter = Counter()
    # Lines that already held a neutral token before replacement (for example this script's own
    # token constants) are not evidence of a replaced name; the context check below skips them.
    # No replacement pattern spans a line break, so line numbers are unchanged by replacement.
    token_lines_before: Dict[str, Set[int]] = {}
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        text = as_text(path.read_bytes())
        if text is None:
            continue
        token_lines_before[path.relative_to(out).as_posix()] = {
            number for number, content in enumerate(text.splitlines(), 1) if CONTEXT_RULE.pattern.search(content)
        }
        new, counts = text, Counter()
        if LICENSE_NAME_RE.match(path.name):
            new, n = COPYRIGHT_RE.subn(lambda m: f"{m.group(1)} the authors (name withheld for anonymous review)", new)
            if n:
                counts["LICENSE copyright holder"] += n
        new, more = apply_rules(new, replace_rules)
        counts.update(more)
        if new != text:
            path.write_bytes(new.encode("utf-8"))
            changed[path.relative_to(out).as_posix()] = counts
            totals.update(counts)
    note = out / NOTE_NAME
    if not note.exists():
        note.write_text(NOTE_TEXT, encoding="utf-8")

    # ---- scan the finished tree
    residual: List[Tuple[str, int, Rule, str]] = []
    review: List[Tuple[str, int, Rule, str]] = []
    other_emails: Counter = Counter()
    github_urls: Counter = Counter()
    korean_words: Counter = Counter()
    binary_hits: Dict[str, Counter] = {}
    literals = binary_literals(ids)
    n_text = n_other = 0
    files = sorted(p for p in out.rglob("*") if p.is_file())
    for path in files:
        rel = path.relative_to(out).as_posix()
        data = path.read_bytes()
        text = as_text(data)
        if text is None:
            n_other += 1
            found = scan_bytes(data, literals)
            if found:
                binary_hits[rel] = found
            continue
        n_text += 1
        residual += [(rel, line, rule, hit) for line, rule, hit in find_hits(text, replace_rules)]
        review += [(rel, line, rule, hit) for line, rule, hit in find_hits(text, review_rules)]
        if rel != NOTE_NAME:
            # A co-author who never committed is not in the git metadata, so the name next to a
            # replaced one survives (e.g. "ANONYMOUS_AUTHOR and <co-author>"); list those lines.
            already = token_lines_before.get(rel, set())
            for number, content in enumerate(text.splitlines(), 1):
                if number not in already and CONTEXT_RULE.pattern.search(content):
                    review.append((rel, number, CONTEXT_RULE, content.strip()[:160]))
        for m in EMAIL_RE.finditer(text):
            if m.group(0).lower() != TOKEN_EMAIL:
                other_emails[m.group(0)] += 1
        for m in GITHUB_URL_RE.finditer(text):
            github_urls[m.group(0)] += 1
        n_words = len(KOREAN_WORD_RE.findall(text))
        if n_words:
            korean_words[rel] = n_words

    cap = max(1, args.max_lines)
    print("== anonymised export (local only: nothing was pushed, created remotely or uploaded) ==")
    print(f"repository   {top}")
    print(f"commit       {sha} ({args.commit})")
    print(f"output       {out}")
    print(f"files        {len(files)} ({n_text} UTF-8 text, {n_other} other)")
    for warning in warnings:
        print(f"WARNING      {warning}")
    print(f"\n-- tracked files left out by --exclude: {len(excluded)} --")
    for rel in excluded[:cap]:
        print(f"  {rel}")
    print("\n-- identifiers collected (source: values) --")
    for source in sorted(ids.sources):
        print(f"  {source}: {', '.join(sorted(ids.sources[source]))}")
    print(f"  local project folder: {ids.project_folder}"
          + (" (listed for review)" if args.keep_project_name else " (replaced)"))
    print(f"\n-- replacements: {sum(totals.values())} in {len(changed)} files --")
    for category, n in totals.most_common():
        print(f"  {category}: {n}")
    for rel in sorted(changed):
        print(f"  {rel}: " + ", ".join(f"{c} x{n}" for c, n in changed[rel].most_common()))
    print(f"\n-- residual identifier hits in text files: {len(residual)} --")
    for rel, line, rule, hit in residual[:cap]:
        print(f"  {rel}:{line}: [{rule.category}] {hit!r}")
    print(f"\n-- non-text files containing identifier bytes: {len(binary_hits)} --")
    for rel, found in sorted(binary_hits.items())[:cap]:
        print(f"  {rel}: " + ", ".join(f"{k!r} x{n}" for k, n in found.most_common()))
    print(f"\n-- for manual review (not replaced): {len(review)} hits --")
    by_category: Dict[str, List[Tuple[str, int, Rule, str]]] = defaultdict(list)
    for item in review:
        by_category[item[2].category].append(item)
    for category in sorted(by_category):
        items = by_category[category]
        per_file = Counter(rel for rel, _, _, _ in items)
        print(f"  [{category}] {len(items)} hits in {len(per_file)} files")
        for rel, line, rule, hit in items[:cap]:
            print(f"    {rel}:{line}: {hit!r}")
        if len(items) > cap:
            print(f"    ... {len(items) - cap} more; per file: "
                  + ", ".join(f"{r} x{n}" for r, n in per_file.most_common()))
    print(f"  other e-mail addresses: {dict(other_emails.most_common(cap)) or 'none'}")
    print(f"  GitHub URLs: {dict(github_urls.most_common(cap)) or 'none'}")
    print(f"  Korean folder words outside paths (ordinary words, left unchanged): "
          f"{sum(korean_words.values())} in {len(korean_words)} files")
    if residual or binary_hits:
        verdict, status = "RESIDUAL IDENTIFIERS FOUND", 1
    elif review:
        verdict, status = f"NO KNOWN IDENTIFIER LEFT; {len(review)} ITEMS NEED A MANUAL DECISION", 3
    else:
        verdict, status = "CLEAN", 0
    print(f"\nverdict: {verdict}")
    return status


if __name__ == "__main__":
    sys.exit(main())
