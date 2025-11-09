from __future__ import annotations
import os, pathlib, sys, datetime

REPO_USER = "NatanIsaacRuskin"
REPO_NAME = "IKF_Market_Pipeline"
BRANCH = os.getenv("CATALOG_BRANCH", "main")
RAW_BASE = f"https://raw.githubusercontent.com/{REPO_USER}/{REPO_NAME}/{BRANCH}/"

IGNORE_DIRS = {".git", ".github", "__pycache__", ".venv", "backups", "data/cache"}
ALLOWED_EXT = {".py", ".ipynb", ".md", ".yaml", ".yml", ".toml", ".json", ".csv", ".txt"}

def should_include(rel: pathlib.Path) -> bool:
    if any(part in IGNORE_DIRS for part in rel.parts):
        return False
    if rel.name.startswith("."):  # hide dotfiles except root configs if you want
        return False
    if rel.suffix.lower() in ALLOWED_EXT:
        return True
    return False

def human_size(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.0f} PB"

def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    files = []
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if p.is_file() and should_include(rel):
            files.append(p)
    files.sort(key=lambda x: str(x).lower())

    lines = []
    lines.append("# IKF Market Pipeline – Code Catalog\n")
    lines.append(f"_Branch: **{BRANCH}** | Files: **{len(files)}**_\n")
    lines.append("## Index\n")
    for p in files:
        rel = p.relative_to(root).as_posix()
        stat = p.stat()
        mtime = datetime.datetime.utcfromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%SZ")
        size = human_size(stat.st_size)
        raw = RAW_BASE + rel
        lines.append(f"- `{rel}`  ({size}, modified {mtime} UTC) → [raw]({raw})")

    lines.append("\n---\n")
    lines.append("## File Previews\n")
    for p in files:
        rel = p.relative_to(root).as_posix()
        raw = RAW_BASE + rel
        ext = p.suffix.lower().lstrip(".")
        lang = {"yml":"yaml"}.get(ext, ext)  # nicer fencing
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            text = f"<<could not read: {e}>>"

        # Keep the catalog small but useful (preview only)
        snippet = text if len(text) <= 4000 else text[:4000] + "\n...\n[truncated]"

        lines.append(f"\n### `{rel}`  •  [raw]({raw})\n")
        lines.append(f"```{lang}\n{snippet}\n```")

    (root / "CATALOG.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote CATALOG.md")

if __name__ == "__main__":
    sys.exit(main())
