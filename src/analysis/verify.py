from __future__ import annotations

import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def fail(msg: str) -> None:
    print(f"VERIFY FAIL: {msg}")
    sys.exit(1)


def strip_latex_to_plaintext(tex: str) -> str:
    """Extract narrative prose only, stripping all LaTeX machinery."""
    tex = re.sub(r"%.*", "", tex)

    m = re.search(r"\\begin\{document\}(.*)\\end\{document\}", tex, flags=re.S)
    if m:
        tex = m.group(1)

    tex = re.sub(r"\\(input|include|includegraphics)\{[^}]*\}", "", tex)
    tex = re.sub(r"\\(begin|end)\{[^}]*\}(\[[^\]]*\])?", "", tex)
    tex = re.sub(r"\\(label|ref|caption)\{[^}]*\}", "", tex)
    tex = re.sub(r"\\[a-zA-Z@]+(\[[^\]]*\])?(\{[^}]*\})*", "", tex)
    tex = tex.replace("{", "").replace("}", "")
    tex = re.sub(r"\$[^$]*\$", "", tex)  # inline math (macros like $\NRows$)
    tex = re.sub(r"\s+", " ", tex).strip()
    return tex


def main() -> None:
    if not (ROOT / "artifacts" / "data_catalog.json").exists():
        fail("Missing artifacts/data_catalog.json (run make profile)")

    if not (ROOT / "paper" / "results.tex").exists():
        fail("Missing paper/results.tex (run make profile / make analysis)")

    tables = list((ROOT / "paper" / "tables").glob("*.tex"))
    if not tables:
        fail("No generated tables found in paper/tables/ (run make analysis)")

    paper_tex = ROOT / "paper" / "paper.tex"
    if not paper_tex.exists():
        fail("Missing paper/paper.tex")

    plain = strip_latex_to_plaintext(paper_tex.read_text(encoding="utf-8"))

    if re.search(r"\d", plain):
        for chunk in plain.split("."):
            if re.search(r"\d", chunk):
                fail(
                    f"Digit in narrative prose: ...{chunk.strip()[:120]}...\n"
                    "Move numbers to results.tex macros or generated tables."
                )

    if not (ROOT / "artifacts" / "findings.md").exists():
        fail("Missing artifacts/findings.md (run make analysis)")

    print("VERIFY OK")


if __name__ == "__main__":
    main()
