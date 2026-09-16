"""MEMORIA — Inserta las figuras en el borrador y lo convierte a Word.

Sustituye cada marcador

    > **[FIGURA 3]** `fig05_dir_acc_vs_base.png` — Pie de figura.

por la imagen real con su pie, y genera `docs/memoria.docx` mediante Pandoc.

    python -m src.reporting.build_memoria

Requiere Pandoc:  winget install --id JohnMacFarlane.Pandoc -e
"""
from __future__ import annotations
import re
import shutil
import subprocess
import sys
from src.config import FIGURES, ROOT

SRC = ROOT / "docs" / "10_TFM_DRAFT.md"
BUILD = ROOT / "docs" / "memoria_build.md"
OUT = ROOT / "docs" / "memoria.docx"

# > **[FIGURA 1]** `fichero.png` — Pie de figura.
PATTERN = re.compile(
    r"^>\s*\*\*\[FIGURA\s+(\d+)\]\*\*\s*`([^`]+)`\s*[—-]\s*(.+?)\s*$",
    re.MULTILINE)

def main() -> None:
    if not SRC.exists():
        sys.exit(f"No se encuentra {SRC}")
    text = SRC.read_text(encoding="utf-8")

    faltan, n = [], 0

    def repl(m: re.Match) -> str:
        nonlocal n
        num, fname, caption = m.group(1), m.group(2), m.group(3).rstrip(".")
        if not (FIGURES / fname).exists():
            faltan.append(fname)
            return f"*[FALTA LA FIGURA {num}: {fname}]*"
        n += 1
        # Pandoc usa el texto alternativo como pie cuando la imagen va sola.
        return f"![Figura {num}. {caption}.](outputs/figures/{fname})"

    text = PATTERN.sub(repl, text)

    # Elimina la nota de trabajo inicial
    text = re.sub(r"^> \*Nota de trabajo.*?\n(?:>.*\n)*", "", text, flags=re.MULTILINE)

    BUILD.write_text(text, encoding="utf-8")
    print(f"{n} figuras insertadas → {BUILD.name}")
    if faltan:
        print(f"⚠ Figuras no encontradas: {', '.join(faltan)}")

    palabras = len(re.findall(r"\b\w+\b", text))
    print(f"~{palabras} palabras → estimación: {palabras / 450:.0f}-{palabras / 350:.0f} caras"
          f" más 13 figuras (≈ media cara cada una)")

    if shutil.which("pandoc") is None:
        print("\nPandoc no está instalado. Instálalo con:")
        print("  winget install --id JohnMacFarlane.Pandoc -e")
        return

    cmd = ["pandoc", str(BUILD), "-o", str(OUT),
           "--resource-path", str(ROOT), "--toc", "--toc-depth=2"]
    ref = ROOT / "docs" / "reference.docx"
    if ref.exists():
        cmd += [f"--reference-doc={ref}"]
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"\n {OUT}")
    print("   Ábrelo en Word, Ctrl+E para seleccionar todo, fuente Arial 11,")
    print("   y comprueba el número de caras sin contar portada ni índice.")

if __name__ == "__main__":
    main()
