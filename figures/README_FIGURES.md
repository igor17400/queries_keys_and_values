# Figures

TikZ source files live in `src/`. Compiled SVGs go to `build/`.

Both `src/` (TikZ sources) and `build/` (compiled SVGs) are tracked by git.

## Build a single figure

```bash
cd figures/src
pdflatex -interaction=nonstopmode FILENAME.tex
pdf2svg FILENAME.pdf ../build/FILENAME.svg
```

## Build all figures

```bash
cd figures/src
for f in *.tex; do
  pdflatex -interaction=nonstopmode "$f" && pdf2svg "${f%.tex}.pdf" "../build/${f%.tex}.svg"
done
```

## Requirements

- `pdflatex` (TeX Live)
- `pdf2svg` (`brew install pdf2svg`)
