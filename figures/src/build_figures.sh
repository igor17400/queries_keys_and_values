#!/bin/bash
# Build all TikZ figures.
# Each folder in src/ contains a .tex file.
# Output: build/<folder_name>.svg

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR"
BUILD_DIR="$SCRIPT_DIR/../build"

mkdir -p "$BUILD_DIR"

for folder in "$SRC_DIR"/*/; do
    name="$(basename "$folder")"
    tex_file="$(find "$folder" -maxdepth 1 -name '*.tex' | head -1)"

    if [ -z "$tex_file" ]; then
        echo "SKIP: $name (no .tex file found)"
        continue
    fi

    echo "BUILD: $name"
    (cd "$folder" && pdflatex -interaction=nonstopmode "$(basename "$tex_file")" > /dev/null 2>&1)

    pdf_file="$folder/${name}.pdf"
    if [ ! -f "$pdf_file" ]; then
        # try any pdf in the folder
        pdf_file="$(find "$folder" -maxdepth 1 -name '*.pdf' | head -1)"
    fi

    if [ -f "$pdf_file" ]; then
        pdf2svg "$pdf_file" "$BUILD_DIR/${name}.svg"
        echo "  OK: build/${name}.svg"
    else
        echo "  FAIL: no PDF generated for $name"
    fi
done
