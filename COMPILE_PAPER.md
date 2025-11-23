# How to Compile the LaTeX Paper to PDF

The paper is located at `outputs/full_paper.tex`. To generate a PDF, you need a LaTeX distribution installed.

## Option 1: Install MiKTeX (Recommended for Windows)

1. **Download MiKTeX:**
   - Visit: https://miktex.org/download
   - Download the Windows installer
   - Run the installer and follow the setup wizard

2. **Compile the paper:**
   ```powershell
   cd outputs
   pdflatex -interaction=nonstopmode full_paper.tex
   pdflatex -interaction=nonstopmode full_paper.tex  # Run twice for references
   ```

3. **Output:**
   - The PDF will be generated as `outputs/full_paper.pdf`

## Option 2: Install TeX Live

1. **Download TeX Live:**
   - Visit: https://www.tug.org/texlive/windows.html
   - Follow installation instructions

2. **Compile the paper:**
   ```powershell
   cd outputs
   pdflatex -interaction=nonstopmode full_paper.tex
   pdflatex -interaction=nonstopmode full_paper.tex  # Run twice for references
   ```

## Option 3: Use Online LaTeX Compiler

If you don't want to install LaTeX locally:

1. **Overleaf (Recommended):**
   - Visit: https://www.overleaf.com
   - Create a free account
   - Create a new project
   - Copy the contents of `outputs/full_paper.tex` into the editor
   - Click "Recompile" to generate the PDF
   - Download the PDF

2. **ShareLaTeX:**
   - Similar to Overleaf
   - Visit: https://www.sharelatex.com

## Option 4: Use Docker (If Docker is installed)

```powershell
docker run --rm -v ${PWD}/outputs:/workdir texlive/texlive:latest pdflatex -interaction=nonstopmode full_paper.tex
```

## Troubleshooting

**Missing packages:**
- If compilation fails due to missing packages, MiKTeX will prompt to install them automatically
- Or manually install: `miktex install <package-name>`

**Bibliography issues:**
- If you have citations, you may also need to run:
  ```powershell
  bibtex full_paper
  pdflatex full_paper.tex
  pdflatex full_paper.tex
  ```

**Figure references:**
- Make sure any referenced figures exist in the `outputs/` directory
- The paper references: `figure_1_pipeline_schematic.png`, `figure_2_detection_sensitivity.png`, etc.

## Quick Start (After Installing MiKTeX)

```powershell
# Navigate to outputs directory
cd outputs

# Compile (run twice for proper cross-references)
pdflatex -interaction=nonstopmode full_paper.tex
pdflatex -interaction=nonstopmode full_paper.tex

# Check if PDF was created
Test-Path full_paper.pdf
```

The PDF will be saved as `outputs/full_paper.pdf`.

