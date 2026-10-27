.DEFAULT_GOAL := pdf
PYTHON ?= python
.PHONY: figures pdf clean supplement supplementary-figure final

figures:
	$(PYTHON) make_figures.py

supplementary-figure:
	$(PYTHON) analysis/make_supp_figure.py --archive ../data/TxtLangFiles1870-1894_TB.zip --outdir .

pdf: figures supplementary-figure
	pdflatex -interaction=nonstopmode -halt-on-error report.tex
	bibtex8 report
	pdflatex -interaction=nonstopmode -halt-on-error report.tex
	pdflatex -interaction=nonstopmode -halt-on-error report.tex

clean:
	rm -f report.aux report.bbl report.blg report.log report.out report.synctex.gz report.toc supplement_preview.aux supplement_preview.log supplement_preview.out

supplement: supplementary-figure
	pdflatex -interaction=nonstopmode -halt-on-error supplement_preview.tex
	pdflatex -interaction=nonstopmode -halt-on-error supplement_preview.tex

# Private submission assembly; NEVER commit the signed Faculty declaration.
# Use: make final DECLARATION=/absolute/path/to/signed_declaration.pdf
final: pdf
	@test -f "$(DECLARATION)" || { echo "Missing DECLARATION=path/to/signed_declaration.pdf"; exit 1; }
	@command -v pdfunite >/dev/null || { echo "Install poppler-utils (pdfunite)"; exit 1; }
	pdfunite report.pdf "$(DECLARATION)" Project2_Submission_with_Declaration.pdf
