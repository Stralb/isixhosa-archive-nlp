.DEFAULT_GOAL := pdf
PYTHON ?= python
.PHONY: figures pdf clean supplement supplementary-figure

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
