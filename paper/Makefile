.DEFAULT_GOAL := pdf
PYTHON ?= python
.PHONY: figures pdf clean supplement

figures:
	$(PYTHON) make_figures.py

pdf: figures
	pdflatex -interaction=nonstopmode -halt-on-error report.tex
	bibtex8 report
	pdflatex -interaction=nonstopmode -halt-on-error report.tex
	pdflatex -interaction=nonstopmode -halt-on-error report.tex

clean:
	rm -f report.aux report.bbl report.blg report.log report.out report.synctex.gz report.toc

supplement:
	$(PYTHON) analysis/make_supp_figure.py --archive ../data/TxtLangFiles1870-1894_TB.zip --outdir .
	pdflatex -interaction=nonstopmode -halt-on-error supplement_preview.tex
	pdflatex -interaction=nonstopmode -halt-on-error supplement_preview.tex
