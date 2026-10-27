.DEFAULT_GOAL := pdf
PYTHON ?= python
.PHONY: figures pdf clean

figures:
	$(PYTHON) make_figures.py

pdf: figures
	cp ../reports/paper_draft/ccn.sty ./ccn.sty
	cp ../reports/paper_draft/apacite.bst ./apacite.bst
	pdflatex -interaction=nonstopmode -halt-on-error report.tex
	bibtex8 report
	pdflatex -interaction=nonstopmode -halt-on-error report.tex
	pdflatex -interaction=nonstopmode -halt-on-error report.tex

clean:
	rm -f report.aux report.bbl report.blg report.log report.out report.synctex.gz report.toc ccn.sty apacite.bst
