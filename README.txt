NLP Project 2 - Historical isiXhosa Newspaper NLP

Requirements: Python 3.13 and the included newspaper archive.
Run all commands from the repository root unless stated otherwise.

1. Install dependencies:
   python -m pip install -r requirements.txt

2. Check the saved experiment results:
   python -m unittest discover -s src -p 'test_*.py'
   python src/run.py check
   python src/audit/independent_ci.py --root .

3. Reproduce the tokenizer and bigram experiments (re-trains 60 models):
   python src/run.py reproduce --data data/TxtLangFiles1870-1894_TB.zip --out ../isixhosa-replay

4. Recompute and verify the ArchiveLens results:
   python src/archivelens/audit.py --data data/TxtLangFiles1870-1894_TB.zip --out results/archivelens --figures results/figures --report reports/ARCHIVELENS_RESULTS_2026-10-26.md --verify
   python src/archivelens/followup.py --data data/TxtLangFiles1870-1894_TB.zip --root . --verify

5. Build the paper (requires pdflatex, bibtex8 and the CCN LaTeX packages):
   cd paper
   make pdf
   make supplement  # separate three-page preview; ethics questionnaire is not yet included

Data file: data/TxtLangFiles1870-1894_TB.zip
SHA-256: 375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5
To check: sha256sum data/TxtLangFiles1870-1894_TB.zip

Main code: src/run.py
ArchiveLens code: src/archivelens/
Model results: results/
Research methods and findings: reports/
Paper source: paper/report.tex
Supplementary source: paper/supplementary.tex

