.PHONY: audit test plan reproduce

DATA ?=
RUN_DIR ?= ../isixhosa-fresh-run

audit:
	python scripts/factory.py audit

test:
	python -m unittest discover -s tests -v
	python scripts/factory.py audit

plan:
	python scripts/factory.py full --archive "$(DATA)" --workdir "$(RUN_DIR)" --dry-run

reproduce:
	python scripts/factory.py full --archive "$(DATA)" --workdir "$(RUN_DIR)"
