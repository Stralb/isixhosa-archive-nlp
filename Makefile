.PHONY: test source-check plan
test:
	python -m unittest discover -s factory/tests -v
	python scripts/check_frozen_source.py

source-check:
	python scripts/check_frozen_source.py

plan:
	python scripts/reproduce_all.py --archive /path/to/TxtLangFiles1870-1894_TB.zip --workdir /tmp/isiXhosa-repro --dry-run
