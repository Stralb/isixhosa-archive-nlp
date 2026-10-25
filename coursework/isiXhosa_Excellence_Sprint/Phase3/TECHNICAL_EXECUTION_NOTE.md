# Phase 3 technical execution note (pre-decision)

The first evaluation scored all 40 frozen models on the fixed held-out
page-view collection and computed issue-bootstrap contrasts. The export then
raised `ValueError` because later-year records include the additional
`year_cluster_ci_low/high` fields absent from the 1888 CSV header.

The correction is only to write the union of row keys when exporting contrasts.
All scoring, bootstrap seeds, model files, input texts, hypotheses, and
thresholds are unchanged. No hypothesis decision or new tuning followed the
failed export. A complete fresh run is required after this correction.
