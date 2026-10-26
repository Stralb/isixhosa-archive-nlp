# Start here: Excellence Sprint Phase 1 (research continuation after Phase 33)

**Scientific status: Phase 1 scientific freeze completed (subject to verification JSON PASS).**

Research program: `../ROADMAP.md`. Frozen factorial study: `FROZEN_PROTOCOL.md`. Risks and decisions: `DECISION_LOG.md`. Independent audit: `outputs/feasibility_audit.json` and `outputs/verification.json`.

Previous phases through Phase 33 remain unchanged; do not rewrite their original hypothesis classifications. New Phase 2 begins with **validation-only** tokenizer/LM implementation. **Do not view or tune against any new 1888-1894 performance before the Phase 2 selection protocol is locked.**

## Reproduce Phase 1 from source archive

```bash
python scripts/audit_phase1.py --archive /path/to/TxtLangFiles1870-1894_TB.zip --phase33 /path/to/isiXhosa_phase33_sampling_to_models --out outputs
python scripts/smoke_phase1.py --archive /path/to/TxtLangFiles1870-1894_TB.zip --out outputs
python /path/to/isiXhosa_phase33_sampling_to_models/scripts/verify_phase33.py --zip /path/to/TxtLangFiles1870-1894_TB.zip --root /path/to/isiXhosa_phase33_sampling_to_models > outputs/legacy_phase33_reverification.json
python scripts/verify_phase1.py --archive /path/to/TxtLangFiles1870-1894_TB.zip --phase33 /path/to/isiXhosa_phase33_sampling_to_models --root .
```

The **new factorial feasibility and sampler routines** read only training (1885 + 1887 Jan-Sep) and validation (1887 Oct-Dec) textual content; 1888-1894 are inventoried by filename only. The *separate baseline Phase 33 re-verifier* necessarily rereads previously scored 1888-1894 text to recheck old results. No **new factorial** 1888-1894 outcome is computed during Phase 1. Saved training samples are hashes and counts, not redistributions of the historical text.

## Scientific caution

Neither the phase count nor an 80-100% rubric aspiration is itself an evidence claim. The strongest output is the narrow, tested explanation that historical layout and sampling control exposure and tokenization, not a demonstrated change in cultural beliefs or a reliable gain in downstream LM accuracy.

Simplest use: `sh RUN_PHASE1.sh /path/to/TxtLangFiles1870-1894_TB.zip /path/to/isiXhosa_phase33_sampling_to_models`. Install `requirements.txt` if needed.
