#!/usr/bin/env python3
"""Populate all fifteen questions in the exact user-supplied 2024 NeurIPS template.

Preserves the official question wording, instructions and guidelines.
CAUTION: '[REVIEW]' flags are deliberate. Human authors must verify all answers.
"""
from __future__ import annotations
import hashlib
import re
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parent
ZIP=ROOT/"original_templates"/"NeurIPS_2024_Checklist_Original.zip"
OUT=ROOT/"neurips_checklist_filled.tex"
EXPECTED_SHA="9e21a868b448b31f978e90c21e8aa2b61e766b11b94fc907cae13bf0c9871e66"

# Ordered EXACTLY as the fifteen NeurIPS 2024 template questions.
# The answers marked TODO and their justifications REQUIRE human authorship/sign-off.
ANSWERS=[
 ("Yes","The two complementary analyses and their limitations are stated in the scientific abstract and Introduction."),
 ("Yes","The paper discloses historical-year reuse, archive-view proxy labels, source confounding, repetition and no semantic or human fairness evaluation."),
 ("NA","The paper reports empirical investigations and no new theorem requiring a formal proof."),
 ("Yes","The repository includes code, fixed seeds, data hashes, frozen protocols and independent verification."),
 ("TODO","Author review required: project source and data are available, but confirm original transcription redistribution rights and final availability before answering."),
 ("Yes","The Methods and linked protocols state sampling arms, tokenizer settings, character budgets, dates, clustering and outcomes."),
 ("Yes","The manuscript reports issue bootstrap intervals, permutation calibration, and their respective inferential caveats."),
 ("No","Detailed CPU, memory and wall-clock totals for the entire exploratory research history have not been reconstructed; project job logs give partial evidence."),
 ("TODO","Author review required: confirm NeurIPS ethics compliance, data rights, attribution and responsible release before making this declaration."),
 ("Yes","The Discussion addresses the benefits of transparent warnings and risks of misinterpreting surface statistics as cultural conclusions."),
 ("TODO","Author review required: assess safeguards and archival rights for original texts already present in the public repository."),
 ("TODO","Author review required: data providers are attributed, but the archive transcription distribution licence or permission must be established."),
 ("Yes","ArchiveLens source, protocols, caveats, tests, independent verifiers and example outputs are included in the repository."),
 ("NA","No crowdsourcing, recruited participants or human annotation experiment was conducted."),
 ("NA","No living human participants were recruited; the authors should check institutional requirements for research on archival texts.")
]

def main() -> None:
    assert hashlib.sha256(ZIP.read_bytes()).hexdigest()==EXPECTED_SHA, "Original checklist ZIP differs from uploaded archive"
    with ZipFile(ZIP) as archive:
        src=archive.read("NeurIPS_2024_checklist/neurips_2024.tex").decode("utf-8")
    begin=src.index(r"\section*{NeurIPS Paper Checklist}")
    end=src.rindex(r"\end{document}")
    checklist=src[begin:end]
    replacements=iter(ANSWERS)
    def change_answer(match:re.Match[str]) -> str:
        state,_=next(replacements)
        return match.group(1)+r"\answer"+state+r"{}"
    pattern=r"(?m)^(\s*\\item\[\] Answer:\s*)\\answerTODO\{\}"
    checklist,n=re.subn(pattern,change_answer,checklist)
    assert n==len(ANSWERS)==15,(n,len(ANSWERS))
    for state,just in ANSWERS:
        # Escape literal TeX special characters not used in any intended markup.
        assert not any(x in just for x in ["\\","%","$","#","&","{","}"])
    it=iter(ANSWERS)
    def change_justification(match:re.Match[str]) -> str:
        return match.group(1)+next(it)[1]
    checklist,j=re.subn(r"(?m)^(\s*\\item\[\] Justification:\s*)\\justificationTODO\{\}",change_justification,checklist)
    assert j==15,(j,"missing justification placeholders")
    assert checklist.count(r"\item {\\bf")==0 # TeX source should use a single slash
    assert "NeurIPS Paper Checklist" in checklist
    assert not re.search(r"\\item\[\] Justification:\s*\\justificationTODO\{\}",checklist)
    OUT.write_text("% Generated from unmodified official NeurIPS 2024 source ZIP. Review all answers.\n"+
                   checklist.rstrip()+"\n",encoding="utf-8")
    print("PASS: generated all 15 official checklist items and full guidance; author review flags retained")

if __name__=="__main__":
    main()
