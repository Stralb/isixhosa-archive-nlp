# Source data and research ethics

The study uses the language-split text archive supplied for the 2026 NLP course, associated with the isiXhosa Intellectual Traditions (IsiXIT) project. The original publication documents the digitisation context and the collection's limits:

Schoots, J., Ngwendu, A., de Wet, J., and Muftic, S. (2023). *IsiXhosa Intellectual Traditions Digital Archive: Digitizing isiXhosa texts from 1870-1914*. [ACL Anthology](https://aclanthology.org/2023.rail-1.7/).

Archive landing page: https://ibali.uct.ac.za/s/isixit/page/welcome

Original course ZIP: `TxtLangFiles1870-1894_TB.zip`

SHA256:

```text
375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5
```

**No original page transcriptions are included in this public repository.** Redistribution rights for the supplied archive have not been independently verified. Scripts accept a locally supplied dataset path. Reproducing numerical results may require the precise dataset version used in the course.

## Respecting what the archive measures

The `EnTxt` and `XhTxt` folders are entire text blocks assigned to archive language views, not validated per-word language labels. Neither newspaper is representative of every isiXhosa community. We do not infer writers' identities, political positions, sentiments, historical truth or experienced social harm from segmentation costs.

A change in model representation is an observable property of the technical pipeline. It is not, without additional evidence, proof of social unfairness. We report a negative downstream result when the full policy change failed to improve later-year BPC.

## Human authorship and research conduct

Course participants remain responsible for verifying the results, assessing copyright and research ethics, and accurately declaring their own contributions and any AI-assisted work under the university's rules. This repository does not invent a contributor list or fill ethics forms on anyone's behalf. The final approved group declaration is separate from the reusable research source files.

## Research use

No software licence has been asserted. Public accessibility does not automatically grant reuse rights. The maintainers should choose a code licence once every contributor with relevant rights agrees.