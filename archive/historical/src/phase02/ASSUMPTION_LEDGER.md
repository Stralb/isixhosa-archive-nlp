# Phase 2 assumption ledger

- The **full finite page inventory and source/page/issue metadata** are assumed known. This is not the same as possessing undigitized missing issues. If number of pages/documents is unknown, our worst-case bounds do not apply directly.
- The outcome is a *clipped normalized lexical word-form count per page*. Its [0,.02] bound is true by definition; it is not a validated measure of word meaning.
- MAR source/page and length/date masks are synthetic, not claims about real nineteenth-century preservation.
- MNAR scenario conditions on full-text lexical statistics unavailable to analysts missing those pages. Neither propensity fitting nor adjustment can recover them without further assumptions.
- Known propensity weights are privileged controls requiring the simulated selection mechanism. Under nonpositivity there is no such unbiased weighting estimate.
- Manski bounds use exactly known observed outcomes and complete frame; they are identification sets, **not 95% intervals** or power guarantees.
- Pattern-mixture gamma bounds cover the finite population only when the gamma premise holds. Simulations evaluate that premise with privileged oracle data.
- Page selection is sometimes correlated by *issue*. Phase 2 measures error over repeated masks; no IID-token statistical p-values or overconfident per-page t-tests are permitted. Formal cluster sampling uncertainty comparison is reserved for Phase 3.
- 1885 page/feature pool was already inspected in earlier work; DEV/TEST are simulation-design partitions only and not fresh archival validation. Independent datasets and historical expertise are indispensable for semantic claims.
