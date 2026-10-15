# Dataset and Evaluation Disclosure

## 1. Record Provenance

This dataset contains a mixture of original-source-derived records, historically
generated or rewritten records, and records whose provenance has not been fully
verified. It should not be described as a collection consisting exclusively of
unaltered, naturally observed social-media interactions. An unresolved record is
not thereby established to be either original or generated; the absence of an
explicit generation marker is not evidence of original-source provenance.

Analyses and descriptions of this dataset should acknowledge this mixed and
partially unresolved provenance. No claim that all records are original
observations, or that their original/generated proportions are fully established,
is made by this release.

## 2. Interpretation of Micro-Level Semantic Metrics

In the released evaluation workflow, participants are sampled with replacement
from an event-level reference pool. Each generated response is paired with the
source text associated with the sampled reference record. Semantic metrics compare
the labels of these reference texts with those of the generated responses.

For stepwise micro-level aggregation, both sides are grouped by the simulated
`time_step`, rather than regrouping reference texts by their original `source_t`.
The reference distribution at a simulated step therefore reflects the records
sampled for that step; it is not necessarily the original observed distribution
for the corresponding day.

Accordingly, these micro-level metrics measure consistency with sampled reference
texts under this matching protocol. They should not be interpreted as strict
predictions of the original real-world daily semantic trajectory, or as evidence
that the same user's response on the same day was predicted. This qualification
concerns the micro-level semantic matching protocol; it does not redefine the
separately computed macro-level targets.
