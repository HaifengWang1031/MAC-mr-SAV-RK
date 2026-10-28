# Domain Docs

## Layout

Single-context: `GLOSSARY.md` at the repository root and ADRs under `docs/adr/`.

## Before exploring

Read `GLOSSARY.md` and ADRs relevant to the area you are exploring.

If these files do not exist, proceed silently. `/domain-modeling` creates them
lazily when terms or decisions are resolved.

## Vocabulary

Use glossary terms in issue titles, proposals, hypotheses, and test names.
If a needed concept is absent, reconsider the term or note the gap for `/domain-modeling`.

## ADR conflicts

Explicitly flag proposals that contradict an existing ADR, identifying the ADR
and explaining why the decision is worth reopening.
