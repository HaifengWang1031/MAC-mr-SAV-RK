# Issue tracker: Local Markdown

Issues and feature specs live as markdown files in `.scratch/`.
There is no external issue tracker. The confirmed project-wide specification remains `docs/spec.md`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`.
- Feature spec: `.scratch/<feature-slug>/spec.md`.
- Implementation tickets: `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`, one file per ticket.
- Triage state: a `Status:` line near the top, using the role strings in `docs/agents/triage-labels.md`.
- Comments append under a `## Comments` heading.

## When a skill says "publish to the issue tracker"

Create a file under `.scratch/<feature-slug>/`, creating the directory if needed.

## When a skill says "fetch the relevant ticket"

Read the referenced file. Resolve ticket numbers within the relevant feature directory.

## Wayfinding operations

Used by `/wayfinder`. The map is a file with one child file per ticket.

- Map: `.scratch/<effort>/map.md`, containing Notes / Decisions-so-far / Fog.
- Child ticket: `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`, with the question in the body. A `Type:` line records `research`, `prototype`, `grilling`, or `task`; a `Status:` line records `claimed` or `resolved`.
- Blocking: a `Blocked by: NN, NN` line near the top. A ticket is unblocked when every listed ticket is `resolved`.
- Frontier: scan for open, unblocked, unclaimed tickets; first by number wins.
- Claim: set `Status: claimed` and save before working.
- Resolve: append the answer under `## Answer`, set `Status: resolved`, then append a gist and link to the map's Decisions-so-far.

## Existing implementation review

The original implementation review uses the empty baseline commit `367ec0d`,
the confirmed scope in `docs/spec.md`, and coding rules in `AGENTS.md` and `CONTEXT.md`.
