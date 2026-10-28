# Triage Labels

Map canonical triage roles to local issue `Status:` values.

| Canonical role | Local status | Meaning |
| --- | --- | --- |
| `needs-triage` | `needs-triage` | Maintainer needs to evaluate |
| `needs-info` | `needs-info` | Waiting on reporter information |
| `ready-for-agent` | `ready-for-agent` | Fully specified, ready for an AFK agent |
| `ready-for-human` | `ready-for-human` | Requires human implementation |
| `wontfix` | `wontfix` | Will not be actioned |

When a skill names a triage role, use its local status above.
Wayfinder's `claimed` and `resolved` states are separate from these triage roles.
