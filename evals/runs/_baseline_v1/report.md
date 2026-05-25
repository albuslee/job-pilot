# JobPilot Eval — 2026-05-25T07:38:51Z

**Config:** model=`sonnet-4.6`, prompt_version=`v1`, n_cases=7

## Summary
| metric | value |
| --- | --- |
| decision_accuracy | 6/7 (85.7%) |
| score_mae | 0.0 (n=2) |
| score_in_band_rate | 2/2 (100.0%) |
| citation_evidence_pass_rate | n/a |
| required_risk_flags_pass_rate | n/a |
| disallowed_risk_flags_pass_rate | n/a |
| p50 latency / p95 | 20.2s / 144.7s |
| total tokens (in / out) | 38,312 / 5,604 |
| total cost | $0.199 |
| mean cost / known case | $0.028 |
| errors | 0 |

## Failures (1)
### anthropic_research_eng
- expected_decision=maybe, got=skip ✗
- reasoning excerpt: "**JD Must-Haves vs. Profile:**"

## Errors (0)
*(none)*
