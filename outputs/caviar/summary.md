# Caviar network demo: summary

_Network structure is real; all agent traits, actions and rewards are simulated and make no claim about real individuals._

Mode: **full**, seed 42.

## 1. Dataset

| stat | value |
|---|---|
| nodes | 110 |
| ties | 205 |
| mutual_pairs | 90 |
| group_sizes | 110 |
| groups | 1 |
| isolates | 0 |
| max_degree | 60 |
| highest_degree_agent | A001 (degree 60) |
| second_highest_degree_agent | A012 (degree 28) |
| average_degree | 3.727 |
| hub_share | 0.293 |
| tie_feature | Call volume |
| tie_feature_nonzero | 205 |
| tie_feature_mean | 0.305 |
| later_network | Phase 11 |
| later_ties | 50 |
| later_isolated_agents | 69 |
| calls_total | 3546 |
| hub_call_share | 0.577 |
| pairs_phase1 | 18 |
| top_agent_phase1 | A001 (degree 12) |
| pairs_phase2 | 28 |
| top_agent_phase2 | A001 (degree 19) |
| pairs_phase3 | 56 |
| top_agent_phase3 | A001 (degree 27) |
| pairs_phase4 | 48 |
| top_agent_phase4 | A001 (degree 23) |
| pairs_phase5 | 39 |
| top_agent_phase5 | A001 (degree 22) |
| pairs_phase6 | 47 |
| top_agent_phase6 | A001 (degree 18) |
| pairs_phase7 | 49 |
| top_agent_phase7 | A001 (degree 24) |
| pairs_phase8 | 58 |
| top_agent_phase8 | A001 (degree 20) |
| pairs_phase9 | 44 |
| top_agent_phase9 | A003 (degree 11) |
| pairs_phase10 | 50 |
| top_agent_phase10 | A001 (degree 13) |
| pairs_phase11 | 50 |
| top_agent_phase11 | A012 (degree 12) |
| later_group_sizes | 39, 2 |

## 2. True vs recovered parameters

| parameter | true | recovered | error |
|---|---|---|---|
| peer_approval | 0.400 | 0.399 | -0.001 |
| material_gain | 0.300 | 0.299 | -0.001 |
| status_gain | 0.300 | 0.304 | 0.004 |
| sanction_risk | -0.600 | -0.604 | -0.004 |
| gamma_status | 4.000 | 4.006 | 0.006 |
| gamma_tie | 2.000 | 2.207 | 0.207 |

Cosine 1.0000; all signs correct: True. gamma_tie (call volume) is a continuous weight with mean 0.31 and spread 0.20 across the 205 ties, which gives the learner variation to identify it.

## 3. Data efficiency

| episodes | decisions | cosine | gamma_status_hat | gamma_tie_hat |
|---|---|---|---|---|
| 1 | 5500 | 0.996 | 5.026 | 3.783 |
| 3 | 16500 | 0.999 | 3.799 | 2.556 |
| 10 | 55000 | 1.000 | 3.792 | 2.505 |
| 50 | 275000 | 1.000 | 3.945 | 2.346 |
| 160 | 880000 | 1.000 | 4.006 | 2.207 |

## 4. Held-out accuracy

| model | action accuracy | good/bad accuracy |
|---|---|---|
| Behavioural cloning | 0.442 | 0.712 |
| Recovered reward | 0.481 | 0.735 |
| Majority class | 0.356 | nan |

Accuracy is capped by the agents' own randomness: they sample actions from a softmax, so even the true reward cannot predict every choice.

## 5. Explanations (held-out decisions, agents with 2 or more neighbours)

| method | top1 | top3 |
|---|---|---|
| Recovered attention | 0.999 | 1.000 |
| Most similar neighbour | 0.682 | 0.906 |
| Random neighbour (expected) | 0.297 | 0.764 |

Spearman correlation between attributed tie importance and |ablation effect|: 0.649 over 60 ties. Surrogate decision tree fidelity: 0.68.

## 6. Sensitivity

| setting | final_bad_share | ci_low | ci_high |
|---|---|---|---|
| true theta | 0.723 | 0.680 | 0.765 |
| sanction weight doubled | 0.284 | 0.248 | 0.320 |

## 7. Cascades

| policy | mean_abs_gap |
|---|---|
| bc | 0.018 |
| recovered | 0.003 |

| policy | mean_first_escalation_step | share_of_agent_runs_escalating |
|---|---|---|
| bc | 3.93 | 1.00 |
| recovered | 3.90 | 1.00 |
| true | 3.91 | 1.00 |

## 8. Interventions

| scenario | baseline_bad_share | scenario_bad_share | delta | ci_low | ci_high |
|---|---|---|---|---|---|
| Hub isolation (A001) | 0.691 | 0.684 | -0.007 | -0.038 | 0.025 |
| Mentor injection (A012) | 0.691 | 0.587 | -0.104 | -0.116 | -0.092 |
| Supervision +0.2 | 0.691 | 0.478 | -0.213 | -0.232 | -0.193 |
| Real disruption (phase 11 graph) | 0.691 | 0.581 | -0.110 | -0.141 | -0.079 |

## 9. Insights

1. Behaviour alone was enough to recover the hidden motives. From 880,000 observed decisions the learner matched the true reward weights with cosine 1.0000 (largest error 0.004, on status gain). With a single episode (5,500 decisions) the cosine was already 0.996.
2. Who an agent listens to matters as much as what they want. Recovered attention named the true influencer in 100% of held-out decisions, against 68% for the most similar neighbour and 30% for a random one.
3. Supervision +0.2 cut the final bad share most, by 0.213 (from 0.69), and its 95% interval excludes zero. The real phase 11 network changed it by -0.110. Removing any single tie moved it by at most 0.004, so in this model behaviour is driven by incentives and supervision more than by any one relationship.
4. A model of motives generalises better than a copy of actions: rolled forward, the recovered reward stays closer to the true cascade than behavioural cloning (mean gap 0.003 vs 0.018). One-step accuracy alone hides this: 0.48 for the recovered reward vs 0.44 for cloning.

## 10. Acceptance criteria (SPEC.md section 11)

| check | passed | value |
|---|---|---|
| Sensitivity: doubling the sanction weight lowers final bad share by at least 0.20 | pass | 0.439 |
| Action variety: every action is at least 5% of decisions | pass | 15.1%, 20.3%, 28.0%, 36.5% |
| Recovery (full data): cosine at least 0.98 | pass | 1.0000 |
| Recovery (full data): every theta within 0.05 of truth | pass | 0.004 |
| Recovery (full data): all four signs correct | pass | True |
| Recovery (1 episode): cosine at least 0.90 | pass | 0.9958 |
| gamma_status within 1.0 of truth | pass | 4.01 (true 4.0) |
| Explanations: Top-1 at least 0.95 | pass | 0.999 |
| Explanations: Top-1 above the most-similar baseline | pass | 0.999 vs 0.682 |
| BC: held-out accuracy above the majority class | pass | 0.442 vs 0.356 |
| Cascade: recovered-policy gap below the BC gap | pass | 0.0026 vs 0.0177 |

All criteria pass on this network.

## Notes on the data

The main network is the `full` call layer; a pair's weight is the calls in both directions added together, and the tie feature is log(1 + calls) scaled to 0 to 1. The later network is phase 11 on the same 110 agents; agents without calls in phase 11 are isolated in it. Ablation covers the 60 ties with the highest edge betweenness, because the network has more than 60 ties.
