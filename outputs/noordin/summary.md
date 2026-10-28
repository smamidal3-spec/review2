# Noordin network demo: summary

_Network structure is real; all agent traits, actions and rewards are simulated and make no claim about real individuals._

Mode: **full**, seed 42.

## 1. Dataset

| stat | value |
|---|---|
| nodes | 28 |
| ties | 38 |
| mutual_pairs | 29 |
| group_sizes | 15, 8, 5 |
| groups | 3 |
| isolates | 0 |
| max_degree | 10 |
| highest_degree_agent | A11 (degree 10) |
| second_highest_degree_agent | A07 (degree 6) |
| average_degree | 2.714 |
| hub_share | 0.263 |
| tie_feature | Kinship |
| tie_feature_nonzero | 34 |
| tie_feature_mean | 0.895 |
| later_network | Post-attack |
| later_ties | 13 |
| later_isolated_agents | 13 |
| pre_attack_ties | 38 |
| kinship_ties | 34 |
| kinship_ties_in_pre | 34 |
| post_attack_ties | 13 |
| post_ties_in_pre | 13 |
| pre_attack_group_sizes | 15, 8, 5 |
| post_attack_isolated_agents | 13 |

## 2. True vs recovered parameters

| parameter | true | recovered | error |
|---|---|---|---|
| peer_approval | 0.400 | 0.401 | 0.001 |
| material_gain | 0.300 | 0.304 | 0.004 |
| status_gain | 0.300 | 0.289 | -0.011 |
| sanction_risk | -0.600 | -0.601 | -0.001 |
| gamma_status | 4.000 | 4.050 | 0.050 |
| gamma_tie | 2.000 | 1.469 | -0.531 |

Cosine 0.9999; all signs correct: True. gamma_tie (kinship) is weakly identified: 34 of the 38 ties have the feature, so the data rarely contrasts a tie with the feature against one without it.

## 3. Data efficiency

| episodes | decisions | cosine | gamma_status_hat | gamma_tie_hat |
|---|---|---|---|---|
| 1 | 1400 | 0.999 | 3.831 | 1.779 |
| 3 | 4200 | 0.999 | 3.396 | 3.110 |
| 10 | 14000 | 1.000 | 3.777 | 1.019 |
| 50 | 70000 | 1.000 | 4.026 | 0.937 |
| 160 | 224000 | 1.000 | 4.050 | 1.469 |

## 4. Held-out accuracy

| model | action accuracy | good/bad accuracy |
|---|---|---|
| Behavioural cloning | 0.450 | 0.723 |
| Recovered reward | 0.470 | 0.732 |
| Majority class | 0.370 | nan |

Accuracy is capped by the agents' own randomness: they sample actions from a softmax, so even the true reward cannot predict every choice.

## 5. Explanations (held-out decisions, agents with 2 or more neighbours)

| method | top1 | top3 |
|---|---|---|
| Recovered attention | 1.000 | 1.000 |
| Most similar neighbour | 0.789 | 0.965 |
| Random neighbour (expected) | 0.319 | 0.809 |

Spearman correlation between attributed tie importance and |ablation effect|: 0.741 over 38 ties. Surrogate decision tree fidelity: 0.68.

## 6. Sensitivity

| setting | final_bad_share | ci_low | ci_high |
|---|---|---|---|
| true theta | 0.719 | 0.687 | 0.751 |
| sanction weight doubled | 0.316 | 0.277 | 0.355 |

## 7. Cascades

| policy | mean_abs_gap |
|---|---|
| bc | 0.015 |
| recovered | 0.002 |

| policy | mean_first_escalation_step | share_of_agent_runs_escalating |
|---|---|---|
| bc | 3.54 | 1.00 |
| recovered | 3.73 | 1.00 |
| true | 3.71 | 1.00 |

## 8. Interventions

| scenario | baseline_bad_share | scenario_bad_share | delta | ci_low | ci_high |
|---|---|---|---|---|---|
| Hub isolation (A11) | 0.714 | 0.671 | -0.043 | -0.062 | -0.024 |
| Mentor injection (A07) | 0.714 | 0.602 | -0.112 | -0.128 | -0.096 |
| Supervision +0.2 | 0.714 | 0.496 | -0.218 | -0.236 | -0.200 |
| Real disruption (post-attack graph) | 0.714 | 0.616 | -0.098 | -0.116 | -0.080 |

## 9. Insights

1. Behaviour alone was enough to recover the hidden motives. From 224,000 observed decisions the learner matched the true reward weights with cosine 0.9999 (largest error 0.011, on status gain). With a single episode (1,400 decisions) the cosine was already 0.999.
2. Who an agent listens to matters as much as what they want. Recovered attention named the true influencer in 100% of held-out decisions, against 79% for the most similar neighbour and 32% for a random one.
3. Supervision +0.2 cut the final bad share most, by 0.218 (from 0.71), and its 95% interval excludes zero. The real post-attack network changed it by -0.098. Removing any single tie moved it by at most 0.009, so in this model behaviour is driven by incentives and supervision more than by any one relationship.
4. A model of motives generalises better than a copy of actions: rolled forward, the recovered reward stays closer to the true cascade than behavioural cloning (mean gap 0.002 vs 0.015). One-step accuracy alone hides this: 0.47 for the recovered reward vs 0.45 for cloning.

## 10. Acceptance criteria (SPEC.md section 11)

| check | passed | value |
|---|---|---|
| Sensitivity: doubling the sanction weight lowers final bad share by at least 0.20 | pass | 0.403 |
| Action variety: every action is at least 5% of decisions | pass | 13.8%, 19.3%, 28.2%, 38.7% |
| Recovery (full data): cosine at least 0.98 | pass | 0.9999 |
| Recovery (full data): every theta within 0.05 of truth | pass | 0.011 |
| Recovery (full data): all four signs correct | pass | True |
| Recovery (1 episode): cosine at least 0.90 | pass | 0.9995 |
| gamma_status within 1.0 of truth | pass | 4.05 (true 4.0) |
| Explanations: Top-1 at least 0.95 | pass | 1.000 |
| Explanations: Top-1 above the most-similar baseline | pass | 1.000 vs 0.789 |
| BC: held-out accuracy above the majority class | pass | 0.450 vs 0.370 |
| Cascade: recovered-policy gap below the BC gap | pass | 0.0024 vs 0.0155 |

All criteria pass on this network.
