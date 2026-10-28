# Knecht network demo: summary

_Network structure is real; all agent traits, actions and rewards are simulated and make no claim about real individuals._

Mode: **full**, seed 42.

## 1. Dataset

| stat | value |
|---|---|
| nodes | 26 |
| ties | 63 |
| mutual_pairs | 28 |
| group_sizes | 26 |
| groups | 1 |
| isolates | 0 |
| max_degree | 10 |
| highest_degree_agent | A09 (degree 10) |
| second_highest_degree_agent | A12 (degree 9) |
| average_degree | 4.846 |
| hub_share | 0.159 |
| tie_feature | Mutual friendship |
| tie_feature_nonzero | 28 |
| tie_feature_mean | 0.444 |
| later_network | Wave 4 |
| later_ties | 86 |
| later_isolated_agents | 1 |
| pairs_wave1 | 63 |
| top_agent_wave1 | A09 (degree 10) |
| pairs_wave2 | 84 |
| top_agent_wave2 | A10 (degree 14) |
| pairs_wave3 | 97 |
| top_agent_wave3 | A04 (degree 13) |
| pairs_wave4 | 86 |
| top_agent_wave4 | A08 (degree 15) |
| wave1_pairs_still_in_wave4 | 0.603 |

## 2. True vs recovered parameters

| parameter | true | recovered | error |
|---|---|---|---|
| peer_approval | 0.400 | 0.397 | -0.003 |
| material_gain | 0.300 | 0.295 | -0.005 |
| status_gain | 0.300 | 0.308 | 0.008 |
| sanction_risk | -0.600 | -0.601 | -0.001 |
| gamma_status | 4.000 | 3.977 | -0.023 |
| gamma_tie | 2.000 | 2.010 | 0.010 |

Cosine 0.9999; all signs correct: True. gamma_tie (mutual friendship) is reasonably identified: 28 of the 63 ties have the feature, so the data often contrasts a tie with the feature against one without it.

## 3. Data efficiency

| episodes | decisions | cosine | gamma_status_hat | gamma_tie_hat |
|---|---|---|---|---|
| 1 | 1300 | 0.999 | 3.341 | 1.844 |
| 3 | 3900 | 0.990 | 3.949 | 1.849 |
| 10 | 13000 | 0.999 | 3.534 | 1.658 |
| 50 | 65000 | 1.000 | 3.751 | 1.935 |
| 160 | 208000 | 1.000 | 3.977 | 2.010 |

## 4. Held-out accuracy

| model | action accuracy | good/bad accuracy |
|---|---|---|
| Behavioural cloning | 0.431 | 0.708 |
| Recovered reward | 0.455 | 0.720 |
| Majority class | 0.376 | nan |

Accuracy is capped by the agents' own randomness: they sample actions from a softmax, so even the true reward cannot predict every choice.

## 5. Explanations (held-out decisions, agents with 2 or more neighbours)

| method | top1 | top3 |
|---|---|---|
| Recovered attention | 0.984 | 1.000 |
| Most similar neighbour | 0.538 | 0.909 |
| Random neighbour (expected) | 0.262 | 0.686 |

Spearman correlation between attributed tie importance and |ablation effect|: 0.434 over 60 ties. Surrogate decision tree fidelity: 0.64.

## 6. Sensitivity

| setting | final_bad_share | ci_low | ci_high |
|---|---|---|---|
| true theta | 0.704 | 0.670 | 0.738 |
| sanction weight doubled | 0.275 | 0.244 | 0.306 |

## 7. Cascades

| policy | mean_abs_gap |
|---|---|
| bc | 0.012 |
| recovered | 0.004 |

| policy | mean_first_escalation_step | share_of_agent_runs_escalating |
|---|---|---|
| bc | 4.03 | 1.00 |
| recovered | 3.99 | 1.00 |
| true | 3.93 | 1.00 |

## 8. Interventions

| scenario | baseline_bad_share | scenario_bad_share | delta | ci_low | ci_high |
|---|---|---|---|---|---|
| Hub isolation (A09) | 0.731 | 0.716 | -0.016 | -0.023 | -0.009 |
| Mentor injection (A12) | 0.731 | 0.587 | -0.144 | -0.158 | -0.131 |
| Supervision +0.2 | 0.731 | 0.489 | -0.243 | -0.256 | -0.229 |
| Real disruption (wave 4 graph) | 0.731 | 0.730 | -0.001 | -0.009 | 0.007 |

## 9. Insights

1. Behaviour alone was enough to recover the hidden motives. From 208,000 observed decisions the learner matched the true reward weights with cosine 0.9999 (largest error 0.008, on status gain). With a single episode (1,300 decisions) the cosine was already 0.999.
2. Who an agent listens to matters as much as what they want. Recovered attention named the true influencer in 98% of held-out decisions, against 54% for the most similar neighbour and 26% for a random one.
3. Supervision +0.2 cut the final bad share most, by 0.243 (from 0.73), and its 95% interval excludes zero. The real wave 4 network changed it by -0.001. Removing any single tie moved it by at most 0.007, so in this model behaviour is driven by incentives and supervision more than by any one relationship.
4. A model of motives generalises better than a copy of actions: rolled forward, the recovered reward stays closer to the true cascade than behavioural cloning (mean gap 0.004 vs 0.012). One-step accuracy alone hides this: 0.46 for the recovered reward vs 0.43 for cloning.

## 10. Acceptance criteria (SPEC.md section 11)

| check | passed | value |
|---|---|---|
| Sensitivity: doubling the sanction weight lowers final bad share by at least 0.20 | pass | 0.428 |
| Action variety: every action is at least 5% of decisions | pass | 13.1%, 19.6%, 28.9%, 38.4% |
| Recovery (full data): cosine at least 0.98 | pass | 0.9999 |
| Recovery (full data): every theta within 0.05 of truth | pass | 0.008 |
| Recovery (full data): all four signs correct | pass | True |
| Recovery (1 episode): cosine at least 0.90 | pass | 0.9990 |
| gamma_status within 1.0 of truth | pass | 3.98 (true 4.0) |
| Explanations: Top-1 at least 0.95 | pass | 0.984 |
| Explanations: Top-1 above the most-similar baseline | pass | 0.984 vs 0.538 |
| BC: held-out accuracy above the majority class | pass | 0.431 vs 0.376 |
| Cascade: recovered-policy gap below the BC gap | pass | 0.0038 vs 0.0122 |

All criteria pass on this network.

## Notes on the data

Knecht is a class of children. Only best-friend nominations are used; no demographic, alcohol or delinquency data enters the pipeline, and every behaviour shown is simulated. From the dataset card only: the class-average delinquency score rose from 1.38 to 1.88 over the four waves. The main network is wave 1 and the later network is wave 4 on the same 26 students; the one student without ties in wave 4 left the class. Missing answers were dropped from the edge list, so a missing answer cannot be told apart from "not a friend".
