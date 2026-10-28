# Drugnet network demo: summary

_Network structure is real; all agent traits, actions and rewards are simulated and make no claim about real individuals._

Mode: **full**, seed 42.

## 1. Dataset

| stat | value |
|---|---|
| nodes | 212 |
| ties | 284 |
| mutual_pairs | 53 |
| group_sizes | 193, 3, 3, 3, 2, 2, 2, 2 and 1 more |
| groups | 9 |
| isolates | 0 |
| max_degree | 15 |
| highest_degree_agent | A045 (degree 15) |
| second_highest_degree_agent | A028 (degree 11) |
| average_degree | 2.679 |
| hub_share | 0.053 |
| tie_feature | Mutual tie |
| tie_feature_nonzero | 53 |
| tie_feature_mean | 0.187 |
| isolates_not_in_file | 81 |

## 2. True vs recovered parameters

| parameter | true | recovered | error |
|---|---|---|---|
| peer_approval | 0.400 | 0.399 | -0.001 |
| material_gain | 0.300 | 0.294 | -0.006 |
| status_gain | 0.300 | 0.308 | 0.008 |
| sanction_risk | -0.600 | -0.596 | 0.004 |
| gamma_status | 4.000 | 4.007 | 0.007 |
| gamma_tie | 2.000 | 2.001 | 0.001 |

Cosine 0.9999; all signs correct: True. gamma_tie (mutual tie) is weakly identified: 53 of the 284 ties have the feature, so the data rarely contrasts a tie with the feature against one without it.

## 3. Data efficiency

| episodes | decisions | cosine | gamma_status_hat | gamma_tie_hat |
|---|---|---|---|---|
| 1 | 10600 | 0.999 | 4.881 | 2.128 |
| 3 | 31800 | 1.000 | 4.180 | 1.943 |
| 10 | 106000 | 0.999 | 4.112 | 1.975 |
| 50 | 530000 | 1.000 | 4.025 | 1.965 |
| 160 | 1696000 | 1.000 | 4.007 | 2.001 |

## 4. Held-out accuracy

| model | action accuracy | good/bad accuracy |
|---|---|---|
| Behavioural cloning | 0.452 | 0.723 |
| Recovered reward | 0.477 | 0.735 |
| Majority class | 0.378 | nan |

Accuracy is capped by the agents' own randomness: they sample actions from a softmax, so even the true reward cannot predict every choice.

## 5. Explanations (held-out decisions, agents with 2 or more neighbours)

| method | top1 | top3 |
|---|---|---|
| Recovered attention | 1.000 | 1.000 |
| Most similar neighbour | 0.677 | 0.957 |
| Random neighbour (expected) | 0.346 | 0.854 |

Spearman correlation between attributed tie importance and |ablation effect|: 0.375 over 60 ties. Surrogate decision tree fidelity: 0.77.

## 6. Sensitivity

| setting | final_bad_share | ci_low | ci_high |
|---|---|---|---|
| true theta | 0.717 | 0.704 | 0.729 |
| sanction weight doubled | 0.284 | 0.274 | 0.294 |

## 7. Cascades

| policy | mean_abs_gap |
|---|---|
| bc | 0.006 |
| recovered | 0.001 |

| policy | mean_first_escalation_step | share_of_agent_runs_escalating |
|---|---|---|
| bc | 3.82 | 1.00 |
| recovered | 3.75 | 1.00 |
| true | 3.72 | 1.00 |

## 8. Interventions

| scenario | baseline_bad_share | scenario_bad_share | delta | ci_low | ci_high |
|---|---|---|---|---|---|
| Hub isolation (A045) | 0.710 | 0.708 | -0.002 | -0.006 | 0.003 |
| Mentor injection (A028) | 0.710 | 0.676 | -0.034 | -0.037 | -0.030 |
| Supervision +0.2 | 0.710 | 0.489 | -0.221 | -0.227 | -0.216 |

## 9. Insights

1. Behaviour alone was enough to recover the hidden motives. From 1,696,000 observed decisions the learner matched the true reward weights with cosine 0.9999 (largest error 0.008, on status gain). With a single episode (10,600 decisions) the cosine was already 0.999.
2. Who an agent listens to matters as much as what they want. Recovered attention named the true influencer in 100% of held-out decisions, against 68% for the most similar neighbour and 35% for a random one.
3. Supervision +0.2 cut the final bad share most, by 0.221 (from 0.71), and its 95% interval excludes zero. This dataset has no later network to compare against. Removing any single tie moved it by at most 0.001, so in this model behaviour is driven by incentives and supervision more than by any one relationship.
4. A model of motives generalises better than a copy of actions: rolled forward, the recovered reward stays closer to the true cascade than behavioural cloning (mean gap 0.001 vs 0.006). One-step accuracy alone hides this: 0.48 for the recovered reward vs 0.45 for cloning.

## 10. Acceptance criteria (SPEC.md section 11)

| check | passed | value |
|---|---|---|
| Sensitivity: doubling the sanction weight lowers final bad share by at least 0.20 | pass | 0.433 |
| Action variety: every action is at least 5% of decisions | pass | 14.1%, 19.7%, 28.1%, 38.1% |
| Recovery (full data): cosine at least 0.98 | pass | 0.9999 |
| Recovery (full data): every theta within 0.05 of truth | pass | 0.008 |
| Recovery (full data): all four signs correct | pass | True |
| Recovery (1 episode): cosine at least 0.90 | pass | 0.9994 |
| gamma_status within 1.0 of truth | pass | 4.01 (true 4.0) |
| Explanations: Top-1 at least 0.95 | pass | 1.000 |
| Explanations: Top-1 above the most-similar baseline | pass | 1.000 vs 0.677 |
| BC: held-out accuracy above the majority class | pass | 0.452 vs 0.378 |
| Cascade: recovered-policy gap below the BC gap | pass | 0.0010 vs 0.0064 |

All criteria pass on this network.

## Notes on the data

The edge list contains only the 212 people with at least one tie. The dataset card lists 293 people; the 81 without ties are not in the file and their IDs cannot be reconstructed, so the model runs on the 212 connected agents. Isolates neither influence anyone nor are influenced, so they add nothing to influence or explanation. Ties are symmetrised "A says they know B" nominations; the tie feature marks a mutual pair. There is no later network, so the real-network intervention is skipped. Ablation covers the 60 ties with the highest edge betweenness. The attribute file (ethnicity, gender) is never read.
