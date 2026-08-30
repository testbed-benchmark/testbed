# Adaptive screening: final learned-policy experiment

**Decision: PASS.**

Environment hash: `27ba90adacffe551da92b6e03e2c93b9dc9747a48bd5879c2e9a744a7475f373`  
Final test: 100,000 episodes, seed 8675309. Learned results average 5 independent training seeds.

## Primary learned-policy differences

| κ deploy | Bayes learned J | Rule-aware learned J | Rule-aware − Bayes | 95% CI | Rule-aware − matched UCB |
| ---: | ---: | ---: | ---: | :--- | ---: |
| 2.5 | 0.355106 | 0.392324 | 0.037217 | [0.035479, 0.038956] | 0.004682 |
| 3.0 | 0.308775 | 0.363023 | 0.054248 | [0.052419, 0.056077] | 0.004601 |

## Risk decomposition and behavior

| κ | Bayes incompat. | Aware incompat. | Bayes distinct | Aware distinct | Bayes last-six | Aware last-six | Bayes default | Aware default |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.5 | 0.165926 | 0.105306 | 14.908 | 11.823 | 0.195 | 0.404 | 0.680 | 0.593 |
| 3.0 | 0.212258 | 0.138438 | 14.908 | 12.057 | 0.163 | 0.352 | 0.742 | 0.647 |

At both gate-selected mismatch values, the matched learned policy has a positive paired advantage over both the Bayes-assumption learned policy and matched rule-aware UCB. The reduction in risk is primarily a reduction in incompatibility, accompanied by narrower search and more late measurements on the selected candidate.

Maximum risk-decomposition residual: 1.11e-16.
