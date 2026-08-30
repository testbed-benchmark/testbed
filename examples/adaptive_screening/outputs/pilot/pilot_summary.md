# Adaptive screening: non-neural go/no-go pilot

**Decision: PASS.**
Passing pair: [2.5, 3.0].

Environment hash: `27ba90adacffe551da92b6e03e2c93b9dc9747a48bd5879c2e9a744a7475f373`  
Code commit: `3a84da59e30dc66514c489e1e1fc6daf82c01600`  
Source snapshot hash (audit only; includes uncommitted experiment code): `768709c6ad10abdd5df2290e7f83968080c9f5351c2a5c0e97e81baca82ab684`

All computations ran locally on CPU in float64. Prior maximum from quadrature: **0.640189392575**.
Validation: 20,000 episodes, seed 271828. Tests: 50,000 episodes, seed 314159, 50,000 episodes, seed 161803.

All configurations were tuned on validation, then frozen before either test dataset was generated. Confidence intervals use paired episode differences on the common latent utilities and candidate-specific observation noise. Pooled intervals concatenate both independent test sets; they do not average seed-level standard errors.

## Frozen hyperparameters

| κ | Bayes β | Aware β | ETC k | ETC S | ETC leader | Static k |
| --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | 2.800000 | 2.800000 | 16 | 19 | mean | 16 |
| 1.000000 | 2.800000 | 2.600000 | 16 | 16 | mean | 12 |
| 1.500000 | 2.800000 | 2.600000 | 16 | 16 | mean | 12 |
| 2.000000 | 2.800000 | 2.000000 | 16 | 16 | mean | 6 |
| 2.500000 | 2.800000 | 1.800000 | 16 | 16 | mean | 6 |
| 3.000000 | 2.800000 | 2.000000 | 15 | 15 | mean | 4 |

## Rule-awareness gains on frozen tests

| κ | Bayes J | Aware J | Gain | Paired SE | 95% lower | 95% upper | 314159 | 161803 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | 0.522700 | 0.522700 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1.000000 | 0.488697 | 0.488833 | 0.000136 | 0.000282 | -0.000417 | 0.000689 | 0.000200 | 0.000071 |
| 1.500000 | 0.450952 | 0.451885 | 0.000933 | 0.000313 | 0.000320 | 0.001546 | 0.001142 | 0.000724 |
| 2.000000 | 0.405281 | 0.418470 | 0.013189 | 0.000889 | 0.011447 | 0.014930 | 0.015462 | 0.010915 |
| 2.500000 | 0.357213 | 0.390013 | 0.032800 | 0.001138 | 0.030569 | 0.035031 | 0.033885 | 0.031715 |
| 3.000000 | 0.310408 | 0.360100 | 0.049692 | 0.001077 | 0.047582 | 0.051802 | 0.050129 | 0.049255 |

## Adaptive advantage and behavior

| κ | Aware−ETC | CI lower | Aware−static | CI lower | Best frozen adaptive−static | Bayes default | Aware default | EIG disagreement | Bayes distinct | Aware distinct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | 0.072584 | 0.071034 | 0.130886 | 0.128656 | 0.130886 | 0.207520 | 0.207520 | 0.000000 | 14.953480 | 14.953480 |
| 1.000000 | 0.076669 | 0.074815 | 0.205935 | 0.203333 | 0.205935 | 0.416730 | 0.413150 | 0.440940 | 14.953480 | 14.469650 |
| 1.500000 | 0.062919 | 0.060978 | 0.248465 | 0.245655 | 0.248465 | 0.515010 | 0.510510 | 0.590000 | 14.953480 | 14.469650 |
| 2.000000 | 0.058135 | 0.055994 | 0.275588 | 0.272639 | 0.275588 | 0.603570 | 0.554520 | 0.677990 | 14.953480 | 12.037490 |
| 2.500000 | 0.063658 | 0.061236 | 0.284569 | 0.281664 | 0.284569 | 0.679030 | 0.582890 | 0.722310 | 14.953480 | 10.942690 |
| 3.000000 | 0.070836 | 0.068569 | 0.279338 | 0.276271 | 0.279338 | 0.741670 | 0.653900 | 0.743940 | 14.953480 | 12.037490 |

The best frozen adaptive gap is descriptive only; no policy is selected or retuned on these test results.

## Pooled policy metrics

| κ | Policy | J | risk | bayes_component | incompatibility | default | disagreement | distinct | budget_fraction | last_six_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | ucb_bayes | 0.522700 | 0.117490 | 0.117490 | 0.000000 | 0.207520 | 0.000000 | 14.953480 | 0.253919 | 0.476843 |
| 0.000000 | ucb_rule_aware | 0.522700 | 0.117490 | 0.117490 | 0.000000 | 0.207520 | 0.000000 | 14.953480 | 0.253919 | 0.476843 |
| 0.000000 | explore_certify | 0.450116 | 0.190074 | 0.190074 | 0.000000 | 0.277500 | 0.000000 | 16.000000 | 0.166403 | 0.525725 |
| 0.000000 | nonadaptive | 0.391814 | 0.248375 | 0.248375 | 0.000000 | 0.239890 | 0.000000 | 16.000000 | 0.049970 | 0.054900 |
| 0.000000 | eig | 0.391814 | 0.248375 | 0.248375 | 0.000000 | 0.239890 | 0.000000 | 16.000000 | 0.049970 | 0.054900 |
| 0.000000 | random | 0.350789 | 0.289400 | 0.289400 | 0.000000 | 0.309720 | 0.000000 | 12.598660 | 0.060033 | 0.060350 |
| 1.000000 | ucb_bayes | 0.488697 | 0.151493 | 0.117490 | 0.034003 | 0.416730 | 0.209910 | 14.953480 | 0.211615 | 0.378385 |
| 1.000000 | ucb_rule_aware | 0.488833 | 0.151357 | 0.118364 | 0.032993 | 0.413150 | 0.206000 | 14.469650 | 0.234150 | 0.391397 |
| 1.000000 | explore_certify | 0.412164 | 0.228025 | 0.196535 | 0.031490 | 0.494560 | 0.207680 | 16.000000 | 0.187040 | 0.497940 |
| 1.000000 | nonadaptive | 0.282898 | 0.357292 | 0.272346 | 0.084946 | 0.667230 | 0.370200 | 12.000000 | 0.027731 | 0.027773 |
| 1.000000 | eig | 0.277803 | 0.362387 | 0.248375 | 0.114011 | 0.679410 | 0.440940 | 16.000000 | 0.022593 | 0.027688 |
| 1.000000 | random | 0.259487 | 0.380702 | 0.289400 | 0.091302 | 0.689500 | 0.382260 | 12.598660 | 0.030690 | 0.030788 |
| 1.500000 | ucb_bayes | 0.450952 | 0.189237 | 0.117490 | 0.071747 | 0.515010 | 0.308370 | 14.953480 | 0.188934 | 0.325655 |
| 1.500000 | ucb_rule_aware | 0.451885 | 0.188304 | 0.118364 | 0.069940 | 0.510510 | 0.303850 | 14.469650 | 0.210994 | 0.340557 |
| 1.500000 | explore_certify | 0.388966 | 0.251223 | 0.196535 | 0.054688 | 0.565350 | 0.276570 | 16.000000 | 0.162657 | 0.433640 |
| 1.500000 | nonadaptive | 0.203420 | 0.436769 | 0.272346 | 0.164423 | 0.804390 | 0.507360 | 12.000000 | 0.016301 | 0.016332 |
| 1.500000 | eig | 0.182316 | 0.457873 | 0.248375 | 0.209498 | 0.828860 | 0.590000 | 16.000000 | 0.012623 | 0.016515 |
| 1.500000 | random | 0.183321 | 0.456869 | 0.289400 | 0.167469 | 0.817660 | 0.510290 | 12.598660 | 0.019562 | 0.019568 |
| 2.000000 | ucb_bayes | 0.405281 | 0.234908 | 0.117490 | 0.117418 | 0.603570 | 0.397010 | 14.953480 | 0.166659 | 0.275403 |
| 2.000000 | ucb_rule_aware | 0.418470 | 0.221720 | 0.133972 | 0.087747 | 0.554520 | 0.345820 | 12.037490 | 0.286577 | 0.418903 |
| 2.000000 | explore_certify | 0.360335 | 0.279854 | 0.196535 | 0.083319 | 0.632210 | 0.340280 | 16.000000 | 0.137875 | 0.367650 |
| 2.000000 | nonadaptive | 0.142882 | 0.497308 | 0.384112 | 0.113196 | 0.865820 | 0.364900 | 6.000000 | 0.022363 | 0.022363 |
| 2.000000 | eig | 0.104984 | 0.535205 | 0.248375 | 0.286830 | 0.917400 | 0.677990 | 16.000000 | 0.006349 | 0.008697 |
| 2.000000 | random | 0.118133 | 0.522057 | 0.289400 | 0.232656 | 0.899220 | 0.590870 | 12.598660 | 0.011702 | 0.011587 |
| 2.500000 | ucb_bayes | 0.357213 | 0.282976 | 0.117490 | 0.165487 | 0.679030 | 0.472570 | 14.953480 | 0.146002 | 0.230622 |
| 2.500000 | ucb_rule_aware | 0.390013 | 0.250176 | 0.149534 | 0.100642 | 0.582890 | 0.367430 | 10.942690 | 0.294783 | 0.414075 |
| 2.500000 | explore_certify | 0.326355 | 0.313835 | 0.196535 | 0.117300 | 0.694400 | 0.400750 | 16.000000 | 0.114597 | 0.305590 |
| 2.500000 | nonadaptive | 0.105444 | 0.534745 | 0.384112 | 0.150634 | 0.913480 | 0.412560 | 6.000000 | 0.014420 | 0.014420 |
| 2.500000 | eig | 0.055786 | 0.584404 | 0.248375 | 0.336028 | 0.962100 | 0.722310 | 16.000000 | 0.003004 | 0.004300 |
| 2.500000 | random | 0.072642 | 0.567547 | 0.289400 | 0.278147 | 0.945420 | 0.636400 | 12.598660 | 0.006772 | 0.006622 |
| 3.000000 | ucb_bayes | 0.310408 | 0.329781 | 0.117490 | 0.212292 | 0.741670 | 0.534990 | 14.953480 | 0.127331 | 0.192333 |
| 3.000000 | ucb_rule_aware | 0.360100 | 0.280089 | 0.133972 | 0.146117 | 0.653900 | 0.443490 | 12.037490 | 0.241750 | 0.343293 |
| 3.000000 | explore_certify | 0.289264 | 0.350926 | 0.210950 | 0.139976 | 0.744650 | 0.430600 | 15.000000 | 0.106396 | 0.255350 |
| 3.000000 | nonadaptive | 0.080762 | 0.559427 | 0.445860 | 0.113567 | 0.934120 | 0.318720 | 4.000000 | 0.016470 | 0.016520 |
| 3.000000 | eig | 0.027189 | 0.613000 | 0.248375 | 0.364625 | 0.983800 | 0.743940 | 16.000000 | 0.001315 | 0.001938 |
| 3.000000 | random | 0.042774 | 0.597416 | 0.289400 | 0.308016 | 0.971190 | 0.661700 | 12.598660 | 0.003798 | 0.003728 |

## Exact gate conditions

### Pair [2.0, 2.5]: FAIL

| Condition | κ | Seed | Result | Observed | Required |
| --- | --- | --- | --- | --- | --- |
| primary_protocol | None | None | PASS | true | Exact preregistered primary configuration, datasets and tuning grids |
| pooled_rule_awareness_gain | 2.000000 | None | FAIL | 0.013188522126619561 | >= 0.02 |
| pooled_rule_awareness_ci | 2.000000 | None | PASS | 0.011446989519053463 | 95% paired CI lower > 0 |
| seed_rule_awareness_gain | 2.000000 | 314159 | PASS | 0.015461871030610686 | >= 0.015 |
| no_seed_sign_reversal | 2.000000 | 314159 | PASS | 0.015461871030610686 | > 0 |
| seed_rule_awareness_gain | 2.000000 | 161803 | FAIL | 0.010915173222628437 | >= 0.015 |
| no_seed_sign_reversal | 2.000000 | 161803 | PASS | 0.010915173222628437 | > 0 |
| adaptive_over_static | 2.000000 | None | PASS | {"ci_low": 0.2726393337957737, "mean": 0.27558803640881807} | Mean and 95% paired CI lower > 0 |
| non_saturated_ucb_bayes | 2.000000 | None | PASS | 0.60357 | Default rate <= 0.85 |
| non_saturated_ucb_rule_aware | 2.000000 | None | PASS | 0.55452 | Default rate <= 0.85 |
| pooled_rule_awareness_gain | 2.500000 | None | PASS | 0.03280001803608112 | >= 0.02 |
| pooled_rule_awareness_ci | 2.500000 | None | PASS | 0.03056878498902496 | 95% paired CI lower > 0 |
| seed_rule_awareness_gain | 2.500000 | 314159 | PASS | 0.033884942886423554 | >= 0.015 |
| no_seed_sign_reversal | 2.500000 | 314159 | PASS | 0.033884942886423554 | > 0 |
| seed_rule_awareness_gain | 2.500000 | 161803 | PASS | 0.03171509318573868 | >= 0.015 |
| no_seed_sign_reversal | 2.500000 | 161803 | PASS | 0.03171509318573868 | > 0 |
| adaptive_over_static | 2.500000 | None | PASS | {"ci_low": 0.28166355428920165, "mean": 0.2845690960205305} | Mean and 95% paired CI lower > 0 |
| non_saturated_ucb_bayes | 2.500000 | None | PASS | 0.67903 | Default rate <= 0.85 |
| non_saturated_ucb_rule_aware | 2.500000 | None | PASS | 0.58289 | Default rate <= 0.85 |
| adaptive_over_explore_certify | None | None | PASS | [{"ci_low": 0.05599430625129692, "kappa": 2.0, "mean": 0.05813456936193672, "pass": true}, {"ci_low": 0.061235789133179265, "kappa": 2.5, "mean": 0.06365846631458923, "pass": true}] | At either pair value, gain >= 0.02 and 95% paired CI lower > 0 |
| reduced_search_breadth | 2.500000 | None | PASS | {"ci_high": -3.9811330544891095, "mean": -4.01079} | Aware minus Bayes <= -1 and 95% paired CI upper < 0 |
| eig_action_disagreement | 2.500000 | None | PASS | 0.72231 | >= 0.10 |

### Pair [2.5, 3.0]: PASS

| Condition | κ | Seed | Result | Observed | Required |
| --- | --- | --- | --- | --- | --- |
| primary_protocol | None | None | PASS | true | Exact preregistered primary configuration, datasets and tuning grids |
| pooled_rule_awareness_gain | 2.500000 | None | PASS | 0.03280001803608112 | >= 0.02 |
| pooled_rule_awareness_ci | 2.500000 | None | PASS | 0.03056878498902496 | 95% paired CI lower > 0 |
| seed_rule_awareness_gain | 2.500000 | 314159 | PASS | 0.033884942886423554 | >= 0.015 |
| no_seed_sign_reversal | 2.500000 | 314159 | PASS | 0.033884942886423554 | > 0 |
| seed_rule_awareness_gain | 2.500000 | 161803 | PASS | 0.03171509318573868 | >= 0.015 |
| no_seed_sign_reversal | 2.500000 | 161803 | PASS | 0.03171509318573868 | > 0 |
| adaptive_over_static | 2.500000 | None | PASS | {"ci_low": 0.28166355428920165, "mean": 0.2845690960205305} | Mean and 95% paired CI lower > 0 |
| non_saturated_ucb_bayes | 2.500000 | None | PASS | 0.67903 | Default rate <= 0.85 |
| non_saturated_ucb_rule_aware | 2.500000 | None | PASS | 0.58289 | Default rate <= 0.85 |
| pooled_rule_awareness_gain | 3.000000 | None | PASS | 0.0496920134355377 | >= 0.02 |
| pooled_rule_awareness_ci | 3.000000 | None | PASS | 0.04758194515889408 | 95% paired CI lower > 0 |
| seed_rule_awareness_gain | 3.000000 | 314159 | PASS | 0.050128847743889565 | >= 0.015 |
| no_seed_sign_reversal | 3.000000 | 314159 | PASS | 0.050128847743889565 | > 0 |
| seed_rule_awareness_gain | 3.000000 | 161803 | PASS | 0.04925517912718583 | >= 0.015 |
| no_seed_sign_reversal | 3.000000 | 161803 | PASS | 0.04925517912718583 | > 0 |
| adaptive_over_static | 3.000000 | None | PASS | {"ci_low": 0.27627062318685125, "mean": 0.2793376186198964} | Mean and 95% paired CI lower > 0 |
| non_saturated_ucb_bayes | 3.000000 | None | PASS | 0.74167 | Default rate <= 0.85 |
| non_saturated_ucb_rule_aware | 3.000000 | None | PASS | 0.6539 | Default rate <= 0.85 |
| adaptive_over_explore_certify | None | None | PASS | [{"ci_low": 0.061235789133179265, "kappa": 2.5, "mean": 0.06365846631458923, "pass": true}, {"ci_low": 0.06856919452206958, "kappa": 3.0, "mean": 0.07083619553539586, "pass": true}] | At either pair value, gain >= 0.02 and 95% paired CI lower > 0 |
| reduced_search_breadth | 3.000000 | None | PASS | {"ci_high": -2.8893767098382517, "mean": -2.91599} | Aware minus Bayes <= -1 and 95% paired CI upper < 0 |
| eig_action_disagreement | 3.000000 | None | PASS | 0.74394 | >= 0.10 |

## Metric conventions and scope

Budget and last-six fractions are unconditional, assigning zero when the default is selected. Selected posterior SD is conditional on deployment. First-leader round counts post-observation rounds 1..T, taking the largest candidate posterior mean (excluding the default), and is conditional on deployment and ever being that leader. A separate never-leader rate records deployed candidates that were never the posterior-mean leader. Empty conditional groups are left blank in CSVs; raw episode metrics remain finite.

Maximum risk-decomposition residual: 1.39e-16. At κ=0 the UCB objectives and their frozen settings coincide exactly.

This deliverable ends at the non-neural gate. No neural policy was initialized or trained, and no HPC jobs were submitted. A passing pilot supports proceeding to the learned-policy experiment; it is not evidence about learned-policy performance.
