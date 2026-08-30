# Controlled Study of Downstream-Rule Mismatch

## Adaptive Candidate Screening with a Fixed Eligibility Rule

## 1. Experiment overview and scope

This specification defines a single adaptive candidate-screening experiment for studying downstream-rule mismatch.

The experimenter sequentially tests a collection of candidate treatments, products, suppliers, models, or interventions. Most candidates are worse than a known default, but a small number are genuinely beneficial. After a fixed testing budget, a downstream decision maker either deploys one candidate or retains the default.

The experimenter uses ordinary simple regret. Its Bayes-optimal rule selects the candidate with the largest positive posterior mean, or the default if all posterior means are nonpositive. The deployed downstream decision maker imposes a fixed evidentiary requirement: a candidate is eligible only if a lower posterior bound exceeds the default. Among eligible candidates, the decision maker selects the one with the largest posterior mean.

The evidentiary multiplier \(\kappa\) controls downstream-rule mismatch. At \(\kappa=0\), the deployed rule is exactly the experimenter’s Bayes-optimal rule. For \(\kappa>0\), a policy that assumes the Bayes rule may search broadly for a high-mean candidate without collecting enough repeated evidence for that candidate to become eligible. A rule-aware policy must balance search against targeted certification.

The implementation must preserve the following structural properties:

- the experimenter’s loss is a standard state-and-action loss and does not depend on the realized data;
- agreement at \(\kappa=0\) is exact;
- the fixed rule creates a certification constraint that the Bayes-assumption policy does not optimize directly;
- the environment is conjugate, inexpensive to simulate, and straightforward to vectorize;
- the policy is trained using a Rao--Blackwellized terminal reward;
- round-robin sampling is an exact expected-information-gain baseline;
- strong non-neural adaptive policies are evaluated before any neural training;
- and the supplied reference benchmarks provide implementation sanity checks in the high-mismatch regime.

The required scope contains one downstream decision maker and the single experiment defined in this document. Do not add alternative experimenter losses, multiple-DM variants, or unrelated ablations before completing the required pilot, learned-policy comparison, and final evaluation.

## 2. Non-negotiable staging rule

**Do not train a neural adaptive policy until the minimal go/no-go procedure in Section 11 has completed and written a passing gate file.**

The implementation should enforce this mechanically:

1. `run_pilot.py` implements and evaluates only analytic and hand-designed policies.
2. `check_gate.py` reads the pilot results and writes `pilot_gate.json`.
3. `train_policy.py` must refuse to run unless `pilot_gate.json` contains `"pass": true` for the exact environment configuration being requested.
4. If the gate fails, stop the experiment and produce the diagnostic report specified in Section 11.9. Do not begin neural-policy tuning, change neural architectures, or launch large policy-gradient runs.

The point of the gate is to establish, cheaply and before reinforcement learning, that:

- the environment contains a meaningful downstream-rule mismatch;
- adapting specifically to the deployed rule matters;
- adaptivity adds value beyond a tuned static allocation;
- the useful regime is not dominated by the default action;
- and the expected search-to-certification behavior is present.

## 3. Scientific question

The experiment should answer the following controlled question:

> When the experimenter’s Bayes-optimal action rule and the deployed fixed action rule agree at zero mismatch, how does the cost of ignoring the deployed rule grow as the deployed rule demands progressively stronger evidence, and can a rule-aware adaptive design reduce that cost by changing how it allocates measurements between search and certification?

The principal comparison is between:

- a **Bayes-assumption design policy**, optimized under the assumption that the Bayes-optimal downstream action rule will be deployed; and
- a **rule-aware design policy**, optimized for the actual fixed downstream eligibility rule.

Both are evaluated using the same deployed fixed rule.

## 4. Statistical model

### 4.1 Candidate utilities and the default

There are \(d\) candidate actions. Candidate \(i\) has unknown incremental utility

\[
\theta_i \sim \mathcal N(\mu_d,\tau^2),
\qquad i=1,\ldots,d,
\]

independently.

There is a known default action \(a=0\) with utility

\[
\theta_0=0.
\]

The final downstream action is

\[
a\in\{0,1,\ldots,d\}.
\]

Choosing \(a=0\) means retaining the default. Choosing \(a=i\) means deploying candidate \(i\).

### 4.2 Sequential observations

At round \(t\in\{1,\ldots,T\}\), the design policy chooses

\[
x_t\in\{1,\ldots,d\}
\]

and observes

\[
y_t=\theta_{x_t}+\epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,\sigma^2).
\]

The history after round \(t\) is

\[
D_t=\{(x_r,y_r)\}_{r=1}^t.
\]

Repeated observations of a candidate are conditionally independent given its utility.

### 4.3 Conjugate posterior

Let

- \(n_{t,i}\) be the number of times candidate \(i\) has been sampled by round \(t\); and
- \(q_{t,i}=\sum_{r:x_r=i}y_r\) be its accumulated observation sum.

Then

\[
\theta_i\mid D_t
\sim
\mathcal N(m_{t,i},s_{t,i}^2),
\]

where

\[
s_{t,i}^2
=
\left(
\tau^{-2}+n_{t,i}\sigma^{-2}
\right)^{-1},
\]

and

\[
m_{t,i}
=
s_{t,i}^2
\left(
\tau^{-2}\mu_d+\sigma^{-2}q_{t,i}
\right).
\]

For the default, define

\[
m_{t,0}=0,
\qquad
s_{t,0}=0.
\]

The environment state can therefore be represented exactly by the arrays

```text
n:    shape [d], integer sample counts
sum_y: shape [d], accumulated observation sums
m:    shape [d], posterior means
s:    shape [d], posterior standard deviations
```

There is no need to retain the complete raw history for inference.

### 4.4 Prior calibration

The prior should make beneficial candidates rare while keeping their expected number stable as \(d\) changes. Let \(K\) be the desired expected number of candidates satisfying \(\theta_i>0\). Set

\[
\mu_d
=
\tau\Phi^{-1}\!\left(\frac{K}{d}\right).
\]

Then

\[
\mathbb E\left[
\sum_{i=1}^d\mathbf 1\{\theta_i>0\}
\right]
=K.
\]

Use

\[
K=2.
\]

For the main experiment,

\[
d=16,
\qquad
\tau=1,
\]

so

\[
\mu_{16}=\Phi^{-1}(2/16)\approx-1.15034938.
\]

### 4.5 Main environment configuration

Use the following values exactly for the first go/no-go run and the primary experiment:

```yaml
d: 16
T: 24
tau: 1.0
sigma: 0.75
expected_positive_candidates: 2
mu: -1.1503493803760083
kappa_grid:
  - 0.0
  - 1.0
  - 1.5
  - 2.0
  - 2.5
  - 3.0
```

Do not tune \(d\), \(T\), \(\tau\), \(\sigma\), \(K\), or \(\mu_d\) after viewing the pilot results. A failed primary gate should be reported as a failed experiment rather than followed by open-ended environment search.

## 5. Experimenter loss and Bayes-optimal action rule

Let

\[
M(\theta)
=
\max_{j\in\{0,1,\ldots,d\}}\theta_j
=
\max\{0,\theta_1,\ldots,\theta_d\}.
\]

The experimenter uses simple regret:

\[
\ell(a,\theta)
=
M(\theta)-\theta_a.
\]

Conditionally on \(D_T\),

\[
\mathbb E[\ell(a,\theta)\mid D_T]
=
\mathbb E[M(\theta)\mid D_T]-m_{T,a}.
\]

The first term is independent of \(a\). The experimenter’s Bayes-optimal action rule is therefore

\[
\pi_a^\star(D_T)
=
\arg\max_{i\in\{0,1,\ldots,d\}}m_{T,i}.
\]

Equivalently,

\[
\pi_a^\star(D_T)
=
\begin{cases}
0,
& \max_{1\leq i\leq d}m_{T,i}\leq0,\\[3pt]
\arg\max_{1\leq i\leq d}m_{T,i},
& \max_{1\leq i\leq d}m_{T,i}>0.
\end{cases}
\]

Ties have probability zero under the continuous model. For deterministic tests and numerical reproducibility, break exact floating-point ties using the smallest candidate index after applying the episode-level random candidate permutation described in Section 10.2.

## 6. Fixed downstream eligibility rule

### 6.1 Eligibility set

For a fixed evidentiary multiplier \(\kappa\geq0\), define

\[
\mathcal E_\kappa(D_T)
=
\left\{
 i\in\{1,\ldots,d\}:
 m_{T,i}-\kappa s_{T,i}>0
\right\}.
\]

Candidate \(i\) is eligible only when its lower posterior bound exceeds the utility of the known default.

### 6.2 Deployed action rule

The downstream decision maker applies

\[
\pi_a^\kappa(D_T)
=
\begin{cases}
0,
& \mathcal E_\kappa(D_T)=\varnothing,\\[3pt]
\displaystyle
\arg\max_{i\in\mathcal E_\kappa(D_T)}m_{T,i},
& \mathcal E_\kappa(D_T)\neq\varnothing.
\end{cases}
\]

Thus, the rule has two stages:

1. discard every candidate that fails the evidentiary requirement;
2. among the eligible candidates, deploy the candidate with the largest posterior mean.

The final selection is based on utility, not on the lower-bound score itself. The lower bound acts only as an eligibility constraint.

### 6.3 Exact agreement at zero mismatch

At \(\kappa=0\),

\[
\mathcal E_0(D_T)
=
\{i:m_{T,i}>0\}.
\]

The fixed rule therefore chooses the candidate with the largest positive posterior mean, or the default when all posterior means are nonpositive. Hence

\[
\pi_a^0(D_T)=\pi_a^\star(D_T)
\]

for every dataset, up to the shared deterministic tie-breaking convention.

This identity must be covered by a unit test over at least \(10^5\) randomly generated posterior states.

### 6.4 Interpretation of \(\kappa\)

Because the posterior is Gaussian, \(m_{T,i}-\kappa s_{T,i}>0\) is equivalent to requiring

\[
\Pr(\theta_i>0\mid D_T)>\Phi(\kappa).
\]

For orientation:

- \(\kappa=0\) requires posterior probability above \(0.5\);
- \(\kappa=1\) requires posterior probability above approximately \(0.841\);
- \(\kappa=1.5\) requires posterior probability above approximately \(0.933\);
- \(\kappa=2\) requires posterior probability above approximately \(0.977\);
- \(\kappa=2.5\) requires posterior probability above approximately \(0.994\);
- \(\kappa=3\) requires posterior probability above approximately \(0.9987\).

The main effect is expected at \(\kappa\in\{2.5,3\}\). Lower values are retained to show the transition from exact agreement to substantial mismatch.

### 6.5 Empirical mismatch measure

In addition to reporting \(\kappa\), report the action-disagreement rate under the exact EIG reference policy:

\[
\operatorname{Disagree}(\kappa)
=
\Pr_{\pi_d^{\mathrm{EIG}}}
\left[
\pi_a^\kappa(D_T)\neq\pi_a^\star(D_T)
\right].
\]

This supplies a policy-independent measure of the effective mismatch induced by a numerical value of \(\kappa\).

## 7. Deployment risk, equivalent return, and exact decomposition

Write

\[
a_\kappa(D_T)=\pi_a^\kappa(D_T).
\]

The deployment risk is

\[
R_\kappa(\pi_d)
=
\mathbb E\left[
M(\theta)-\theta_{a_\kappa(D_T)}
\right].
\]

By iterated expectation,

\[
R_\kappa(\pi_d)
=
\mathbb E[M(\theta)]
-
\mathbb E\left[m_{T,a_\kappa(D_T)}\right].
\]

Define

\[
J_\kappa(\pi_d)
=
\mathbb E\left[r_\kappa(D_T)\right],
\]

where the Rao--Blackwellized terminal reward is

\[
r_\kappa(D_T)
=
m_{T,a_\kappa(D_T)}.
\]

For the default, this reward is zero. Minimizing deployment risk is exactly equivalent to maximizing \(J_\kappa\), because \(\mathbb E[M(\theta)]\) is constant across design policies.

### 7.1 Bayes-risk component

Define the experimenter’s optimal posterior value

\[
V^\star(D_T)
=
\max_{i\in\{0,1,\ldots,d\}}m_{T,i}
=
\max\left\{0,\max_i m_{T,i}\right\}.
\]

The Bayes-risk component is

\[
B(\pi_d)
=
\mathbb E[M(\theta)]
-
\mathbb E[V^\star(D_T)].
\]

### 7.2 Incompatibility component

The conditional incompatibility of the fixed rule is

\[
\delta_\kappa(D_T)
=
V^\star(D_T)-m_{T,a_\kappa(D_T)}
\geq0.
\]

The design-dependent incompatibility cost is

\[
C_\kappa(\pi_d)
=
\mathbb E[\delta_\kappa(D_T)].
\]

The total risk decomposes exactly as

\[
R_\kappa(\pi_d)
=
B(\pi_d)+C_\kappa(\pi_d).
\]

At \(\kappa=0\),

\[
C_0(\pi_d)=0
\]

for every design policy.

### 7.3 Computing \(\mathbb E[M(\theta)]\)

For the independent Gaussian prior and default utility zero,

\[
\mathbb E[M(\theta)]
=
\int_0^\infty
\left[
1-
\Phi\!\left(\frac{x-\mu_d}{\tau}\right)^d
\right]\,\mathrm dx.
\]

Compute this scalar once for each \(d\) using `scipy.integrate.quad` with absolute and relative tolerances \(10^{-10}\). Store it in the experiment metadata. Do not estimate it separately for each policy.

The identities to verify numerically are

```text
risk = expected_prior_max - mean(selected_posterior_mean)
bayes_component = expected_prior_max - mean(max(0, max_i posterior_mean_i))
incompatibility = mean(max(0, max_i posterior_mean_i) - selected_posterior_mean)
risk = bayes_component + incompatibility
```

The final equality should hold to within \(10^{-10}\) when all quantities are computed from the same evaluation trajectories in float64.

## 8. Why adaptivity and rule awareness should matter

The prior makes beneficial candidates rare. A useful policy must first search for a promising candidate. Once one is found, the best next observation depends on the rule that will use the final data.

### 8.1 Bayes-assumption policy

A policy trained for \(\kappa=0\) is rewarded for finding the candidate with the largest posterior mean. Once a candidate appears promising, it can continue searching for an even better candidate because the assumed downstream rule does not require a positive lower bound beyond having a positive posterior mean.

### 8.2 Rule-aware policy

For \(\kappa>0\), a high posterior mean is not enough. The deployed rule may retain the default unless at least one candidate satisfies

\[
m_{T,i}>\kappa s_{T,i}.
\]

A rule-aware policy should therefore:

- abandon candidates after poor observations;
- search broadly early in the trajectory;
- stop broad search earlier when a plausible leader emerges;
- allocate repeated late measurements to one or two promising candidates;
- reduce the posterior standard deviation of the likely selected candidate;
- increase the probability that a genuinely beneficial candidate becomes eligible;
- and reduce the probability that the fixed rule retains the default despite the existence of a useful candidate.

### 8.3 Structural source of the rule-awareness gap

The experimenter’s Bayes-optimal action rule depends only on posterior means. Its terminal value is

\[
V^\star(D_T)
=
\max\left\{0,\max_i m_{T,i}\right\}.
\]

A design policy trained at \(\kappa=0\) is therefore rewarded for producing a large posterior mean for at least one candidate, but it is not directly required to make that candidate satisfy a positive lower-bound constraint.

For \(\kappa>0\), the deployed rule additionally depends on posterior uncertainty through

\[
m_{T,i}-\kappa s_{T,i}>0.
\]

The rule-aware objective assigns zero terminal reward when no candidate is eligible, even when one or more candidates have positive posterior means. Repeated measurements of a promising candidate can therefore be valuable because they reduce \(s_{T,i}\) and move the candidate across the eligibility boundary. This creates the intended separation: the Bayes-assumption policy primarily values continued search for a larger posterior mean, while the rule-aware policy must account for whether the final dataset will certify a candidate for deployment.

## 9. Policies to implement

All policies operate on the same conjugate posterior state. The primary metric is \(J_\kappa\), with higher values better. Report deployment risk as \(R_\kappa=\mathbb E[M]-J_\kappa\).

### 9.1 Bayes-assumption learned policy

Train one policy for

\[
\kappa_{\mathrm{train}}=0.
\]

This policy solves

\[
\pi_d^{\mathrm B}
\in
\arg\max_{\pi_d}J_0(\pi_d).
\]

At evaluation time, apply the actual deployed rule \(\pi_a^\kappa\) for each \(\kappa\) of interest. The design policy itself remains the one trained at zero mismatch.

### 9.2 Rule-aware learned policy

For each selected nonzero training value \(\kappa\), train

\[
\pi_d^{\mathrm F,\kappa}
\in
\arg\max_{\pi_d}J_\kappa(\pi_d).
\]

Evaluate this policy under the matching deployed rule and, for transfer diagnostics, under every value in the evaluation grid.

### 9.3 UCB-style adaptive heuristic

For \(\beta\geq0\), define

\[
I_{t,i}^{\beta}
=
m_{t,i}+\beta s_{t,i}.
\]

At each round, select

\[
x_{t+1}
=
\arg\max_i I_{t,i}^{\beta}.
\]

This heuristic searches candidates with high posterior means or high remaining uncertainty. It is not claimed to be a theoretically optimal bandit policy; it is an inexpensive and expressive one-parameter adaptive policy class for detecting whether rule awareness matters.

Tune \(\beta\) on the validation set over

```text
beta_grid = [0.0, 0.2, 0.4, ..., 3.8, 4.0]
```

using the exact deployment return for the target training value of \(\kappa\).

Define:

- `ucb_bayes`: the \(\beta\) selected by maximizing \(J_0\);
- `ucb_rule_aware[kappa]`: the \(\beta\) selected separately by maximizing \(J_\kappa\).

The main cheap estimate of the value of rule awareness is

\[
J_\kappa(\text{ucb\_rule\_aware}[\kappa])
-
J_\kappa(\text{ucb\_bayes}).
\]

### 9.4 Tuned explore-then-certify heuristic

This is a strong two-stage adaptive baseline with hyperparameters

\[
(k,S,h),
\]

where:

- \(k\in\{1,\ldots,d\}\) is the number of candidates searched;
- \(S\in\{k,\ldots,T-1\}\) is the number of search-stage measurements; and
- \(h\in\{\texttt{mean},\texttt{margin}\}\) is the leader-selection rule.

For each episode:

1. draw a uniform random permutation of the \(d\) candidates;
2. take the first \(k\) candidates as the search subset;
3. for rounds \(1,\ldots,S\), sample the subset cyclically in its permuted order;
4. after round \(S\), choose a single candidate \(c\) from the search subset:
   - if `h = mean`, set \(c=\arg\max_i m_{S,i}\);
   - if `h = margin`, set \(c=\arg\max_i(m_{S,i}-\kappa s_{S,i})\);
5. allocate every remaining observation to \(c\).

Tune all valid combinations of \((k,S,h)\) separately for each \(\kappa\) using the validation set. This is at most

\[
2\sum_{k=1}^{d}(T-k)
\]

configurations and is inexpensive in the main setting.

### 9.5 Optimized nonadaptive subset design

For each

\[
k\in\{1,\ldots,d\},
\]

choose a uniform random subset of \(k\) candidates at the start of the episode and allocate the complete budget as evenly as possible across them. Counts may differ by at most one. Randomize which subset candidates receive the extra observations when \(T\) is not divisible by \(k\).

Tune \(k\) separately for every deployment value \(\kappa\) using validation simulations. This is the strongest exchangeable static allocation in this family.

### 9.6 Exact expected-information-gain baseline

If candidate \(i\) is measured \(n_i\) times, the mutual information is

\[
I(\theta;D_T)
=
\frac12
\sum_{i=1}^d
\log\left(
1+\frac{n_i\tau^2}{\sigma^2}
\right).
\]

For fixed \(\sum_i n_i=T\), concavity of the logarithm implies that EIG is maximized by balancing the counts as evenly as possible. The exact EIG policy is therefore round-robin sampling, with candidate order randomized independently at the beginning of each episode.

### 9.7 Uniform random adaptive design

At every round, sample a candidate uniformly from \(\{1,\ldots,d\}\), independently of the history.

### 9.8 Optional posterior-mean greedy heuristic

The case \(\beta=0\) of the UCB family is a greedy posterior-mean policy. It need not be reported separately if it is already included in the UCB tuning table.

## 10. Simulation and common-random-number protocol

### 10.1 Pre-generated latent episodes

For each evaluation episode \(e\), pre-generate

```text
theta[e, i]      for e in 1..N and i in 1..d
noise[e, i, n]   for e in 1..N, i in 1..d, n in 1..T
uniform[e, t]    for e in 1..N and t in 1..T
permutation[e]   a random permutation of 1..d
```

When a policy samples candidate \(i\) for the \(n\)-th time, reveal

\[
y_{i,n}=\theta_i+\epsilon_{i,n}.
\]

This makes the latent utilities and every candidate-specific potential observation identical across policies, despite their different adaptive trajectories.

For stochastic categorical policies, use `uniform[e, t]` with inverse-CDF categorical sampling so that the same uniforms are reused across compared policies. This does not make trajectories identical, but it provides a consistent common-random-number coupling.

### 10.2 Candidate permutations and tie-breaking

Every episode has an independent random permutation of candidate labels. Deterministic policies act on candidates in the permuted order and break exact ties using the earliest candidate in that order. This prevents arbitrary candidate indices from affecting the result.

The neural network is permutation equivariant and should not require a fixed candidate order. Nevertheless, apply a random candidate permutation to every training trajectory as a defensive implementation check.

### 10.3 Numerical precision

Use:

- float32 for neural-policy training;
- float64 for pilot tuning, final policy evaluation, confidence intervals, and decomposition checks.

Enable JAX 64-bit mode in all pilot and evaluation entry points.

## 11. Mandatory minimal go/no-go procedure

This section must be completed before any neural policy is initialized or trained.

### 11.1 Pilot objective

The pilot asks whether inexpensive adaptive policy classes already exhibit the scientific phenomenon required by the paper. It is not a neural-policy benchmark.

The pilot compares:

- `ucb_bayes`, tuned once at \(\kappa=0\);
- `ucb_rule_aware[kappa]`, tuned separately at each \(\kappa\);
- the best explore-then-certify heuristic at each \(\kappa\);
- the best nonadaptive subset policy at each \(\kappa\);
- exact EIG round-robin;
- and uniform random sampling.

### 11.2 Exact pilot datasets and seeds

Use three independent, pre-generated datasets:

```yaml
validation:
  seed: 271828
  episodes: 20000

test_1:
  seed: 314159
  episodes: 50000

test_2:
  seed: 161803
  episodes: 50000
```

Hyperparameters are selected using only the validation set. The two test sets are used only after all pilot hyperparameters have been frozen.

### 11.3 Pilot tuning procedure

For every \(\kappa\) in

```text
[0.0, 1.0, 1.5, 2.0, 2.5, 3.0]
```

perform the following:

1. Evaluate every \(\beta\in\{0,0.2,\ldots,4.0\}\) on the validation set under the deployed rule \(\pi_a^\kappa\).
2. Store the maximizing value as `beta_rule_aware[kappa]`.
3. Store `beta_bayes = beta_rule_aware[0.0]` once and reuse it unchanged at all nonzero deployment values.
4. Exhaustively tune the explore-then-certify grid from Section 9.4.
5. Tune the nonadaptive subset size \(k\in\{1,\ldots,16\}\).
6. Freeze all selected hyperparameters.
7. Evaluate every frozen policy on both independent test sets.

If two validation configurations differ by less than \(10^{-4}\) in estimated return, choose the simpler configuration using the following deterministic tie-breaks:

- for UCB, choose the smaller \(\beta\);
- for nonadaptive subsets, choose the smaller \(k\);
- for explore-then-certify, choose the smaller search budget \(S\), then the smaller \(k\), then `mean` before `margin`.

### 11.4 Pilot metrics

For each policy, \(\kappa\), and test seed, report:

- equivalent return \(J_\kappa\);
- deployment risk \(R_\kappa\);
- Bayes-risk component \(B\);
- incompatibility component \(C_\kappa\);
- fixed-versus-Bayes action disagreement rate;
- probability of selecting the default;
- probability of selecting a candidate with \(\theta_i>0\);
- mean utility of the selected action;
- mean number of distinct candidates sampled;
- fraction of the complete budget allocated to the finally selected candidate;
- fraction of the final six observations allocated to the finally selected candidate;
- posterior standard deviation of the selected candidate, conditional on deploying a candidate;
- and the first round at which the finally selected candidate became the posterior-mean leader.

For every comparison between two policies, calculate paired per-episode differences using the common-random-number coupling. Report the mean, standard error, and 95% normal confidence interval. A paired bootstrap with 2,000 resamples may be used as a secondary check.

### 11.5 Candidate passing pairs

Only the following adjacent high-mismatch pairs are eligible to open the neural-training gate:

```text
(2.0, 2.5)
(2.5, 3.0)
```

The first pair that satisfies all criteria below, checked in the listed order, becomes the `passing_pair`. In practice, the preliminary pilot suggests that `(2.5, 3.0)` is the more likely passing pair.

### 11.6 Rule-awareness separation criterion

For a test seed \(s\), define

\[
G_s(\kappa)
=
J_\kappa(\text{ucb\_rule\_aware}[\kappa])
-
J_\kappa(\text{ucb\_bayes}).
\]

For a candidate pair \((\kappa_1,\kappa_2)\), require all of the following:

1. For each \(\kappa\in\{\kappa_1,\kappa_2\}\), the pooled gain over both test sets is at least \(0.02\).
2. For each test seed separately and each \(\kappa\), the gain is at least \(0.015\).
3. For each \(\kappa\), the lower endpoint of the pooled paired 95% confidence interval is strictly greater than zero.
4. No test seed reverses the sign of the gain.

These thresholds are intentionally larger than typical Monte Carlo uncertainty and small policy-training fluctuations.

### 11.7 Adaptivity and non-saturation criteria

For the same candidate pair, require:

1. At at least one of the two \(\kappa\) values, the rule-aware UCB policy exceeds the best tuned explore-then-certify return by at least \(0.02\), with a paired 95% confidence interval whose lower endpoint is positive.
2. At both \(\kappa\) values, the rule-aware UCB policy exceeds the best nonadaptive subset policy, with a positive paired 95% confidence interval.
3. At both \(\kappa\) values, neither `ucb_bayes` nor `ucb_rule_aware[kappa]` selects the default in more than 85% of episodes.
4. At the larger \(\kappa\) in the pair, the rule-aware UCB policy samples at least one fewer distinct candidate on average than `ucb_bayes`, and the upper endpoint of the paired 95% confidence interval for
   \[
   \text{distinct}_{\mathrm{aware}}-
   \text{distinct}_{\mathrm{Bayes}}
   \]
   is below zero.
5. Under round-robin sampling at the larger \(\kappa\), the fixed rule and the Bayes rule disagree on at least 10% of episodes.

### 11.8 Gate decision

Set

```json
{
  "pass": true,
  "passing_pair": [kappa_1, kappa_2],
  "environment_hash": "...",
  "pilot_result_hash": "..."
}
```

only if one candidate pair satisfies every criterion in Sections 11.6 and 11.7 on the frozen test evaluations.

The environment hash must include at least:

```text
d, T, tau, sigma, K, mu, kappa_grid,
validation seed and size,
test seeds and sizes,
all baseline tuning grids,
code commit hash
```

`train_policy.py` must recompute the requested environment hash and compare it with the gate file. A gate from another configuration is invalid.

### 11.9 Required behavior when the gate fails

If neither candidate pair passes:

1. write `pilot_gate.json` with `"pass": false`;
2. do not train a neural policy;
3. save all pilot tables and tuning curves;
4. produce `pilot_failure_report.md` containing:
   - the best observed rule-aware gain at every \(\kappa\);
   - confidence intervals;
   - the best adaptive-versus-static gap;
   - default-selection rates;
   - action-disagreement rates;
   - distinct-candidate counts;
   - and which exact gate conditions failed;
5. stop and request scientific review before changing the environment.

Do not automatically increase \(\kappa\), reduce \(T\), change the prior, or launch architecture searches after a failed gate. That would recreate the open-ended calibration process this procedure is designed to prevent.

### 11.10 Reference smoke-test values

The following values are approximate reference benchmarks for the exact environment and policy definitions in this specification. They are not acceptance criteria, but large deviations may indicate an implementation error:

| \(\kappa\) | Bayes-tuned adaptive return | Rule-aware adaptive return | Approximate gain |
|---:|---:|---:|---:|
| 0.0 | 0.520 | 0.520 | 0.000 |
| 1.0 | 0.485 | 0.485 | 0.000 |
| 1.5 | 0.447 | 0.449 | 0.002 |
| 2.0 | 0.402 | 0.416 | 0.014 |
| 2.5 | 0.354 | 0.389 | 0.035 |
| 3.0 | 0.308 | 0.360 | 0.052 |

At \(\kappa=3\), the reference evaluation sampled approximately 15 distinct candidates under the Bayes-tuned policy and 11.5 under the rule-aware policy. The corresponding candidate-deployment probabilities were approximately 25.7% and 35.5%. Treat these only as broad smoke-test targets.

## 12. Neural policy parameterization

This section is executed only after the gate passes.

Use a categorical, permutation-equivariant policy over the \(d\) candidates. Do not use an LSTM over an arbitrarily ordered raw history.

### 12.1 Candidate-local features

For candidate \(i\) at time \(t\), construct

\[
\xi_{t,i}
=
\left(
 m_{t,i},
 s_{t,i},
 \log(s_{t,i}+10^{-8}),
 \frac{n_{t,i}}{T},
 \frac{m_{t,i}}{s_{t,i}+10^{-6}},
 m_{t,i}-\kappa_{\mathrm{train}}s_{t,i}
\right).
\]

The final component allows the architecture to represent the deployed eligibility margin. For the Bayes-assumption policy, use \(\kappa_{\mathrm{train}}=0\).

### 12.2 Candidate encoder

Apply the same two-layer MLP to every candidate:

```text
Dense(64)
SiLU
Dense(64)
SiLU
```

This yields embeddings \(e_{t,i}\in\mathbb R^{64}\).

### 12.3 Permutation-invariant pooled summaries

Compute

```text
mean_pool = mean_i e[t, i]    # shape 64
max_pool  = max_i e[t, i]     # shape 64
```

and the following scalar global features:

```text
t / T
(T - t) / T
largest posterior mean
second-largest posterior mean
largest eligibility margin
second-largest eligibility margin
mean posterior standard deviation
minimum posterior standard deviation
fraction of candidates sampled at least once
entropy of normalized sampling counts
```

Define count entropy as zero when \(t=0\). Normalize it by \(\log d\), so it lies in \([0,1]\).

### 12.4 Candidate scorer

For each candidate, concatenate

```text
candidate embedding: 64
mean pool:           64
max pool:            64
global scalars:      10
```

and apply

```text
Dense(128)
SiLU
Dense(64)
SiLU
Dense(1)
```

The resulting scalar is the candidate logit. The design distribution is

\[
\pi_\phi(x_{t+1}=i\mid D_t)
=
\operatorname{softmax}_i(h_{t,i}).
\]

Initialize all layers with standard variance-scaled initialization. Initialize the final bias to zero so that the initial policy is exchangeable.

### 12.5 Architecture tests

Before training, verify:

1. permuting candidate features permutes the output logits identically;
2. applying the inverse permutation recovers the original logits to tolerance \(10^{-6}\);
3. parameter count is independent of \(d\);
4. gradients are finite on a batch containing both entirely unsampled and heavily sampled candidates.

## 13. Neural-policy training

### 13.1 Training values of \(\kappa\)

Let the gate-selected pair be

\[
(\kappa_1,\kappa_2).
\]

The initial neural training set is exactly

\[
\{0,\kappa_1,\kappa_2\}.
\]

Train:

- one Bayes-assumption policy at \(\kappa_{\mathrm{train}}=0\);
- one rule-aware policy at \(\kappa_{\mathrm{train}}=\kappa_1\);
- one rule-aware policy at \(\kappa_{\mathrm{train}}=\kappa_2\).

Do not train the remaining mismatch values until these three policies have passed the neural optimization checks in Section 13.7.

### 13.2 Optional UCB imitation initialization

To reduce optimization risk, initialize each neural policy by behavior cloning the corresponding tuned UCB heuristic:

- the Bayes-assumption policy imitates `ucb_bayes`;
- each rule-aware policy imitates `ucb_rule_aware[kappa]`.

Generate fresh prior trajectories online. At every time step, use the UCB-selected candidate as the supervised class label. Optimize categorical cross-entropy with:

```yaml
optimizer: Adam
learning_rate: 0.001
updates: 1000
trajectory_batch_size: 512
states_per_trajectory: 24
```

Use all time-step states in the loss. Save a checkpoint after behavior cloning and evaluate it on the neural validation set. This initialization changes only the starting point, not the final policy-gradient objective.

For completeness, a from-scratch run may be included as an appendix diagnostic. It must not replace the UCB-initialized run specified for the main workflow.

### 13.3 REINFORCE objective

For trajectory \(j\), define

\[
r_j
=
m_{T,a_{\kappa_{\mathrm{train}}}(D_T^{(j)})}.
\]

For a batch of \(B\) independent trajectories, use the leave-one-out baseline

\[
b_j
=
\frac{1}{B-1}
\sum_{k\neq j}r_k.
\]

The trajectory advantage is

\[
A_j=r_j-b_j.
\]

Divide the stopped-gradient advantages by the stopped-gradient batch standard deviation plus \(10^{-6}\). The actor loss is

\[
\mathcal L_{\mathrm{actor}}
=
-
\frac1B
\sum_{j=1}^B
\widehat A_j
\sum_{t=0}^{T-1}
\log\pi_\phi(x_{t+1}^{(j)}\mid D_t^{(j)}).
\]

Add entropy regularization:

\[
\mathcal L
=
\mathcal L_{\mathrm{actor}}
-
\alpha_t
\frac1{BT}
\sum_{j,t}
\mathcal H\!\left[
\pi_\phi(\cdot\mid D_t^{(j)})
\right].
\]

The downstream action is deterministic and nondifferentiable, but it appears only inside the score-function return and therefore requires no relaxation.

### 13.4 Training hyperparameters

Use:

```yaml
reinforce:
  optimizer: AdamW
  learning_rate_initial: 0.0003
  learning_rate_final: 0.00003
  warmup_steps: 250
  total_updates: 6000
  trajectory_batch_size: 1024
  weight_decay: 0.00001
  global_gradient_clip: 1.0
  entropy_coefficient_initial: 0.01
  entropy_coefficient_final: 0.001
  entropy_decay_steps: 3000
  validation_every_updates: 200
  validation_episodes: 10000
  early_stopping_patience_updates: 1600
  minimum_validation_improvement: 0.0005
```

Use a linear warmup followed by cosine learning-rate decay. Decay the entropy coefficient linearly for the first 3,000 updates and then hold it at 0.001.

Every gradient update must use newly simulated trajectories. Do not reuse the fixed validation or test episodes for training.

### 13.5 Training and validation seeds

Use the following independent policy-training seeds:

```text
[101, 202, 303, 404, 505]
```

Initially run seeds `[101, 202, 303]`. Run the final two seeds only after all three initial policies meet the optimization checks below.

Use a fixed neural validation environment seed of

```text
validation_seed = 424242
```

with 10,000 pre-generated episodes. Validation is used only for checkpoint selection and early stopping.

### 13.6 Checkpointing

Save:

- the post-imitation checkpoint;
- the most recent checkpoint;
- and the checkpoint with the highest validation \(J_{\kappa_{\mathrm{train}}}\).

Final evaluation must use the best validation checkpoint. Never choose checkpoints using the final test set.

### 13.7 Neural optimization checks

For each training value of \(\kappa\), compare the learned policy with the UCB heuristic used for initialization.

A trained policy passes optimization if:

1. its mean validation return is no more than \(0.005\) below the corresponding tuned UCB return;
2. at least two of the first three training seeds satisfy this threshold individually;
3. training return and held-out return do not show persistent divergence;
4. final policy entropy is not numerically zero at every state;
5. the policy does not collapse to repeatedly sampling one arbitrary candidate before observing data;
6. all gradients and parameters remain finite.

If any policy fails after the first three seeds:

- inspect implementation tests and optimization logs;
- allow one rerun with the same architecture and hyperparameters after correcting an identified software error;
- do not initiate a hyperparameter sweep;
- and do not run the final two seeds until the issue is resolved.

The environment gate establishes that the scientific signal exists. Failure at this stage should be treated as an optimization or implementation problem, not as a reason to recalibrate the environment.

## 14. Final evaluation protocol

### 14.1 Final test set

Use one fixed final environment test set:

```yaml
seed: 8675309
episodes: 100000
```

Pre-generate latent utilities, candidate-specific noise arrays, policy uniforms, and candidate permutations as in Section 10.

Evaluate every trained seed of every learned policy on this same test set. Also evaluate every frozen heuristic and static baseline on the same episodes.

### 14.2 Policy stochasticity

The principal learned-policy result uses categorical sampling from the learned policy. Also report greedy-logit evaluation as an optimization diagnostic, but do not substitute it for the stochastic result unless the paper explicitly defines the deployed design policy as deterministic.

### 14.3 Reporting uncertainty

Separate two forms of uncertainty:

1. **evaluation uncertainty:** paired episode-level standard errors or bootstrap intervals on the fixed test set;
2. **training uncertainty:** standard deviation across independently trained policy seeds.

For learned policies, report the mean of the five training-seed estimates. Include both:

```text
mean across training seeds
standard deviation across training seeds
paired evaluation standard error for the seed-averaged per-episode return
```

For the main Bayes-versus-rule-aware comparison, form a seed-averaged per-episode return for each policy, then compute paired differences on the common test episodes.

### 14.4 Cross-evaluation grid

Every learned policy should be evaluated under

```text
kappa_deploy = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0]
```

without retraining. This yields a training-versus-deployment transfer table even when only three policies have initially been trained.

If the three-policy result succeeds and time remains, train additional matched rule-aware policies for the unused deployment values. These extra runs are optional and must not delay the minimal main-paper result.

## 15. Primary comparisons

At each gate-selected nonzero \(\kappa\), compare:

1. Bayes-assumption learned policy, trained at \(\kappa=0\);
2. matched rule-aware learned policy;
3. matched rule-aware UCB heuristic;
4. Bayes-tuned UCB heuristic;
5. best explore-then-certify heuristic;
6. best nonadaptive subset;
7. exact EIG round-robin;
8. uniform random sampling.

The central quantity is

\[
J_\kappa(\pi_d^{\mathrm F,\kappa})
-
J_\kappa(\pi_d^{\mathrm B}),
\]

or equivalently

\[
R_\kappa(\pi_d^{\mathrm B})
-
R_\kappa(\pi_d^{\mathrm F,\kappa}).
\]

The experiment succeeds scientifically when the rule-aware learned policy preserves the heuristic separation established by the gate and the difference is explained by reduced incompatibility and more concentrated late certification.

## 16. Required figures and tables

### 16.1 Minimal main-paper figure

Use a three-panel figure.

#### Panel A: deployment risk versus mismatch

Plot deployment risk under the actual fixed rule against \(\kappa\). Include:

- Bayes-assumption learned policy;
- matched rule-aware learned policies at trained values;
- Bayes-tuned UCB;
- matched rule-aware UCB;
- exact EIG round-robin;
- and best nonadaptive subset.

At untrained values, do not draw a matched learned-policy curve unless a corresponding policy was actually trained. Sparse markers at the trained values are acceptable.

#### Panel B: risk decomposition

For the Bayes-assumption and matched rule-aware learned policies, show

\[
B(\pi_d),
\qquad
C_\kappa(\pi_d),
\qquad
R_\kappa(\pi_d).
\]

A stacked bar chart at the two gate-selected mismatch values is sufficient. The expected pattern is that the rule-aware policy mainly improves total risk by reducing \(C_\kappa\), possibly at the cost of a small increase in \(B\).

#### Panel C: search-versus-certification behavior

At \(\kappa=0\) and the larger gate-selected mismatch value, report:

- mean number of distinct candidates sampled;
- fraction of the complete budget assigned to the finally selected candidate;
- fraction of the last six measurements assigned to that candidate;
- and default-selection rate.

The rule-aware policy should search fewer candidates and concentrate more late measurements on its likely deployed candidate.

### 16.2 Main table

Report:

| Design policy | \(J_\kappa\) | Deployment risk | Bayes component | Incompatibility | Default rate | Distinct candidates |
|---|---:|---:|---:|---:|---:|---:|
| Bayes-assumption learned |  |  |  |  |  |  |
| Rule-aware learned |  |  |  |  |  |  |
| Bayes-tuned UCB |  |  |  |  |  |  |
| Rule-aware UCB |  |  |  |  |  |  |
| Explore-then-certify |  |  |  |  |  |  |
| Best nonadaptive |  |  |  |  |  |  |
| EIG round-robin |  |  |  |  |  |  |
| Random |  |  |  |  |  |  |

Produce one table for each of the two gate-selected mismatch values or use grouped columns.

### 16.3 Optional appendix figures

Only after the main result is complete:

- full training-versus-deployment heatmap;
- action-disagreement rate versus \(\kappa\);
- posterior uncertainty of the selected candidate;
- representative candidate-level trajectories;
- training curves for all seeds;
- histogram of the round at which the final selected candidate became the leader;
- and the full heuristic hyperparameter tuning curves.

## 17. Optional ablations after the primary result

These are strictly lower priority than completing the main three-policy experiment.

### 17.1 Dimension ablation

Use

\[
d\in\{8,16,32\},
\qquad
T=\operatorname{round}(1.5d),
\qquad
K=2,
\qquad
\tau=1,
\qquad
\sigma=0.75,
\]

with

\[
\mu_d=\Phi^{-1}(2/d).
\]

Repeat the heuristic gate before training at any new dimension. Do not assume the main \(\kappa\) pair transfers unchanged.

### 17.2 Budget ablation

At \(d=16\), use

\[
T\in\{16,24,32\}.
\]

The main expected effect is largest at an intermediate budget. Run heuristic policies first and train neural policies only at settings that independently pass the gate.

### 17.3 Additional matched policies

After success at \(\{0,\kappa_1,\kappa_2\}\), optionally train matched policies for the remaining values in

```text
[1.0, 1.5, 2.0, 2.5, 3.0]
```

that were not already trained. This supports a smoother curve but is not required for the minimal claim.

## 18. Unit and integration tests

Implement the following tests before running the pilot.

### 18.1 Posterior tests

1. Compare incremental count-and-sum posterior updates with a batch Gaussian conjugate calculation.
2. Verify the posterior variance depends only on the count.
3. Verify repeated updates remain positive and finite for \(T=24\).
4. Compare empirical posterior-predictive moments with analytic moments.

### 18.2 Action-rule tests

1. For \(\kappa=0\), verify the fixed and Bayes actions agree over at least \(10^5\) random posterior states.
2. Verify an ineligible candidate is never selected.
3. Verify the selected nondefault candidate has the largest posterior mean among eligible candidates.
4. Verify the default is selected exactly when the eligibility set is empty.
5. Test deterministic tie-breaking explicitly with constructed equal means.

### 18.3 Risk tests

1. Verify \(R=B+C\) to float64 tolerance.
2. Verify \(C_0=0\).
3. Compare the Rao--Blackwellized return \(m_{T,a}\) with the Monte Carlo average of sampled \(\theta_a\) over many episodes.
4. Compare \(\mathbb E[M]\) from quadrature with a large prior Monte Carlo estimate.

### 18.4 Policy tests

1. Verify round-robin counts differ by at most one.
2. Verify the nonadaptive subset policy never samples outside its chosen subset.
3. Verify explore-then-certify samples only its final leader after round \(S\).
4. Verify the UCB policy selects the maximum index under the tie-breaking convention.
5. Verify common-random-number evaluation is deterministic for fixed seeds.
6. Verify neural policy permutation equivariance.

### 18.5 End-to-end smoke test

Run every policy for 256 episodes with \(d=4\), \(T=6\), and all \(\kappa\) values. Assert:

- no NaNs or infinities;
- all actions lie in valid ranges;
- all posterior standard deviations are positive;
- every risk decomposition closes numerically;
- and all output files contain the expected columns.

## 19. Recommended code structure

Use the following layout or a functionally equivalent one:

```text
adaptive_screening/
  README.md
  pyproject.toml
  configs/
    main.yaml
  screening/
    __init__.py
    config.py
    environment.py
    posterior.py
    action_rules.py
    metrics.py
    common_randomness.py
    quadrature.py
    policies/
      __init__.py
      round_robin.py
      random_policy.py
      nonadaptive_subset.py
      explore_then_certify.py
      ucb.py
      neural.py
    pilot.py
    gate.py
    imitate.py
    train.py
    evaluate.py
    plots.py
    io.py
  scripts/
    run_pilot.py
    check_gate.py
    train_policy.py
    evaluate_all.py
    make_figures.py
  tests/
    test_posterior.py
    test_action_rules.py
    test_risk_decomposition.py
    test_baselines.py
    test_common_randomness.py
    test_neural_equivariance.py
    test_smoke.py
  outputs/
```

Use JAX for simulation and neural training, Optax for optimization, SciPy only for the one-dimensional prior-maximum quadrature, and pandas or Polars for result tables. Plot with Matplotlib.

## 20. Required command sequence

The README should expose the following workflow.

### 20.1 Install and test

```bash
pip install -e .
pytest -q
```

### 20.2 Run the non-neural pilot

```bash
python scripts/run_pilot.py \
  --config configs/main.yaml \
  --output outputs/pilot
```

### 20.3 Check the gate

```bash
python scripts/check_gate.py \
  --pilot-dir outputs/pilot \
  --output outputs/pilot/pilot_gate.json
```

The command should print either

```text
PILOT GATE: PASS
```

or

```text
PILOT GATE: FAIL -- NEURAL TRAINING DISABLED
```

### 20.4 Train the three initial policies

The training script reads the gate-selected pair automatically:

```bash
python scripts/train_policy.py \
  --config configs/main.yaml \
  --gate outputs/pilot/pilot_gate.json \
  --train-kappa gate_pair_and_zero \
  --seeds 101 202 303 \
  --output outputs/training_initial
```

The command must terminate with a clear error when the gate is absent, false, or for another environment hash.

### 20.5 Run optimization checks

```bash
python scripts/evaluate_all.py \
  --config configs/main.yaml \
  --gate outputs/pilot/pilot_gate.json \
  --checkpoints outputs/training_initial \
  --mode validation_check \
  --output outputs/optimization_check
```

Only after this passes, run the final two seeds:

```bash
python scripts/train_policy.py \
  --config configs/main.yaml \
  --gate outputs/pilot/pilot_gate.json \
  --train-kappa gate_pair_and_zero \
  --seeds 404 505 \
  --output outputs/training_final
```

### 20.6 Final evaluation and figures

```bash
python scripts/evaluate_all.py \
  --config configs/main.yaml \
  --gate outputs/pilot/pilot_gate.json \
  --checkpoints outputs/training_initial outputs/training_final \
  --mode final_test \
  --output outputs/final

python scripts/make_figures.py \
  --results outputs/final \
  --output outputs/figures
```

## 21. Required output files

### 21.1 Pilot outputs

```text
outputs/pilot/config_resolved.yaml
outputs/pilot/environment_hash.txt
outputs/pilot/ucb_validation.csv
outputs/pilot/explore_certify_validation.csv
outputs/pilot/nonadaptive_validation.csv
outputs/pilot/test_metrics_by_seed.csv
outputs/pilot/paired_comparisons.csv
outputs/pilot/pilot_summary.md
outputs/pilot/pilot_gate.json
```

On failure, also write

```text
outputs/pilot/pilot_failure_report.md
```

### 21.2 Training outputs

For every \(\kappa\) and seed:

```text
checkpoint_bc.msgpack
checkpoint_best.msgpack
checkpoint_last.msgpack
training_log.csv
validation_log.csv
resolved_config.yaml
code_commit.txt
```

### 21.3 Final outputs

```text
outputs/final/policy_metrics_by_seed.csv
outputs/final/policy_metrics_aggregate.csv
outputs/final/paired_policy_differences.csv
outputs/final/risk_decomposition.csv
outputs/final/behavior_metrics.csv
outputs/final/training_deployment_matrix.csv
outputs/final/final_summary.md
outputs/figures/main_risk_vs_mismatch.pdf
outputs/figures/main_risk_decomposition.pdf
outputs/figures/main_policy_behavior.pdf
outputs/figures/main_results_table.tex
```

Every table should include the environment hash and code commit hash in metadata.

## 22. Suggested implementation order

Follow this order exactly:

1. Implement configuration loading and deterministic hashing.
2. Implement common-random-number episode generation.
3. Implement conjugate posterior updates.
4. Implement Bayes and fixed eligibility action rules.
5. Implement equivalent return and exact risk decomposition.
6. Implement unit tests for Sections 18.1--18.3.
7. Implement round-robin, random, nonadaptive subset, UCB, and explore-then-certify policies.
8. Implement policy and end-to-end tests.
9. Run the 256-episode smoke test.
10. Run the complete validation tuning stage.
11. Freeze pilot hyperparameters.
12. Run both pilot test seeds.
13. Compute the gate and write `pilot_gate.json`.
14. If the gate fails, write the failure report and stop.
15. If the gate passes, implement the neural policy.
16. Verify neural permutation equivariance.
17. Behavior-clone the matching UCB policies.
18. Run REINFORCE for seeds 101, 202, and 303.
19. Run the neural optimization checks.
20. If the checks pass, train seeds 404 and 505.
21. Run the final 100,000-episode evaluation.
22. Produce the three main panels and main result table.
23. Only then consider optional additional mismatch values or ablations.

## 23. Intended empirical conclusions

The experiment should support only claims that are directly verified.

The intended main conclusions are:

1. At \(\kappa=0\), the Bayes and fixed downstream rules coincide exactly, so Bayes-assumption and rule-aware policy objectives are identical.
2. As the fixed rule imposes a stronger eligibility requirement, a policy trained under the Bayes-rule assumption incurs increasing incompatibility cost.
3. A rule-aware policy reduces deployment risk by collecting a dataset that the fixed rule can use, primarily by shifting measurements from broad search toward targeted certification.
4. The effect is not explained only by a different static allocation: the adaptive policy outperforms the optimized nonadaptive subset design.
5. The learned policy matches or improves on the strong UCB and explore-then-certify baselines rather than merely beating round-robin.
6. The EIG baseline remains balanced and performs poorly in the high-mismatch regime because it does not react to observed candidate quality.

Do not claim monotonic changes in every behavioral statistic unless the measured curves support them. The central required result is lower deployment risk and lower incompatibility cost under the actual fixed rule.

## 24. Minimal main-paper result

The minimum acceptable paper result consists of:

1. a passed non-neural pilot gate;
2. learned policies at \(\kappa=0\) and the two gate-selected high-mismatch values;
3. a statistically clear rule-aware improvement at both high-mismatch values;
4. comparison with UCB, explore-then-certify, optimized nonadaptive, EIG, and random baselines;
5. an exact Bayes-risk/incompatibility decomposition;
6. and one behavioral panel showing reduced search breadth and increased late certification.

Dimension ablations, budget ablations, a dense mismatch sweep, and multiple-DM studies are outside the required deliverable. Add them only after the complete minimal result has been produced.

## 25. Implementation summary

Implement the single-DM eligibility-rule experiment in this document as the controlled downstream-mismatch study.

The primary configuration is

\[
d=16,
\qquad
T=24,
\qquad
\tau=1,
\qquad
\sigma=0.75,
\qquad
K=2.
\]

The experimenter minimizes simple regret. The Bayes-optimal downstream action selects the largest positive posterior mean. The fixed downstream rule requires \(m_i-\kappa s_i>0\) before a candidate may be selected and otherwise retains the default.

The non-neural pilot is mandatory. Neural policies are trained only if the pilot demonstrates a stable rule-awareness gain of at least \(0.02\) at two adjacent high-mismatch values, a genuine adaptive advantage, non-saturated default behavior, and the predicted reduction in search breadth.

The mandatory non-neural gate prevents neural training unless strong adaptive heuristics first establish a stable rule-awareness gap, a genuine adaptivity advantage, non-saturated deployment behavior, and the expected search-to-certification shift.
