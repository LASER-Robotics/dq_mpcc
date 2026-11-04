# Configuration-selection protocol

This document gives the complete rule used to select the deployed MPCC and
DQ-MPCC configurations reported in the article.

## 1. Tuning

Both controllers are tuned offline with Optuna: a multi-objective NSGA-III
exploration, a TPE refinement, and a final stage under matched progress-speed
bounds. Each stage uses 100 trials per controller and three MuJoCo seeds.
The optimized gain vector is

    theta = [q_phi (or q_q for MPCC), q_c, q_l, w_omega, w_s, v_s_max]

with isotropic weights `Q = q I_3`. The thrust-rate weight `W_df = 1e-3` is
fixed and not searched. The two objectives are the minimum lap time (with a
penalty for gate-safety, completion, body-rate, and jerk violations) and the
worst gate-crossing offset over the seeds (unpenalized).

`pareto_fronts.csv` lists every non-dominated configuration of the final
stage: 12 MPCC and 11 DQ-MPCC points out of 100 trials per controller.

## 2. Screening (five runs)

For each controller, the ten fastest front configurations whose worst tuning
offset satisfies `delta_max <= R_safe = 0.30 m` are re-evaluated over five
full-path software-in-the-loop runs on the onboard computer with physically
solid gates. Candidates completing all five runs are retained, and the
fastest retained candidate is selected provisionally.

## 3. Validation (n = 30 within n = 100)

Because a five-run screening is underpowered, deployment further requires
`delta_max <= R_safe` across the n = 100 validation campaign. The n = 100
validation extends the corresponding n = 30 screening campaign and therefore
includes its first 30 chronological runs; a configuration that exceeds
`R_safe` at n = 30 therefore also exceeds it at n = 100.

## 4. Outcome

- The baseline's fastest five-run candidate (2.86 s) exceeds `R_safe` in 10 of
  its 93 completed runs and is rejected; the deployed baseline satisfies the rule.
- All seven non-dominated baseline configurations faster than the deployed
  gains exceed `R_safe`: six already at n = 30 (34.0-46.1 cm), and the seventh
  at n = 100 (48.8 cm). That seventh is also the only examined baseline
  configuration with a solid-gate contact during a timed racing lap (7/100,
  against none for either deployed configuration).
- The deployed DQ-MPCC gains, two further DQ-MPCC front configurations, and the
  exact inverse-left-Jacobian variant at the deployed gains pass both stages.
- Selected: **MPCC trial 76** and **DQ-MPCC trial 46** (`selected = true`).

`R_safe = R_gate - r_drone` is a geometric clearance, not an adjustable
tolerance.
