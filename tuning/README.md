# Tuning data

| File | Content |
|--|--|
| `pareto_fronts.csv` | All non-dominated points of the final tuning stage (12 MPCC, 11 DQ-MPCC) |
| `selected_gains.csv` | The deployed configurations: MPCC trial 76 and DQ-MPCC trial 46 |
| `selection_protocol.md` | Screening, nested n = 30 / n = 100 validation, and selection rule |

All controller weights are reported using the notation and pose-error
convention of the manuscript and the deployed controller.

## Columns

| Column | Manuscript symbol | Meaning |
|--|--|--|
| `controller` | | `MPCC` (baseline) or `DQ-MPCC` |
| `trial` | | Optuna trial number of the final stage |
| `selected` | | `true` for the deployed configuration |
| `lap_time_min_s` | `T_min` | Tuning objective 1 (minimum lap time, s) |
| `worst_gate_offset_m` | `delta_max^worst` | Tuning objective 2 (worst gate-crossing offset over seeds, m) |
| `q_attitude` | `q_q` (MPCC), `q_phi` (DQ-MPCC) | Attitude weight, `Q = q I_3` |
| `q_contour` | `q_c` | Contouring weight |
| `q_lag` | `q_l` | Lag weight |
| `w_body_rate` | `w_omega` | Body-rate tracking weight |
| `w_progress` | `w_s` | Progress reward `W_s` |
| `v_progress_max_mps` | `v_s,max` | Progress-speed bound (m/s) |
| `w_thrust_rate` | `W_df` | Thrust-rate weight, fixed at 1e-3 |

The tuning objectives are evaluated over three tuning seeds and are distinct
from the n = 100 validation statistics reported in the article
(`../results/paper_metrics.csv`).
