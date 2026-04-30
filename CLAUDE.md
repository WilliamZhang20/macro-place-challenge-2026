# Running Iterations

For a submission file in `file.py` you need to run

```
source ~/myenv/bin/activate # activate python libraries
evaluate file.py --all # for IBM benchmarks
evaluate file.py --ng45 # for NG45 benchmarks
```

# Relevant Files

- `external/DREAMPlace`
- `submissions/` - for examples and submitted files
- `README.md`

# Research Notes: Submission Alternatives

Goal: minimize average proxy cost while guaranteeing strict zero hard-macro overlaps. The strongest current evidence is that the IBM `initial.plc` placements already have excellent proxy but fail strict validation due to microscopic overlaps. Therefore, legalization quality and proxy-preserving repair dominate fancy global optimization.

## Current Best Baseline: `submissions/dccp_placer.py`

- Status: best known complete valid IBM sweep in this workspace.
- Observed full IBM result: average proxy `1.4556`, total overlaps `0`, runtime about `176.55s`.
- Strengths:
  - Very robust strict legalization.
  - Beats or matches RePlAce on several cases and slightly beats RePlAce average in this harness.
  - Handles troublesome cases such as `ibm02`, `ibm10`, `ibm12` well.
- Weaknesses:
  - Imports CVXPY/DCCP; canonicalization is slow and not elegant.
  - Circle packing is a relaxation, not exact rectangle placement.
  - Much of its benefit appears to be careful micro-overlap repair rather than true global optimization.
- Research interpretation:
  - Treat this as the target to beat, not as the architectural model to copy.
  - The useful idea is proxy-preserving legal repair; the DCCP machinery itself is suspect.

## DREAMPlace Branch: `submissions/dreamplace_moreau_placer.py`

- Status: useful proposal generator and fast analytical baseline, but not currently best as a direct submission.
- Key behavior:
  - L-BFGS-B Moreau-HPWL can reduce smooth wirelength, but legalization often gives back or worsens proxy.
  - With `lbfgs_iters=0` and tiny `overlap_gap`, its legalizer becomes a minimal-repair control.
- Observed tiny-gap minimal-repair samples:
  - `ibm01`: about `1.03849`, valid with tiny gap.
  - `ibm02`: about `1.56585`, valid after strict clamping; worse than DCCP `1.5604`.
  - `ibm06`: about `1.65928`, better than the earlier DCCP run `1.6710`.
  - `ibm09`: about `1.11262`, valid after strict clamping; near DCCP.
- Weaknesses:
  - Base legalizer only repairs hard macros; soft macro bounds need wrapper-level clamping.
  - Smooth HPWL pressure can damage density/congestion.
  - True-proxy oracle with a small budget gave only tiny improvements.
- Research interpretation:
  - DREAMPlace exploitation is promising only with true-proxy selection and a better density-aware legalizer.
  - Use it as a branch in the loop, not as an unconditional replacement for DCCP.

## DREAMPlace Exploit Wrapper: `submissions/dreamplace_exploit_placer.py`

- Status: experimental no-DCCP wrapper around `dreamplace_moreau_placer.py`.
- Design:
  - Candidate 1: tiny-gap minimal repair.
  - Other candidates: very anchored Moreau/DREAMPlace variants.
  - Strict validation and true proxy choose the winner.
- Partial sweep before user stopped it:
  - `ibm01`: `1.0385`, valid.
  - `ibm02`: `1.5658`, valid, worse than DCCP.
  - `ibm03`: `1.3255`, valid, close to DCCP.
  - `ibm04`: `1.3134`, valid, close to DCCP.
  - `ibm06`: `1.6593`, valid, better than earlier DCCP.
  - `ibm07`: `1.6659`, valid, much worse than DCCP.
  - `ibm08`: invalid in partial run, so wrapper fallback/validation is insufficient.
  - `ibm09`: `1.1127`, valid.
- Research interpretation:
  - Not submission-ready because `ibm08` invalidated.
  - Useful evidence: DREAMPlace minimal repair can beat DCCP on some cases but is unstable across all cases.
  - Next DREAMPlace work should first fix fallback validity, then run a full sweep only once.

## CasADi/IPOPT Branch: `submissions/casadi_placer.py`

- Status: experimental
- Design:
  - No CVXPY/DCCP.
  - Uses IPOPT with linearized rectangle separation cuts for overlapping pairs.
  - Finishes with deterministic axis-push legalization.
- Observed result:
  - `ibm01`: valid `1.0561`, worse than DCCP/minimal repair `1.0385`.
- Diagnosis:
  - Full-chip IPOPT moved macros at floorplan scale to satisfy cut constraints and wire terms.
  - That is too much movement for IBM, where initial overlaps are microscopic.
  - CasADi should not be used as a global solver here yet.
- Research interpretation:
  - CasADi may still be useful for small connected overlap components or local repair, not whole-chip placement.
  - Next CasADi experiment should solve only overlap components and minimize displacement, with no global wire term.
  
# Research Loop Guidance

Before implementing another branch, follow this loop:

1. Critique the current best and the new idea hard.
2. State the exact unknown being tested.
3. Run the smallest benchmark experiment that answers that unknown.
4. Compare against `dccp_placer.py`, DREAMPlace minimal repair, and strict initial proxy.
5. Only run `--all` after the candidate is valid on sampled cases and has a plausible path to average improvement.

Current highest-value next experiments:

- Build a no-DCCP minimal-overlap-component repairer: exact rectangle repair on connected overlap components, minimize displacement, strict bounds, no global wire objective.
- Fix `dreamplace_exploit_placer.py` fallback validity, especially `ibm08`, then run one full sweep.
- Try CasADi only on local overlap components, not full-chip.
- Add density-aware tie-breaking to legal repair; proxy losses are mostly density/congestion, not HPWL.
