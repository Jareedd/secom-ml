# Phase 6: final chronological holdout

Training: 1253 runs, 87 failures (2008-07-19 11:55:00 through 2008-10-02 19:25:00).
Test: 314 runs, 17 failures (2008-10-02 20:54:00 through 2008-10-17 06:07:00).

**Primary calibrated xgboost, 10:1:** saving -5.9% versus never alarm. 95% paired IID interval -12.5% to -2.4%; 20-run circular-block sensitivity interval -13.7% to -3.1%.

## Ranking and probabilities

| Model | AP (PR AUC) | ROC AUC | Brier | Mean prediction | Actual failure rate |
|---|---:|---:|---:|---:|---:|
| calibrated xgboost (primary) | 0.0704 | 0.5617 | 0.0519 | 6.0% | 5.4% |
| random forest (secondary) | 0.0880 | 0.5631 | 0.0522 | 7.3% | 5.4% |

## Fixed cost policies

Costs per 1,000 runs, in units of one false-alarm inspection. Correct decisions cost zero; these costs are assumptions, not fab measurements.

| Model | Ratio | Threshold | Caught / failures | False alarms | Misses | Policy cost | Never alarm | 0.5 cutoff | Inspect all | Saving vs best naive (95% IID) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| calibrated xgboost | 5:1 | 16.67% | 0/17 | 4 | 17 | 283.4 | 270.7 | 270.7 | 945.9 | -4.7% (-11.8% to -0.8%) |
| calibrated xgboost | 10:1 | 9.09% | 0/17 | 10 | 17 | 573.2 | 541.4 | 541.4 | 945.9 | -5.9% (-12.5% to -2.4%) |
| calibrated xgboost | 20:1 | 4.76% | 17/17 | 297 | 0 | 945.9 | 1082.8 | 1082.8 | 945.9 | +0.0% (-52.0% to +0.0%) |
| calibrated xgboost | 50:1 | 1.96% | 17/17 | 297 | 0 | 945.9 | 2707.0 | 2707.0 | 945.9 | +0.0% (+0.0% to +0.0%) |
| random forest | 5:1 | 16.67% | 0/17 | 4 | 17 | 283.4 | 270.7 | 270.7 | 945.9 | -4.7% (-12.0% to -0.9%) |
| random forest | 10:1 | 9.09% | 7/17 | 67 | 10 | 531.8 | 541.4 | 541.4 | 945.9 | +1.8% (-38.3% to +28.4%) |
| random forest | 20:1 | 4.76% | 15/17 | 237 | 2 | 882.2 | 1082.8 | 1082.8 | 945.9 | +6.7% (-41.1% to +21.4%) |
| random forest | 50:1 | 1.96% | 17/17 | 297 | 0 | 945.9 | 2707.0 | 2707.0 | 945.9 | +0.0% (+0.0% to +0.0%) |

## Interpretation limits

- Model roles, costs, thresholds and no drift correction were fixed before final predictions. No threshold search or promotion of the secondary model is allowed.
- The cheaper naive comparator is recalculated on each bootstrap sample. It is a retrospective benchmark using observed prevalence, not a deployable oracle.
- 2000 paired resamples were requested per interval. Counts retained after excluding single-class resamples are in the CSVs. Intervals condition on fitted models and this period; they omit training/selection uncertainty and cannot guarantee performance under future drift.
- IID resampling assumes independent runs. Circular blocks of 20 consecutive runs give a predeclared dependence sensitivity check, with an artificial end-to-start join; block length is not estimated.
- Brier references in the metrics CSV distinguish the deployable training prevalence forecast from the retrospective test-prevalence reference.
- Earlier loaders parsed the complete files before taking the training slice; aggregate test counts were already reported. Test predictions were first evaluated in this phase. The shared sensor list uses all training covariates, so earlier CV is not completely fold-isolated; the chronological final holdout remains excluded from every fit and sensor choice.

Regenerate this report without refitting: `python -m secom_ml.final_test --report-only`.
