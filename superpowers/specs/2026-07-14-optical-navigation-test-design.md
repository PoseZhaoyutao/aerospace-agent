# Optical Navigation Test Design

## Scope and evidence boundary

This suite defines the acceptance boundary for optical navigation (camera line-of-sight and star observations) without claiming that the navigation estimator exists. The current `navigation_orbit_determination` package is `interface_only`; it may validate input contracts and return a `CapabilityGap`, but it must not produce an orbit or attitude solution.

## Contract cases

| ID | Case | Expected result now |
|---|---|---|
| ON-C01 | Valid observation: epoch, time scale, frame, camera, unit LOS, positive-semidefinite 2×2 angular covariance, provenance | Pass: Pydantic contract accepts it |
| ON-C02 | Zero/non-unit/non-finite LOS, non-positive, asymmetric, or non-positive-semidefinite covariance | Pass: contract rejects it |
| ON-C03 | Duplicate IDs or out-of-order epochs | Pass: request contract rejects it |
| ON-C04 | Orbit-determination request with fewer than three observations | Pass: request contract rejects it |
| ON-C05 | Extra fields or missing units/frame/time metadata | Pass: strict contract rejects it |

## Estimator acceptance cases (pending implementation)

1. Star-catalog matching: known-answer IDs, false matches, duplicate candidates, and confidence thresholding.
2. Attitude determination: TRIAD/Wahba known-answer rotation, covariance propagation, and frame handoff.
3. Bearing-only orbit determination: three-observation minimum, degenerate geometry rejection, and truth-state error thresholds.
4. Batch/recursive estimation: deterministic replay, outlier rejection, covariance positive-semidefinite checks, and checkpoint recovery.
5. Timing: UTC/TAI/TT conversion, exposure midpoint, latency, and epoch alignment.
6. Cross-validation: independent propagation/transform results with explicit position, velocity, and event-time tolerances.
7. Safety: unavailable estimator returns `CapabilityGap`; no fabricated state, tool call, or `ReviewResult` success is permitted.

The pending cases become executable only after a verified estimator, catalog dependency, and numerical truth-data fixtures are approved. Until then, the support chain may use `SpaceBasicTools` for time conversion, frame conversion, propagation, and cross-validation, but those tools do not constitute optical navigation.

## Fixtures and artifact policy

Use deterministic synthetic observations and truth states under `tests/fixtures/optical_navigation/` only after the estimator is implemented. Keep generated plots, databases, and model outputs under `.test-artifacts/` during diagnosis and delete them after successful runs.
