# Design Document Revision Guide

This note is a rewrite guide for the contest design document. It is intentionally stricter than a normal project summary.

## High-Risk Claims To Soften

Replace or tighten the following types of wording:

- `near-perfect`, `perfect`, `almost no cost`
- `can be directly deployed on STM32`
- `microsecond-level` unless backed by measured numbers in code or tables
- `indispensable` or `irreplaceable` unless supported by multiple ablations

Recommended style:

- state the exact metric
- state the exact scenario
- state the exact limitation

Example:

- weak: `the method can almost perfectly classify traffic`
- better: `the method achieved 99.93% accuracy on AP-Case2, while the GCS scenario remained challenging due to severe class imbalance and feature overlap`

## Section-Level Rewrite Priorities

### 1. Problem Fit

Do not overstate that the project is already a V2X solution. Present it as:

- a UAV security system used as the verified heterogeneous edge-security scenario
- a method with potential transfer value to A-ICV communication settings

This avoids a strong reviewer asking why all experiments are on UAV datasets rather than vehicle-road datasets.

### 2. Results

Reduce emphasis on raw `Accuracy`. Increase emphasis on:

- `F1-Macro`
- `F1-Normal`
- `F1-Attack`
- `AUC-ROC`
- `PR-AUC`
- class imbalance interpretation

For GCS, explicitly state:

- the attack-detection objective was prioritized
- the normal-class false-positive rate remains a limitation
- the system is therefore better suited to high-recall alerting plus downstream review

Do not frame this as “not a model problem.” That sounds defensive.

### 3. Innovation

Keep the innovation section focused on four points:

- heterogeneous feature projection
- frozen transfer strategy under limited labels
- lightweight edge-oriented backbone
- online explanation and visualization pipeline

Avoid inflating routine engineering choices into standalone innovation claims.

### 4. Deployment

Be precise about what is actually implemented:

- simulated streaming inference is implemented
- gateway and dashboard are implemented
- static online demo is available

Do not imply that a field deployment on real UAV hardware has already been completed unless you have direct evidence.

## Numbers That Must Be Unified

Before exporting the final PDF, verify that the same numbers are used consistently across:

- design document tables
- video narration
- README / submission guide
- plots and csv files
- script comments or printed summaries

Priority checks:

- source-domain validation accuracy
- GCS validation/test accuracy
- GCS class weights
- CPU latency
- total parameter count

## Reviewer-Facing Rewrite Rule

Every strong claim should satisfy one of the following:

- visible in a result table
- reproducible from a script
- demonstrated in the video
- directly shown in the interface

If not, remove it or rewrite it as a future direction.
