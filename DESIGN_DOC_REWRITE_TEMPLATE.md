# Design Document Rewrite Template

This template is intended to replace high-risk wording in the current contest design document. It is not the final formatted submission, but a safer narrative scaffold.

## 1. Target Problem And Value

This work targets intrusion detection in heterogeneous UAV communication environments. The system is designed for three representative deployment points: UAV nodes, access points, and the ground control station. These deployment points differ significantly in protocol layer, feature dimension, and data distribution, which makes direct model reuse difficult.

The project focuses on three practical constraints:

- heterogeneous input feature spaces across deployment points
- limited labeled samples in newly deployed target scenarios
- strict latency and parameter budgets for edge deployment

The proposed method is validated on UAV communication security data, while the underlying transfer mechanism is relevant to broader heterogeneous edge-security settings, including intelligent connected transportation systems where cross-node communication also exhibits protocol and feature heterogeneity.

## 2. Design Rationale

Instead of training a separate deep model from scratch for each target scenario, the project adopts a two-stage transfer strategy:

1. learn generic traffic anomaly representations from a large source-domain dataset
2. adapt to each target scenario through a lightweight domain-specific projection module while freezing the shared feature extractor

This design reduces dependence on large labeled target datasets and avoids the instability of full-parameter retraining under limited supervision.

## 3. Core Method

The method contains three main modules:

- `DomainProjector`: maps source-domain or target-domain input features of different dimensions into a shared latent space
- `SharedBackbone`: extracts compact traffic representations from the projected features
- `Classifier`: outputs binary intrusion decisions

In the target-domain stage, only the projector and classifier are updated, while the shared backbone remains frozen. This is intended to preserve source-domain traffic knowledge and reduce catastrophic forgetting.

## 4. Implementation And System Pipeline

The system includes both model training and deployment-side components:

- source-domain pretraining
- target-domain finetuning
- ablation and baseline evaluation
- few-shot analysis
- real-time inference simulation
- FastAPI gateway for alert aggregation
- Vue-based monitoring dashboard for visualization

The deployed demonstration pipeline shows how edge inference results can be reported to a gateway and visualized as latency curves, threat trends, explanation views, and alert logs.

## 5. Results Presentation Strategy

Use the following rule throughout the document:

- use `Accuracy` only as a supporting metric
- use `F1-Macro`, `F1-Normal`, `F1-Attack`, `AUC-ROC`, and `PR-AUC` as the main metrics

Suggested wording for the difficult GCS scenario:

The GCS scenario is substantially more challenging than the UAV and AP scenarios because of severe class imbalance and stronger feature overlap between normal and attack samples. Under this condition, the current model configuration prioritizes attack recall to reduce missed detections, at the cost of a relatively high false-positive rate on the normal class. This makes the system more suitable for high-recall early warning followed by downstream review.

## 6. Innovation Section

Keep the innovation claims limited to the following:

1. a learnable projection mechanism for heterogeneous feature spaces
2. a frozen-transfer finetuning strategy for low-label target domains
3. a lightweight edge-oriented detection backbone
4. a complete inference-to-visualization demonstration chain

Do not separately elevate ordinary engineering choices unless they are directly supported by experiments.

## 7. Summary Section

Suggested summary tone:

This work demonstrates that a lightweight heterogeneous transfer strategy can reduce target-domain adaptation cost while maintaining competitive detection performance across multiple UAV communication scenarios. Experimental results indicate that the method is especially useful when labeled target data are limited and deployment resources are constrained. At the same time, the GCS scenario reveals that class imbalance and feature overlap remain open challenges, which motivates future work on cost-sensitive learning and false-positive suppression.
