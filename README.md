# Predictive Partition Placement in Flash Crowds

### AI-Assisted Workload Forecasting and Latency-Aware Placement for Distributed Databases

<p align="center">
  <strong>Predictive Systems · Distributed Databases · Workload Forecasting · Performance Optimization</strong>
</p>

---

## Overview

**Predictive Partition Placement in Flash Crowds** is a research project investigating how workload forecasting and cross-layer cost estimation can improve partition placement decisions in distributed database systems during sudden workload surges.

Distributed databases must distribute data and transaction processing across multiple nodes while balancing network latency, processing capacity, queueing delays, transaction coordination overhead, and data movement costs. Under flash-crowd conditions, rapidly changing request volumes can make decisions based solely on current system state ineffective.

This project investigates a predictive alternative: estimate future system conditions and use those estimates to guide placement decisions before congestion becomes severe.

The research combines workload traces, network-topology information, forecasting models, and a cost-based placement framework to evaluate predictive placement against reactive and baseline strategies.

> **Research focus:** Can forecasting future workload and system conditions improve transaction and partition placement decisions under flash-crowd workloads, and under what conditions does prediction provide a measurable benefit?

---

## Table of Contents

* [Problem Statement](#problem-statement)
* [Research Objectives](#research-objectives)
* [Core Approach](#core-approach)
* [System Architecture](#system-architecture)
* [Placement Strategies](#placement-strategies)
* [Cost Model](#cost-model)
* [Datasets and Experimental Inputs](#datasets-and-experimental-inputs)
* [Experimental Methodology](#experimental-methodology)
* [Evaluation Metrics](#evaluation-metrics)
* [Repository Structure](#repository-structure)
* [Technology and Research Components](#technology-and-research-components)
* [Results and Interpretation](#results-and-interpretation)
* [Reproducibility](#reproducibility)
* [Limitations and Future Work](#limitations-and-future-work)
* [Research Significance](#research-significance)

---

## Problem Statement

Flash crowds are sudden increases in demand that can overload particular services, partitions, or database nodes. In distributed databases, these surges can cause:

* Increased transaction latency and tail-latency degradation.
* Uneven workload distribution across nodes.
* CPU utilization and request-queue growth.
* Higher network communication costs.
* Increased distributed transaction coordination overhead.
* Additional costs when moving or redistributing data.

Traditional reactive strategies make placement decisions using the system state observed at decision time. Although useful, these approaches may respond only after a workload surge has already affected system performance.

This project investigates whether predicting upcoming workload and resource conditions can enable more informed placement decisions.

Rather than optimizing a single metric in isolation, the proposed research considers the interaction between workload demand, network characteristics, transaction coordination, queueing effects, and placement-related overhead.

### Central Research Question

**Does predictive placement outperform reactive placement under flash-crowd conditions when both strategies are evaluated under comparable workloads and system assumptions?**

The answer must be established experimentally. Better forecasting accuracy alone does not guarantee better end-to-end database performance.

---

## Research Objectives

The project is organized around the following objectives:

1. **Workload characterization:** Study workload patterns and demand variations using trace-driven experimental inputs.
2. **Predictive forecasting:** Investigate regression-based forecasting, including Ridge regression, to estimate future workload or system conditions.
3. **Cross-layer cost estimation:** Model the effects of network latency, transaction coordination, queueing, and placement changes.
4. **Predictive placement:** Investigate how forecasted conditions can guide placement decisions.
5. **Baseline comparison:** Compare predictive strategies with reactive and static placement policies.
6. **Horizon analysis:** Evaluate how the prediction horizon affects placement decisions and performance.
7. **Performance evaluation:** Analyze latency distributions, tail latency, aggregate performance, and workload-dependent trade-offs.
8. **Reproducibility:** Preserve experiment configurations, results, tables, figures, and research documentation for further analysis.

---

## Core Approach

The research follows a forecast–estimate–place–evaluate workflow.

### 1. Workload observation

Historical workload information is used to characterize demand and identify changes relevant to placement decisions.

### 2. Future-state forecasting

A forecasting model estimates future conditions from available historical information. Ridge regression is one of the predictive methods investigated in the project.

### 3. Cross-layer cost estimation

Candidate placement decisions are evaluated using a cost model that considers relevant network, transaction-processing, queueing, and placement-related factors.

### 4. Placement selection

The predictive policy selects a placement based on estimated future costs rather than relying exclusively on instantaneous observations.

### 5. Experimental evaluation

The resulting policy is compared with alternative placement strategies under defined workloads and prediction horizons.

The central hypothesis is that forecasting can provide a useful decision advantage when workload changes are sufficiently predictable and the forecast is relevant to the placement decision.

---

## System Architecture

The research framework can be understood through the following logical components.

```text
             Historical Workload Traces
                        |
                        v
              Workload Characterization
                        |
                        v
               Predictive Forecasting
                 (Ridge Regression)
                        |
                        v
              Future-State Estimation
                        |
                        v
              Cross-Layer Cost Model
             /          |           \
            /           |            \
     Network Cost   Queueing Cost   Transaction Cost
            \           |            /
             \          |           /
                        v
                Placement Policy
                        |
                        v
              Candidate Placement
                        |
                        v
               Experimental Evaluation
                        |
                        v
            Metrics, Tables and Figures
```

This diagram represents the research workflow rather than a claim that every component operates as a continuously deployed production service.

### Architectural components

| Component          | Responsibility                                                        |
| ------------------ | --------------------------------------------------------------------- |
| Workload inputs    | Represent request demand and workload variation.                      |
| Network model      | Represents communication latency between relevant locations or nodes. |
| Forecasting model  | Estimates future workload or system conditions.                       |
| Cost model         | Combines relevant performance and placement costs.                    |
| Placement policies | Implements alternative strategies for making placement decisions.     |
| Experiment runner  | Executes configured experiments and records outcomes.                 |
| Analysis pipeline  | Produces metrics, comparisons, tables, and visualizations.            |

---

## Placement Strategies

The project investigates multiple placement policies to distinguish the effects of prediction from those of ordinary placement decisions.

### Static placement

Uses a predetermined placement strategy without adapting decisions to forecasted future conditions.

This provides a baseline for evaluating the benefits and costs of more adaptive policies.

### Reactive placement

Uses currently observed conditions to guide placement decisions.

This strategy represents a conventional alternative to forecasting future conditions.

### Predictive placement

Uses forecasted conditions and the placement cost model to guide decisions.

The objective is to anticipate workload changes and select placements that reduce expected future cost.

### Oracle placement

An oracle policy can serve as a reference if it is supplied with future information unavailable to the other policies.

It provides an indication of the potential benefit of perfect or privileged information. Oracle results should not be interpreted as achievable production performance unless the same information is available to the deployed system.

**Important:** Comparisons are meaningful only when the policies use consistent workloads, comparable constraints, and clearly documented assumptions.

---

## Cost Model

The placement problem is formulated as a cost-minimization problem.

Let:

* \(p\) denote a candidate placement.
* \(t\) denote the current decision time.
* \(H\) denote the prediction horizon.
* \(\hat{s}(t+H)\) denote the estimated future system state.
* \(C_{\mathrm{net}}\) denote network communication cost.
* \(C_{\mathrm{txn}}\) denote transaction coordination cost.
* \(C_{\mathrm{queue}}\) denote queueing and processing cost.
* \(C_{\mathrm{move}}\) denote the cost associated with changing placement.

A conceptual objective is:

$$
p^* =
\underset{p \in \mathcal{P}}{\arg\min}
\ J(p \mid \hat{s}(t+H))
$$

where \(\mathcal{P}\) is the set of feasible candidate placements.

One possible decomposition of the objective is:

$$
J(p) =
\alpha C_{\mathrm{net}}(p)
+\beta C_{\mathrm{txn}}(p)
+\gamma C_{\mathrm{queue}}(p)
+\delta C_{\mathrm{move}}(p)
$$

Here, \(\alpha,\beta,\gamma,\delta\) represent weights assigned to the different cost components.

This expression describes the general optimization principle. The exact implementation, units, parameter values, and included terms must be taken from the project's configuration and source code.

### Network communication cost

Network cost captures the communication overhead associated with accessing distributed resources.

Network-topology information can be used to represent latency differences between locations. The project investigates the Abilene network topology as a source of network-latency information.

### Transaction coordination cost

Distributed transactions can require communication between participating nodes. Two-phase commit (2PC) is relevant because its coordination overhead can increase transaction latency when communication spans distant or congested nodes.

### Queueing and processing cost

A placement decision must consider the processing load at candidate nodes.

Queueing models, including the M/M/1 model where its assumptions are appropriate, provide a way to reason about how increasing utilization can increase expected waiting time.

### Placement-change cost

Moving or redistributing data can introduce additional communication, processing, and operational overhead.

A placement policy that minimizes immediate latency without accounting for movement costs may make decisions that are inefficient over a longer period.

---

## Datasets and Experimental Inputs

The project investigates trace-driven workload analysis and network-aware placement.

| Input                      | Role in the research                                                             |
| -------------------------- | -------------------------------------------------------------------------------- |
| Alibaba Cluster Trace 2018 | Workload-related trace data for studying demand patterns and workload variation. |
| Abilene network topology   | Network-topology input for modeling communication costs.                         |
| Experiment configurations  | Define the parameters and conditions under which experiments are evaluated.      |
| Generated result artifacts | Support comparisons across policies, workloads, and prediction horizons.         |

The presence of a dataset in the research framework does not imply that every experiment uses every dataset. Refer to the experiment configurations and data documentation to identify the inputs used for a particular result.

Large datasets are intentionally excluded from the Git repository through `.gitignore`. This keeps the repository manageable while allowing the research code and documentation to be version-controlled.

---

## Experimental Methodology

The evaluation is organized around comparisons between placement strategies under controlled experimental conditions.

### Policy comparison

Compare static, reactive, predictive, and oracle policies where applicable.

### Prediction-horizon analysis

Evaluate how the interval between the current observation and the forecasted future state affects placement decisions.

Shorter horizons may better capture immediate changes, while longer horizons may offer more time to react but introduce greater forecasting uncertainty.

### Workload variation

Evaluate the policies across relevant workload patterns, including conditions intended to represent flash-crowd behavior.

### Cross-layer cost evaluation

Examine how network communication, transaction coordination, queueing, and placement-related costs influence the resulting decisions.

### Result analysis

Aggregate experiment outputs into tables and visualizations, and examine both overall performance and individual workload behavior.

A robust comparison should use consistent experimental settings and report the conditions under which each policy succeeds or fails.

---

## Evaluation Metrics

The following metrics are relevant to assessing predictive placement.

| Metric                    | Purpose                                                                        |
| ------------------------- | ------------------------------------------------------------------------------ |
| P50 latency               | Measures median transaction latency.                                           |
| P95 latency               | Measures latency experienced by 95% of observations or less.                   |
| P99 latency               | Captures tail-latency behavior and slow transactions.                          |
| Aggregate latency or cost | Supports comparison of overall performance under a defined aggregation method. |
| Forecast error            | Measures how closely predictions match observed future values.                 |
| Workload throughput       | Measures completed transactions or requests per unit time, where recorded.     |
| Placement overhead        | Captures costs associated with placement decisions or changes, where measured. |

The exact metric definitions and aggregation procedures should be checked against the relevant experiment scripts and result files.

Forecast accuracy and placement performance should be analyzed separately: a model can predict system conditions accurately without producing better placement decisions.

---

## Repository Structure

The repository is organized around implementation code, configurations, experiments, results, and research documentation.

```text
Predictive-Partition-Placement-in-Flash-Crowds/
│
├── configs/              # Experiment and model configurations
├── data/                 # Local datasets; excluded from Git
├── figures/               # Research figures and visualizations
├── logs/                  # Local experiment logs; excluded from Git
├── manuscript/            # Research manuscript and related documents
├── results/               # Experiment outputs
├── results_frozen/        # Preserved result artifacts
├── scripts/               # Experiment and utility scripts
├── snapshots/             # Preserved project snapshots
├── src/                   # Source implementation
├── tables/                # Research tables and summaries
├── tests/                 # Test suite
├── .gitignore             # Git exclusion rules
├── DATA_LOG.md            # Dataset-related documentation
├── ENV.md                 # Environment documentation
├── PREREG.md              # Preregistration documentation
├── PREREG.sha256          # Preregistration checksum
└── README.md               # Project overview
```

This is a high-level overview. Consult the actual directory contents for the complete list of files and the responsibilities of individual modules.

---

## Technology and Research Components

The project brings together the following technical areas:

* **Python:** Research implementation and experiment tooling.
* **Regression-based forecasting:** Ridge regression for predictive modeling.
* **Distributed databases:** Transaction processing, distributed communication, and placement decisions.
* **Network modeling:** Topology-aware communication-cost estimation.
* **Queueing theory:** Analysis of load and waiting-time effects.
* **Experimental analysis:** Metrics, result aggregation, tables, and visualizations.
* **Version control:** Git and GitHub for source and research-artifact management.

The exact package versions, installation requirements, and supported runtime environment should be taken from the project's environment documentation and dependency files.

---

## Results and Interpretation

The project evaluates whether predictive placement offers measurable advantages over alternative strategies.

An important distinction is made between:

1. **Forecasting quality:** How accurately the model predicts future conditions.
2. **Decision quality:** Whether those predictions lead to better placement choices.
3. **System performance:** Whether the resulting choices improve end-to-end latency or other operational metrics.

These are related but distinct outcomes.

A predictive policy may perform well under a workload that follows recognizable patterns but offer little benefit under an unpredictable surge. Prediction errors, model overhead, inaccurate cost estimates, and placement-change costs can offset the benefits of forecasting.

For this reason, the research should report policy comparisons by workload and metric rather than assuming that predictive placement always outperforms reactive placement.

The repository's result files, tables, and figures are the authoritative sources for individual experimental outcomes.

---

## Reproducibility

The repository preserves implementation code, configurations, experiment outputs, and research documentation to support reproducible investigation.

To begin working with the project:

1. Clone the repository.
2. Review `ENV.md` for environment setup instructions.
3. Review `DATA_LOG.md` to understand the dataset requirements and available data.
4. Inspect `configs/` to identify the parameters used by the experiments.
5. Inspect `scripts/` and `src/` to identify the experiment entry points and implementation components.
6. Review `tests/` for the available validation procedures.
7. Examine `results/`, `results_frozen/`, `figures/`, and `tables/` to understand the preserved experimental outputs.
8. Consult `manuscript/` for the research formulation and experimental discussion.

### Data availability

Large local datasets are excluded from version control. Cloning the repository therefore does not necessarily provide every input required to reproduce all experiments.

Users should consult the dataset documentation and obtain the required data independently, subject to the relevant dataset's access conditions and licensing.

### Reproducibility considerations

For meaningful comparisons, record:

* Dataset and workload version.
* Experiment configuration.
* Forecasting model and prediction horizon.
* Placement policy.
* Cost-model parameters.
* Software environment.
* Evaluation metrics and aggregation method.

Exact execution commands should be taken from the repository's current scripts and environment instructions.

---

## Limitations and Future Work

Several research directions remain important for establishing the practical value of predictive placement.

### Forecast robustness

Investigate how prediction errors affect placement decisions during sudden, irregular, or previously unseen workload surges.

### Cost-model calibration

Assess whether estimated network, transaction, queueing, and movement costs accurately reflect observed behavior.

### Prediction-horizon selection

Determine whether different workloads benefit from different forecast horizons and whether the horizon should adapt dynamically.

### Placement stability

Investigate whether prediction-driven decisions cause unnecessary movement or frequent placement changes.

### Generalization

Evaluate whether conclusions hold across different workload patterns, network topologies, and distributed database configurations.

### End-to-end validation

Validate simulated or trace-driven findings against actual execution in a distributed database environment.

### Operational overhead

Measure the computational and communication costs introduced by forecasting and decision-making.

These directions are essential for determining whether improvements observed in controlled experiments translate into practical system-level benefits.

---

## Research Significance

This project explores the intersection of predictive analytics, distributed database optimization, and workload-aware placement.

Its central contribution is the investigation of **using forecasted future conditions, rather than only current observations, to inform placement decisions under flash-crowd workloads**.

The research emphasizes that predictive systems must be evaluated by the quality of the decisions they produce, not merely by the accuracy of their forecasts.

By examining multiple placement policies, prediction horizons, and cross-layer cost components, the project aims to identify when predictive placement is beneficial, when it is ineffective, and what factors determine the difference.

---

## Repository

**Project:** Predictive Partition Placement in Flash Crowds

**GitHub:** [Predictive-Partition-Placement-in-Flash-Crowds](https://github.com/Tuhin-Samantray/Predictive-Partition-Placement-in-Flash-Crowds)

---

*This repository is a research prototype. Experimental findings should be interpreted in the context of the documented workloads, configurations, assumptions, and evaluation procedures.*
