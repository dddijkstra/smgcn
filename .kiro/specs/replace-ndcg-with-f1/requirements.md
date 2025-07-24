# Requirements Document

## Introduction

This specification outlines the requirements for replacing the NDCG (Normalized Discounted Cumulative Gain) evaluation metric with F1 score in the SMGCN project. The current evaluation system uses NDCG@K as one of the key metrics alongside Precision@K, Recall@K, and RMRR. The goal is to replace NDCG with F1 score to provide a more balanced evaluation metric that considers both precision and recall.

## Requirements

### Requirement 1

**User Story:** As a researcher using the SMGCN model, I want the evaluation system to use F1 score instead of NDCG, so that I can get a more balanced assessment of the model's performance that equally weights precision and recall.

#### Acceptance Criteria

1. WHEN the model evaluation is performed THEN the system SHALL calculate F1@K scores instead of NDCG@K scores
2. WHEN F1@K is calculated THEN the system SHALL use the formula F1 = 2 * (precision * recall) / (precision + recall) for each K value
3. WHEN precision or recall is zero THEN the system SHALL return F1 score as 0 to avoid division by zero
4. WHEN the evaluation results are displayed THEN the system SHALL show F1@K values in place of NDCG@K values
5. WHEN the evaluation results are logged THEN the system SHALL record F1@K metrics instead of NDCG@K metrics

### Requirement 2

**User Story:** As a developer maintaining the SMGCN codebase, I want the F1 calculation to be integrated seamlessly into the existing evaluation framework, so that the code remains clean and maintainable.

#### Acceptance Criteria

1. WHEN the batch_test.py module is updated THEN the system SHALL replace ndcg_at_k function with f1_at_k function
2. WHEN the test function is called THEN the system SHALL calculate F1 scores using the same K values as other metrics [5, 10, 15, 20]
3. WHEN the evaluation loop processes results THEN the system SHALL maintain the same data structure format but with F1 values
4. WHEN the model training loop logs metrics THEN the system SHALL display F1 scores with the same formatting as other metrics
5. WHEN the final results are saved THEN the system SHALL write F1 scores to the output file instead of NDCG scores

### Requirement 3

**User Story:** As a user running the SMGCN training script, I want the output format to remain consistent, so that I can easily interpret the results and compare them with previous runs.

#### Acceptance Criteria

1. WHEN the training progress is displayed THEN the system SHALL show F1 scores in the same position where NDCG was previously shown
2. WHEN the final performance summary is printed THEN the system SHALL include F1 scores with clear labeling
3. WHEN results are written to files THEN the system SHALL use "f1" as the metric name instead of "ndcg"
4. WHEN the verbose output is enabled THEN the system SHALL display F1 scores alongside precision, recall, and RMRR
5. WHEN the best iteration results are reported THEN the system SHALL include the best F1 scores achieved

### Requirement 4

**User Story:** As a researcher comparing different models, I want the F1 calculation to be mathematically correct and consistent, so that I can trust the evaluation results for academic or research purposes.

#### Acceptance Criteria

1. WHEN F1@K is calculated THEN the system SHALL use the precision and recall values already computed for the same K
2. WHEN multiple test samples are processed THEN the system SHALL average F1 scores across all test cases
3. WHEN F1 scores are computed THEN the system SHALL handle edge cases where precision + recall = 0
4. WHEN the evaluation completes THEN the system SHALL ensure F1 scores are in the range [0, 1]
5. WHEN F1 calculation encounters numerical issues THEN the system SHALL handle them gracefully without crashing