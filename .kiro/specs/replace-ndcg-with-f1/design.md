# Design Document

## Overview

This design document outlines the implementation approach for replacing NDCG (Normalized Discounted Cumulative Gain) with F1 score in the SMGCN evaluation system. The change involves modifying the evaluation pipeline to calculate F1@K metrics instead of NDCG@K while maintaining the existing code structure and output format.

## Architecture

The evaluation system follows a modular architecture where metrics are calculated in `utils/batch_test.py` and consumed in `smgcn_main.py`. The current flow is:

1. **Model Inference**: Generate predictions for test data
2. **Metric Calculation**: Calculate precision, recall, NDCG, and RMRR for each K value
3. **Result Aggregation**: Average metrics across all test samples
4. **Output Generation**: Display and save results

The modified architecture will replace the NDCG calculation step with F1 calculation while keeping all other components unchanged.

## Components and Interfaces

### 1. Metric Calculation Module (`utils/batch_test.py`)

#### Current Interface:
```python
def test(model, users_to_test, test_group_list, drop_flag=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)), 'rmrr': np.zeros(len(Ks))}
    # ... calculation logic ...
    return result
```

#### Modified Interface:
```python
def test(model, users_to_test, test_group_list, drop_flag=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)),
              'f1': np.zeros(len(Ks)), 'rmrr': np.zeros(len(Ks))}
    # ... calculation logic ...
    return result
```

#### New F1 Calculation Function:
```python
def f1_at_k(precision, recall):
    """
    Calculate F1 score from precision and recall values
    Args:
        precision: Precision value
        recall: Recall value
    Returns:
        F1 score (0 if precision + recall = 0)
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
```

### 2. Training Loop Module (`smgcn_main.py`)

#### Current Variables:
- `ndcg_loger`: List to store NDCG values across epochs
- `ret['ndcg']`: NDCG results from test function

#### Modified Variables:
- `f1_loger`: List to store F1 values across epochs  
- `ret['f1']`: F1 results from test function

#### Output Format Changes:
- Replace "ndcg" with "f1" in performance strings
- Update final result formatting to show F1 scores
- Modify file output to write F1 instead of NDCG

### 3. Evaluation Loop Logic

#### Current Calculation Flow:
```python
for ii in range(len(topN)):
    # Calculate precision and recall
    precision_n[ii] = precision_n[ii] + float(number / topN[ii])
    recall_n[ii] = recall_n[ii] + float(number / len(v))
    # Calculate NDCG
    ndcg_n[ii] = ndcg_n[ii] + ndcg_at_k(r, topN[ii])
    # Calculate RMRR
    rmrr_n[ii] = rmrr_n[ii] + mrr_score / len(v)
```

#### Modified Calculation Flow:
```python
for ii in range(len(topN)):
    # Calculate precision and recall (unchanged)
    current_precision = float(number / topN[ii])
    current_recall = float(number / len(v))
    precision_n[ii] = precision_n[ii] + current_precision
    recall_n[ii] = recall_n[ii] + current_recall
    # Calculate F1 using precision and recall
    f1_n[ii] = f1_n[ii] + f1_at_k(current_precision, current_recall)
    # Calculate RMRR (unchanged)
    rmrr_n[ii] = rmrr_n[ii] + mrr_score / len(v)
```

## Data Models

### Evaluation Result Structure

#### Before:
```python
result = {
    'precision': np.array([p@5, p@10, p@15, p@20]),
    'recall': np.array([r@5, r@10, r@15, r@20]),
    'ndcg': np.array([ndcg@5, ndcg@10, ndcg@15, ndcg@20]),
    'rmrr': np.array([rmrr@5, rmrr@10, rmrr@15, rmrr@20])
}
```

#### After:
```python
result = {
    'precision': np.array([p@5, p@10, p@15, p@20]),
    'recall': np.array([r@5, r@10, r@15, r@20]),
    'f1': np.array([f1@5, f1@10, f1@15, f1@20]),
    'rmrr': np.array([rmrr@5, rmrr@10, rmrr@15, rmrr@20])
}
```

### Logging Data Structure

#### Before:
```python
ndcg_loger = []  # List of NDCG arrays across epochs
ndcgs = np.array(ndcg_loger)  # Final NDCG results
```

#### After:
```python
f1_loger = []  # List of F1 arrays across epochs
f1s = np.array(f1_loger)  # Final F1 results
```

## Error Handling

### Division by Zero Protection
The F1 calculation includes protection against division by zero when precision + recall = 0:

```python
def f1_at_k(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
```

### Numerical Stability
- Ensure F1 scores remain in [0, 1] range
- Handle edge cases where precision or recall might be NaN
- Maintain consistent data types (float) throughout calculations

### Graceful Degradation
- If F1 calculation fails, log error and continue with other metrics
- Provide meaningful error messages for debugging
- Ensure the evaluation doesn't crash due to F1 calculation issues

## Testing Strategy

### Unit Tests
1. **F1 Calculation Function**:
   - Test with normal precision/recall values
   - Test with zero precision or recall
   - Test with precision + recall = 0
   - Test edge cases (precision=1, recall=1)

2. **Integration Tests**:
   - Test full evaluation pipeline with F1 calculation
   - Verify F1 scores are correctly averaged across test samples
   - Ensure output format matches expected structure

### Validation Tests
1. **Mathematical Correctness**:
   - Verify F1 = 2 * (P * R) / (P + R) for known values
   - Compare F1 results with manual calculations
   - Test with synthetic data with known ground truth

2. **Consistency Tests**:
   - Ensure F1 scores are consistent across multiple runs
   - Verify F1 values are in expected range [0, 1]
   - Test with different K values [5, 10, 15, 20]

### Performance Tests
1. **Execution Time**:
   - Measure F1 calculation overhead compared to NDCG
   - Ensure no significant performance degradation
   - Profile memory usage during evaluation

2. **Scalability**:
   - Test with large test datasets
   - Verify performance with different batch sizes
   - Ensure consistent behavior across different hardware

## Implementation Plan

### Phase 1: Core F1 Function
- Implement `f1_at_k()` function with error handling
- Add unit tests for the F1 calculation
- Validate mathematical correctness

### Phase 2: Evaluation Pipeline Integration
- Modify `test()` function in `batch_test.py`
- Replace NDCG calculation with F1 calculation
- Update result dictionary structure

### Phase 3: Training Loop Updates
- Update variable names in `smgcn_main.py`
- Modify logging and output formatting
- Update performance string generation

### Phase 4: Testing and Validation
- Run comprehensive tests with existing datasets
- Validate F1 scores against manual calculations
- Ensure backward compatibility of output format

### Phase 5: Documentation Updates
- Update code comments and docstrings
- Modify any configuration or help text
- Update example outputs in documentation