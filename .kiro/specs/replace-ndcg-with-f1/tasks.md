# Implementation Plan

- [x] 1. Implement F1 calculation function
  - Create f1_at_k function with proper error handling for division by zero
  - Add input validation to ensure precision and recall are valid numbers
  - Include comprehensive docstring with usage examples
  - _Requirements: 1.2, 1.3, 4.3, 4.4_

- [ ] 2. Update batch_test.py evaluation logic
  - [x] 2.1 Replace NDCG variables with F1 variables
    - Change ndcg_n array initialization to f1_n
    - Update result dictionary to use 'f1' key instead of 'ndcg'
    - Modify variable names throughout the test function
    - _Requirements: 2.1, 2.3_

  - [x] 2.2 Integrate F1 calculation into evaluation loop
    - Replace ndcg_at_k function call with f1_at_k function call
    - Pass current precision and recall values to f1_at_k function
    - Ensure F1 calculation happens for each K value [5, 10, 15, 20]
    - _Requirements: 1.1, 2.2, 4.1, 4.2_

  - [ ] 2.3 Remove NDCG helper functions
    - Remove dcg_at_k function as it's no longer needed
    - Remove ndcg_at_k function as it's no longer needed
    - Clean up any unused imports related to NDCG calculation
    - _Requirements: 2.1_

- [-] 3. Update main training script (smgcn_main.py)
  - [x] 3.1 Replace NDCG logging variables
    - Change ndcg_loger to f1_loger for storing F1 scores across epochs
    - Update variable references from ret['ndcg'] to ret['f1']
    - Modify final results processing to use F1 arrays instead of NDCG arrays
    - _Requirements: 2.4, 3.3_

  - [x] 3.2 Update performance output strings
    - Modify verbose output format to display F1 scores instead of NDCG
    - Update final performance summary to show F1 results
    - Ensure consistent formatting with other metrics (precision, recall, RMRR)
    - _Requirements: 3.1, 3.2, 3.4_

  - [ ] 3.3 Update result file output
    - Change output file writing to include F1 scores instead of NDCG
    - Update final_perf string formatting to use F1 values
    - Ensure file output maintains the same structure with F1 replacing NDCG
    - _Requirements: 3.3, 3.5_

- [ ] 4. Add comprehensive error handling
  - [ ] 4.1 Handle edge cases in F1 calculation
    - Add protection against NaN values in precision or recall
    - Ensure F1 scores remain in valid range [0, 1]
    - Add logging for any numerical issues encountered
    - _Requirements: 4.3, 4.4, 4.5_

  - [ ] 4.2 Add graceful degradation for evaluation failures
    - Wrap F1 calculation in try-catch blocks
    - Provide meaningful error messages for debugging
    - Ensure evaluation continues even if F1 calculation fails for some samples
    - _Requirements: 4.5_

- [ ] 5. Create unit tests for F1 functionality
  - [ ] 5.1 Test F1 calculation function
    - Write tests for normal precision/recall values (e.g., 0.5, 0.7)
    - Test edge cases: precision=0, recall=0, both=0, both=1
    - Verify mathematical correctness: F1 = 2*P*R/(P+R)
    - Test with various K values to ensure consistency
    - _Requirements: 4.1, 4.2, 4.4_

  - [ ] 5.2 Test integration with evaluation pipeline
    - Create test data with known ground truth for F1 validation
    - Verify F1 scores are correctly averaged across test samples
    - Test with different batch sizes and dataset configurations
    - Ensure output format matches expected structure
    - _Requirements: 2.2, 2.3, 4.2_

- [ ] 6. Validate mathematical correctness
  - [ ] 6.1 Manual verification of F1 calculations
    - Create small test cases with manually calculated expected F1 scores
    - Run evaluation on test cases and compare results
    - Verify F1 scores match manual calculations within acceptable tolerance
    - _Requirements: 4.1, 4.2_

  - [ ] 6.2 Cross-validation with external F1 implementations
    - Compare F1 results with sklearn.metrics.f1_score for validation
    - Test with synthetic datasets with known characteristics
    - Ensure consistency across different input scenarios
    - _Requirements: 4.1, 4.4_

- [ ] 7. Performance testing and optimization
  - [ ] 7.1 Measure evaluation performance impact
    - Benchmark evaluation time before and after F1 implementation
    - Ensure no significant performance degradation compared to NDCG
    - Profile memory usage during F1 calculation
    - _Requirements: 2.2_

  - [ ] 7.2 Test with large datasets
    - Run evaluation on full Herb and NetEase datasets
    - Verify consistent performance across different dataset sizes
    - Ensure scalability with large test sets
    - _Requirements: 2.2, 4.2_

- [ ] 8. Update documentation and comments
  - [ ] 8.1 Update code documentation
    - Add docstrings to new F1 calculation function
    - Update comments in modified evaluation code
    - Update any inline documentation referencing NDCG
    - _Requirements: 2.1_

  - [ ] 8.2 Update configuration and help text
    - Modify any parameter descriptions that mention NDCG
    - Update example outputs in code comments
    - Ensure consistency in metric naming throughout codebase
    - _Requirements: 3.1, 3.2_

- [ ] 9. Integration testing with full pipeline
  - [ ] 9.1 End-to-end testing with training pipeline
    - Run complete training cycle with F1 evaluation
    - Verify all output formats work correctly with F1 scores
    - Test model saving and loading with F1-based evaluation
    - _Requirements: 1.1, 2.4, 3.4_

  - [ ] 9.2 Regression testing
    - Ensure other metrics (precision, recall, RMRR) remain unchanged
    - Verify training convergence behavior is not affected
    - Test with different model configurations and hyperparameters
    - _Requirements: 2.3, 4.5_