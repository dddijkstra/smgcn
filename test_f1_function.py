#!/usr/bin/env python3
"""
Test script for the f1_at_k function to verify mathematical correctness
and error handling according to requirements 1.2, 1.3, 4.3, 4.4
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.batch_test import f1_at_k
import numpy as np

def test_f1_calculation():
    """Test F1 calculation with various precision and recall values"""
    print("Testing F1 calculation function...")
    
    # Test case 1: Normal values
    precision, recall = 0.5, 0.5
    expected = 0.5
    result = f1_at_k(precision, recall)
    print(f"Test 1 - P={precision}, R={recall}: F1={result:.6f}, Expected={expected}")
    assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"
    
    # Test case 2: Different precision and recall
    precision, recall = 0.8, 0.6
    expected = 2 * (0.8 * 0.6) / (0.8 + 0.6)  # 0.6857142857142857
    result = f1_at_k(precision, recall)
    print(f"Test 2 - P={precision}, R={recall}: F1={result:.6f}, Expected={expected:.6f}")
    assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"
    
    # Test case 3: Perfect precision, zero recall
    precision, recall = 1.0, 0.0
    expected = 0.0
    result = f1_at_k(precision, recall)
    print(f"Test 3 - P={precision}, R={recall}: F1={result:.6f}, Expected={expected}")
    assert result == expected, f"Expected {expected}, got {result}"
    
    # Test case 4: Zero precision, perfect recall
    precision, recall = 0.0, 1.0
    expected = 0.0
    result = f1_at_k(precision, recall)
    print(f"Test 4 - P={precision}, R={recall}: F1={result:.6f}, Expected={expected}")
    assert result == expected, f"Expected {expected}, got {result}"
    
    # Test case 5: Both zero (division by zero case)
    precision, recall = 0.0, 0.0
    expected = 0.0
    result = f1_at_k(precision, recall)
    print(f"Test 5 - P={precision}, R={recall}: F1={result:.6f}, Expected={expected}")
    assert result == expected, f"Expected {expected}, got {result}"
    
    # Test case 6: Perfect scores
    precision, recall = 1.0, 1.0
    expected = 1.0
    result = f1_at_k(precision, recall)
    print(f"Test 6 - P={precision}, R={recall}: F1={result:.6f}, Expected={expected}")
    assert result == expected, f"Expected {expected}, got {result}"
    
    print("✓ All F1 calculation tests passed!")

def test_input_validation():
    """Test input validation and error handling"""
    print("\nTesting input validation...")
    
    # Test invalid types
    try:
        f1_at_k("0.5", 0.5)
        assert False, "Should have raised TypeError for string input"
    except TypeError as e:
        print(f"✓ Correctly caught TypeError for string precision: {e}")
    
    try:
        f1_at_k(0.5, None)
        assert False, "Should have raised TypeError for None input"
    except TypeError as e:
        print(f"✓ Correctly caught TypeError for None recall: {e}")
    
    # Test NaN values
    try:
        f1_at_k(float('nan'), 0.5)
        assert False, "Should have raised ValueError for NaN precision"
    except ValueError as e:
        print(f"✓ Correctly caught ValueError for NaN precision: {e}")
    
    try:
        f1_at_k(0.5, float('inf'))
        assert False, "Should have raised ValueError for infinite recall"
    except ValueError as e:
        print(f"✓ Correctly caught ValueError for infinite recall: {e}")
    
    # Test out of range values
    try:
        f1_at_k(-0.1, 0.5)
        assert False, "Should have raised ValueError for negative precision"
    except ValueError as e:
        print(f"✓ Correctly caught ValueError for negative precision: {e}")
    
    try:
        f1_at_k(0.5, 1.1)
        assert False, "Should have raised ValueError for recall > 1"
    except ValueError as e:
        print(f"✓ Correctly caught ValueError for recall > 1: {e}")
    
    print("✓ All input validation tests passed!")

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("\nTesting edge cases...")
    
    # Test with numpy types
    precision = np.float64(0.7)
    recall = np.float32(0.3)
    result = f1_at_k(precision, recall)
    expected = 2 * (0.7 * 0.3) / (0.7 + 0.3)
    print(f"NumPy types - P={precision}, R={recall}: F1={result:.6f}, Expected={expected:.6f}")
    assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"
    
    # Test with integer inputs
    precision = 1
    recall = 0
    result = f1_at_k(precision, recall)
    expected = 0.0
    print(f"Integer inputs - P={precision}, R={recall}: F1={result:.6f}, Expected={expected}")
    assert result == expected, f"Expected {expected}, got {result}"
    
    # Test very small values
    precision = 1e-10
    recall = 1e-10
    result = f1_at_k(precision, recall)
    expected = 1e-10  # F1 should equal precision and recall when they're equal
    print(f"Very small values - P={precision}, R={recall}: F1={result:.2e}, Expected={expected:.2e}")
    assert abs(result - expected) < 1e-15, f"Expected {expected}, got {result}"
    
    print("✓ All edge case tests passed!")

if __name__ == "__main__":
    test_f1_calculation()
    test_input_validation()
    test_edge_cases()
    print("\n🎉 All tests passed! F1 function is working correctly.")