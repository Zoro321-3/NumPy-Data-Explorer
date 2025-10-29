"""
NumPy Data Explorer
-------------------
A comprehensive project demonstrating NumPy fundamentals including:
- Array creation, indexing, and slicing
- Mathematical, axis-wise, and statistical operations
- Reshaping and broadcasting techniques
- Save/load operations
- Performance comparison with Python lists

Author: Syntecxhub Data Science Intern
Project: Week 1 - NumPy Data Explorer
"""

import numpy as np
import time
import sys

# ============================================================================
# 1. ARRAY CREATION, INDEXING & SLICING
# ============================================================================

def array_creation_demo():
    """Demonstrates various ways to create NumPy arrays"""
    print("="*70)
    print("1. ARRAY CREATION, INDEXING & SLICING")
    print("="*70)
    
    # Creating 1D array
    arr_1d = np.array([1, 2, 3, 4, 5])
    print("\n1D Array:")
    print(arr_1d)
    
    # Creating 2D array
    arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("\n2D Array:")
    print(arr_2d)
    
    # Creating 3D array
    arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print("\n3D Array:")
    print(arr_3d)
    
    # Special arrays
    zeros = np.zeros((3, 3))
    ones = np.ones((2, 4))
    identity = np.eye(3)
    random_arr = np.random.rand(3, 3)
    range_arr = np.arange(0, 10, 2)
    linspace_arr = np.linspace(0, 1, 5)
    
    print("\nZeros Array (3x3):")
    print(zeros)
    print("\nOnes Array (2x4):")
    print(ones)
    print("\nIdentity Matrix (3x3):")
    print(identity)
    print("\nRandom Array (3x3):")
    print(random_arr)
    print("\nRange Array (0 to 10, step 2):")
    print(range_arr)
    print("\nLinspace Array (5 values from 0 to 1):")
    print(linspace_arr)
    
    # Indexing examples
    print("\n--- INDEXING EXAMPLES ---")
    sample_arr = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    print("\nSample Array:")
    print(sample_arr)
    print(f"\nElement at [0, 0]: {sample_arr[0, 0]}")
    print(f"Element at [1, 2]: {sample_arr[1, 2]}")
    print(f"Element at [2, 1]: {sample_arr[2, 1]}")
    
    # Slicing examples
    print("\n--- SLICING EXAMPLES ---")
    print("\nFirst row:", sample_arr[0, :])
    print("Last column:", sample_arr[:, -1])
    print("First 2 rows, first 2 columns:\n", sample_arr[:2, :2])
    print("Every other element in 1D:", arr_1d[::2])
    
    return arr_2d, random_arr


# ============================================================================
# 2. MATHEMATICAL & STATISTICAL OPERATIONS
# ============================================================================

def mathematical_operations(arr):
    """Demonstrates mathematical operations on arrays"""
    print("\n" + "="*70)
    print("2. MATHEMATICAL & STATISTICAL OPERATIONS")
    print("="*70)
    
    print("\nOriginal Array:")
    print(arr)
    
    # Basic mathematical operations
    print("\n--- BASIC MATH OPERATIONS ---")
    print("\nArray + 10:")
    print(arr + 10)
    print("\nArray * 2:")
    print(arr * 2)
    print("\nArray squared:")
    print(arr ** 2)
    print("\nSquare root of array:")
    print(np.sqrt(arr))
    
    # Element-wise operations between arrays
    arr2 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    print("\nSecond Array:")
    print(arr2)
    print("\nArray1 + Array2:")
    print(arr + arr2)
    print("\nArray1 * Array2:")
    print(arr * arr2)
    
    # Statistical operations
    print("\n--- STATISTICAL OPERATIONS ---")
    data = np.random.randint(1, 100, size=(5, 4))
    print("\nRandom Data (5x4):")
    print(data)
    print(f"\nMean: {np.mean(data):.2f}")
    print(f"Median: {np.median(data):.2f}")
    print(f"Standard Deviation: {np.std(data):.2f}")
    print(f"Variance: {np.var(data):.2f}")
    print(f"Min: {np.min(data)}")
    print(f"Max: {np.max(data)}")
    print(f"Sum: {np.sum(data)}")
    
    # Axis-wise operations
    print("\n--- AXIS-WISE OPERATIONS ---")
    print("\nMean along axis 0 (columns):", np.mean(data, axis=0))
    print("Mean along axis 1 (rows):", np.mean(data, axis=1))
    print("\nSum along axis 0 (columns):", np.sum(data, axis=0))
    print("Sum along axis 1 (rows):", np.sum(data, axis=1))
    print("\nMax along axis 0 (columns):", np.max(data, axis=0))
    print("Min along axis 1 (rows):", np.min(data, axis=1))
    
    return data


# ============================================================================
# 3. RESHAPING & BROADCASTING
# ============================================================================

def reshaping_broadcasting_demo():
    """Demonstrates reshaping and broadcasting techniques"""
    print("\n" + "="*70)
    print("3. RESHAPING & BROADCASTING")
    print("="*70)
    
    # Reshaping
    print("\n--- RESHAPING ---")
    arr = np.arange(12)
    print("\nOriginal 1D Array (12 elements):")
    print(arr)
    
    reshaped_3x4 = arr.reshape(3, 4)
    print("\nReshaped to 3x4:")
    print(reshaped_3x4)
    
    reshaped_2x6 = arr.reshape(2, 6)
    print("\nReshaped to 2x6:")
    print(reshaped_2x6)
    
    reshaped_3d = arr.reshape(2, 2, 3)
    print("\nReshaped to 3D (2x2x3):")
    print(reshaped_3d)
    
    # Flatten
    flattened = reshaped_3x4.flatten()
    print("\nFlattened back to 1D:")
    print(flattened)
    
    # Transpose
    print("\nTranspose of 3x4 array:")
    print(reshaped_3x4.T)
    
    # Broadcasting
    print("\n--- BROADCASTING ---")
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    vector = np.array([10, 20, 30])
    
    print("\nMatrix (3x3):")
    print(matrix)
    print("\nVector (1x3):")
    print(vector)
    print("\nMatrix + Vector (broadcasting):")
    print(matrix + vector)
    
    # Broadcasting with column vector
    col_vector = np.array([[1], [2], [3]])
    print("\nColumn Vector (3x1):")
    print(col_vector)
    print("\nMatrix + Column Vector (broadcasting):")
    print(matrix + col_vector)
    
    # Broadcasting with scalar
    print("\nMatrix * 5 (scalar broadcasting):")
    print(matrix * 5)
    
    return reshaped_3x4


# ============================================================================
# 4. SAVE/LOAD OPERATIONS
# ============================================================================

def save_load_demo(arr):
    """Demonstrates saving and loading NumPy arrays"""
    print("\n" + "="*70)
    print("4. SAVE/LOAD OPERATIONS")
    print("="*70)
    
    # Save single array
    print("\nSaving array to 'sample_array.npy'...")
    np.save('sample_array.npy', arr)
    print("Array saved successfully!")
    
    # Load single array
    print("\nLoading array from 'sample_array.npy'...")
    loaded_arr = np.load('sample_array.npy')
    print("Loaded Array:")
    print(loaded_arr)
    
    # Save multiple arrays
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    arr3 = np.array([[7, 8], [9, 10]])
    
    print("\nSaving multiple arrays to 'multiple_arrays.npz'...")
    np.savez('multiple_arrays.npz', array1=arr1, array2=arr2, array3=arr3)
    print("Multiple arrays saved successfully!")
    
    # Load multiple arrays
    print("\nLoading multiple arrays from 'multiple_arrays.npz'...")
    loaded_data = np.load('multiple_arrays.npz')
    print("\nArray 1:", loaded_data['array1'])
    print("Array 2:", loaded_data['array2'])
    print("Array 3:\n", loaded_data['array3'])
    
    # Save to text file
    print("\nSaving array to 'sample_array.txt'...")
    np.savetxt('sample_array.txt', arr, fmt='%d')
    print("Array saved to text file!")
    
    # Load from text file
    print("\nLoading array from 'sample_array.txt'...")
    loaded_txt = np.loadtxt('sample_array.txt')
    print("Loaded Array from text:")
    print(loaded_txt)


# ============================================================================
# 5. PERFORMANCE COMPARISON: NUMPY VS PYTHON LISTS
# ============================================================================

def performance_comparison():
    """Compares NumPy performance with Python lists"""
    print("\n" + "="*70)
    print("5. PERFORMANCE COMPARISON: NUMPY VS PYTHON LISTS")
    print("="*70)
    
    size = 1000000
    
    # Python list operations
    print(f"\nTesting with {size:,} elements...\n")
    
    # Test 1: Creation
    print("--- Test 1: Array/List Creation ---")
    start = time.time()
    python_list = list(range(size))
    python_time = time.time() - start
    print(f"Python list creation: {python_time:.6f} seconds")
    
    start = time.time()
    numpy_array = np.arange(size)
    numpy_time = time.time() - start
    print(f"NumPy array creation: {numpy_time:.6f} seconds")
    print(f"NumPy is {python_time/numpy_time:.2f}x faster")
    
    # Test 2: Sum
    print("\n--- Test 2: Sum Operation ---")
    start = time.time()
    python_sum = sum(python_list)
    python_time = time.time() - start
    print(f"Python list sum: {python_time:.6f} seconds")
    
    start = time.time()
    numpy_sum = np.sum(numpy_array)
    numpy_time = time.time() - start
    print(f"NumPy array sum: {numpy_time:.6f} seconds")
    print(f"NumPy is {python_time/numpy_time:.2f}x faster")
    
    # Test 3: Element-wise multiplication
    print("\n--- Test 3: Element-wise Multiplication ---")
    start = time.time()
    python_mult = [x * 2 for x in python_list]
    python_time = time.time() - start
    print(f"Python list multiplication: {python_time:.6f} seconds")
    
    start = time.time()
    numpy_mult = numpy_array * 2
    numpy_time = time.time() - start
    print(f"NumPy array multiplication: {numpy_time:.6f} seconds")
    print(f"NumPy is {python_time/numpy_time:.2f}x faster")
    
    # Test 4: Memory usage
    print("\n--- Test 4: Memory Usage ---")
    python_size = sys.getsizeof(python_list)
    numpy_size = numpy_array.nbytes
    print(f"Python list memory: {python_size:,} bytes")
    print(f"NumPy array memory: {numpy_size:,} bytes")
    print(f"NumPy uses {python_size/numpy_size:.2f}x less memory")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run all demonstrations"""
    print("\n" + "="*70)
    print(" "*15 + "NUMPY DATA EXPLORER")
    print(" "*10 + "Syntecxhub Data Science Internship")
    print(" "*20 + "Week 1 Project")
    print("="*70)
    
    # Run all demonstrations
    arr_2d, random_arr = array_creation_demo()
    data = mathematical_operations(arr_2d)
    reshaped = reshaping_broadcasting_demo()
    save_load_demo(reshaped)
    performance_comparison()
    
    print("\n" + "="*70)
    print(" "*15 + "PROJECT COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nFiles created:")
    print("  - sample_array.npy")
    print("  - multiple_arrays.npz")
    print("  - sample_array.txt")
    print("\nNext steps:")
    print("  1. Review the code and understand each section")
    print("  2. Upload this code to your GitHub repository")
    print("  3. Submit through the official Submission Form")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
