import numpy as np

def hosvd(tensor, rank):
    """
    Perform HOSVD on the input tensor and return the factor matrices.

    Parameters:
    tensor (numpy.ndarray): The input tensor of shape (I_1, I_2, ..., I_N).
    rank (int): The CPD rank (number of rank-1 tensors to decompose into).

    Returns:
    list of numpy.ndarray: Factor matrices initialized from HOSVD.
    """
    factors = []
    for mode in range(tensor.ndim):
        # Unfold the tensor along the current mode
        unfold_tensor = np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)
        
        # Perform SVD on the unfolded tensor
        U, _, _ = np.linalg.svd(unfold_tensor, full_matrices=False)
        
        # Append the left singular vectors (truncated to the specified rank)
        factors.append(U[:, :rank])
    
    return factors

def khatri_rao(matrices):
    """
    Compute the Khatri-Rao product (column-wise Kronecker product) of a list of matrices.
    
    Parameters:
    matrices (list of numpy.ndarray): List of factor matrices.
    
    Returns:
    numpy.ndarray: The Khatri-Rao product.
    """
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.einsum('ik,jk->ijk', result, matrix).reshape(-1, result.shape[1])
    return result

def normalize_factors(factors):
    """
    Normalize factor matrices and return the scaling factors (lambdas).
    
    Parameters:
    factors (list of numpy.ndarray): List of factor matrices to be normalized.
    
    Returns:
    list of numpy.ndarray: List of normalized factor matrices.
    numpy.ndarray: Weights (lambdas) for each rank-1 component.
    """
    rank = factors[0].shape[1]  # Number of components (rank)
    lambdas = np.ones(rank, dtype=factors[0].dtype)  # Initialize lambdas (weights) as complex or real based on factors
    
    for r in range(rank):
        norm = 1
        for factor in factors:
            column_norm = np.linalg.norm(factor[:, r])  # Frobenius norm of the column
            if column_norm > 0:  # Avoid division by zero
                factor[:, r] /= column_norm  # Normalize the column
            norm *= column_norm  # Multiply norms to compute lambda_r
        lambdas[r] = norm
    
    return factors, lambdas

def initialize_factors(tensor, rank):
    """
    Initialize the factor matrices for CPD decomposition.

    Parameters:
    tensor (numpy.ndarray): The input tensor of shape (I_1, I_2, ..., I_N).
    rank (int): The CPD rank (number of rank-1 tensors to decompose into).

    Returns:
    list of numpy.ndarray: List of factor matrices initialized randomly based on tensor dtype.
    """
    factors = []
    for mode in range(tensor.ndim):
        size = tensor.shape[mode]
        # Initialize factor matrices with the same dtype as the input tensor
        factors.append(np.random.rand(size, rank).astype(tensor.dtype))
    return factors

def cpd_als(tensor, rank, max_iter=100, tol=1e-6, hosvd_init=True):
    """
    Perform CPD decomposition of a tensor using the ALS method with HOSVD initialization.
    
    Parameters:
    tensor (numpy.ndarray): The input tensor of shape (I_1, I_2, ..., I_N).
    rank (int): The CPD rank (number of rank-1 tensors to decompose into).
    max_iter (int): Maximum number of ALS iterations.
    tol (float): Tolerance for convergence (based on the reconstruction error).
    hosvd_init (bool): Whether to initialize the factor matrices using HOSVD.
    
    Returns:
    list of numpy.ndarray: The normalized factor matrices of the CPD decomposition.
    numpy.ndarray: Weights (lambdas) for each rank-1 component.
    float: The final reconstruction error.
    """
    # Step 1: Initialize factor matrices using HOSVD or random initialization
    if hosvd_init:
        factors = hosvd(tensor, rank)
    else:
        factors = initialize_factors(tensor, rank)
    
    # Get the shape of the tensor
    shape = tensor.shape
    norm_tensor = np.linalg.norm(tensor)  # Norm of the original tensor for error calculation

    for iteration in range(max_iter):
        for mode in range(tensor.ndim):
            # Step 2: Update the factor matrix for the current mode
            
            # Exclude the current mode's factor matrix
            other_factors = [factors[i] for i in range(tensor.ndim) if i != mode]
            
            # Khatri-Rao product of all other factors
            kr_product = khatri_rao(other_factors)

            # Unfold the tensor along the current mode
            unfold_tensor = np.moveaxis(tensor, mode, 0).reshape(shape[mode], -1)

            # Update the factor matrix using the least squares solution
            factors[mode] = np.linalg.lstsq(kr_product, unfold_tensor.T, rcond=None)[0].T

        # Step 3: Check for convergence using reconstruction error
        # Reconstruct the tensor from the factor matrices
        reconstructed = np.zeros(shape, dtype=tensor.dtype)  # Ensure same dtype (complex or real)

        # Normalize the factors and extract lambdas (weights) at the end of ALS iterations
        factors, lambdas = normalize_factors(factors)
        
        # Reconstruct the tensor using the normalized factors and lambdas
        for r in range(rank):
            outer_product = np.outer(factors[0][:, r], factors[1][:, r])
            for i in range(2, tensor.ndim):
                outer_product = np.tensordot(outer_product, factors[i][:, r], axes=0)
            reconstructed += lambdas[r] * outer_product

        # Compute the Frobenius norm of the difference (error)
        error = np.linalg.norm(tensor - reconstructed) / norm_tensor

        if error < tol:
            break

    return factors, lambdas, error

# Function to allow user to input a tensor of arbitrary dimensions and type (real/complex)
def user_input_tensor():
    """
    Get a tensor from user input of arbitrary dimensions, allowing for real or complex entries.
    
    Returns:
    numpy.ndarray: User input tensor of any order.
    """
    # Get the number of dimensions of the tensor
    ndim = int(input("Enter the number of dimensions (N) of the tensor: "))
    
    # Get the size of each dimension
    shape = []
    for i in range(ndim):
        size = int(input(f"Enter the size of dimension {i + 1}: "))
        shape.append(size)
    
    # Initialize an empty tensor
    tensor = np.zeros(shape, dtype=complex)  # Use complex type for generality
    
    # Fill the tensor with user inputs
    print(f"Enter the tensor elements one by one for shape {shape}.")
    print("You can enter real numbers (e.g., 1.0) or complex numbers (e.g., 1+2j):")
    it = np.nditer(tensor, flags=['multi_index'])
    while not it.finished:
        value = input(f"Entry {it.multi_index}: ")
        tensor[it.multi_index] = complex(value)  # Convert the input string to a complex number
        it.iternext()
    
    return tensor

def automatic_rank_cpd(tensor, max_rank=10, max_iter=100, tol=1e-6):
    """
    Automatically determine the CPD rank by starting at rank 1 and increasing
    the rank until the reconstruction error is below the tolerance threshold.
    
    Parameters:
    tensor (numpy.ndarray): The input tensor of shape (I_1, I_2, ..., I_N).
    max_rank (int): Maximum rank to attempt.
    max_iter (int): Maximum number of ALS iterations.
    tol (float): Tolerance for the reconstruction error.
    
    Returns:
    list of numpy.ndarray: Factor matrices corresponding to the best rank.
    numpy.ndarray: Weights (lambdas) for each rank-1 component.
    int: The chosen rank.
    float: The final reconstruction error.
    """
    rank = 1
    while rank <= max_rank:
        print(f"Trying rank {rank}...")
        factors, lambdas, error = cpd_als(tensor, rank, max_iter=max_iter, tol=tol, hosvd_init=True)
        
        print(f"Rank {rank} error: {error:.6f}")
        if error < tol:
            print(f"Optimal rank found: {rank}")
            return factors, lambdas, rank, error
        
        rank += 1
    
    print(f"Maximum rank {max_rank} reached. Returning best approximation.")
    return factors, lambdas, rank - 1, error

def reconstruct_tensor(factors, lambdas):
    """
    Reconstruct the tensor using the factor matrices and lambdas.
    
    Parameters:
    factors (list of numpy.ndarray): The factor matrices.
    lambdas (numpy.ndarray): The weights (lambdas) for each rank-1 component.
    
    Returns:
    numpy.ndarray: The reconstructed tensor.
    """
    shape = [factor.shape[0] for factor in factors]
    reconstructed = np.zeros(shape, dtype=factors[0].dtype)  # Same dtype as factors (complex or real)
    
    rank = len(lambdas)
    
    for r in range(rank):
        outer_product = np.outer(factors[0][:, r], factors[1][:, r])
        for i in range(2, len(factors)):
            outer_product = np.tensordot(outer_product, factors[i][:, r], axes=0)
        reconstructed += lambdas[r] * outer_product
    
    return reconstructed

# Example Usage:

# Get the tensor from user input
tensor = user_input_tensor()

# Automatically determine the rank for CPD using ALS with HOSVD initialization
factors, lambdas, chosen_rank, final_error = automatic_rank_cpd(tensor)

# Reconstruct the tensor using the factor matrices and lambdas
reconstructed_tensor = reconstruct_tensor(factors, lambdas)

# Display the factor matrices, lambdas, the chosen rank, and the final reconstruction
print(f"\nFinal chosen rank: {chosen_rank} with error: {final_error}")
print(f"\nLambdas (weights):\n{lambdas}")
print("\nFactor matrices:")
for i, factor in enumerate(factors):
    print(f"Factor matrix {i + 1} shape: {factor.shape}\n{factor}")

print(f"\nReconstructed Tensor:\n{reconstructed_tensor}")
print(f"\nOriginal Tensor:\n{tensor}")

# Check how close the original tensor is to the reconstructed tensor
difference = np.linalg.norm(tensor - reconstructed_tensor)
print(f"\nDifference between original and reconstructed tensor (Frobenius norm): {difference}")
