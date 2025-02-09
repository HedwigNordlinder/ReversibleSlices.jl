# Core type for handling nested models
struct NestedModelStructure{T<:Real}
    n_models::Int  # Number of models
    dims::Vector{Int}  # Dimension of each model
    full_cov::Matrix{T}  # Covariance matrix for largest model
    
    # Constructor with validation
    function NestedModelStructure{T}(n_models::Int, dims::Vector{Int}, full_cov::Matrix{T}) where T<:Real
        # Validate dimensions
        @assert length(dims) == n_models "Number of dimensions must match number of models"
        @assert issorted(dims) "Model dimensions must be strictly increasing"
        @assert size(full_cov, 1) == size(full_cov, 2) == dims[end] "Covariance matrix size must match largest model"
        @assert isposdef(full_cov) "Covariance matrix must be positive definite"
        
        new{T}(n_models, dims, full_cov)
    end
end

# Get the marginalized log prior density for a parameter vector in a specific model
function log_prior_density(
    problem::RJESSProblem{T}, 
    model_index::Int, 
    params::Vector{T}) where T<:Real
    
    @assert length(params) == problem.model_dimensions[model_index]
    
    # Get marginalized covariance for this model
    model_cov = get_model_covariance(problem.nested_structure, model_index)
    
    # Compute multivariate normal log density
    return logpdf(MvNormal(zeros(length(params)), model_cov), params)
end

# Method to get marginalized covariance for a specific model
function get_model_covariance(nested::NestedModelStructure, model_index::Int)
    @assert 1 ≤ model_index ≤ nested.n_models
    dim = nested.dims[model_index]
    return nested.full_cov[1:dim, 1:dim]
end

# Method to compute conditional covariance when jumping up
function get_conditional_covariance(nested::NestedModelStructure, from_index::Int, to_index::Int)
    @assert from_index < to_index ≤ nested.n_models "Invalid model transition"
    
    dim_from = nested.dims[from_index]
    dim_to = nested.dims[to_index]
    
    # Extract relevant blocks of the full covariance matrix
    Σ11 = nested.full_cov[1:dim_from, 1:dim_from]
    Σ12 = nested.full_cov[1:dim_from, (dim_from+1):dim_to]
    Σ22 = nested.full_cov[(dim_from+1):dim_to, (dim_from+1):dim_to]
    
    # Compute conditional covariance
    return Σ22 - Σ12' * inv(Σ11) * Σ12
end

# Helper function to generate initial state for a given model
function sample_initial_state(
    problem::RJESSProblem{T}, 
    model_index::Int) where T<:Real
    
    cov = get_model_covariance(problem.nested_structure, model_index)
    dim = problem.model_dimensions[model_index]
    
    return rand(MvNormal(zeros(dim), cov))
end