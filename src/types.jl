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

# Main problem specification type
struct RJESSProblem{T<:Real, F<:Function}
    # User provided elements
    loglikelihood::F                 # Function that computes log likelihood for any model
    full_covariance::Matrix{T}      # Covariance matrix for largest model
    model_dimensions::Vector{Int}    # Dimension of each model
    model_priors::Vector{T}         # Prior probabilities for each model
    
    # Derived elements
    n_models::Int
    nested_structure::NestedModelStructure{T}
    
    # Constructor with validation
    function RJESSProblem{T,F}(
        loglikelihood::F,
        full_covariance::Matrix{T},
        model_dimensions::Vector{Int},
        model_priors::Vector{T}
    ) where {T<:Real, F<:Function}
        
        # Basic validation
        n_models = length(model_dimensions)
        @assert n_models == length(model_priors) "Number of models must match number of priors"
        @assert isapprox(sum(model_priors), one(T)) "Model priors must sum to 1"
        @assert all(>=(0), model_priors) "Model priors must be non-negative"
        @assert issorted(model_dimensions) "Model dimensions must be strictly increasing"
        
        # Create nested structure
        nested_structure = NestedModelStructure{T}(n_models, model_dimensions, full_covariance)
        
        new{T,F}(loglikelihood, full_covariance, model_dimensions, model_priors, 
                 n_models, nested_structure)
    end
end

# Convenience constructor that infers type parameters
function RJESSProblem(
    loglikelihood::F,
    full_covariance::Matrix{T},
    model_dimensions::Vector{Int},
    model_priors::Vector{T}
) where {T<:Real, F<:Function}
    return RJESSProblem{T,F}(loglikelihood, full_covariance, model_dimensions, model_priors)
end