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

# Constructor that infers type parameters
function RJESSProblem(
    loglikelihood::F,
    full_covariance::Matrix{T},
    model_dimensions::Vector{Int},
    model_priors::Vector{T}
) where {T<:Real, F<:Function}
    return RJESSProblem{T,F}(loglikelihood, full_covariance, model_dimensions, model_priors)
end