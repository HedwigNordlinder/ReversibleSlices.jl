struct NestedModelStructure{T<:Real}
    n_models::Int  
    dims::Vector{Int}  
    full_cov::Matrix{T}  
    conditional_cache::ConditionalCache{T}
    
    function NestedModelStructure{T}(n_models::Int, dims::Vector{Int}, full_cov::Matrix{T}) where T<:Real
        @assert length(dims) == n_models 
        @assert issorted(dims) 
        @assert size(full_cov, 1) == size(full_cov, 2) == dims[end]
        @assert isposdef(full_cov)
        
        # Initialize cache
        cache = ConditionalCache{T}()
        
        # Pre-compute conditional covariances
        for i in 1:(n_models-1)
            for j in (i+1):n_models
                dim_from = dims[i]
                dim_to = dims[j]
                
                Σ11 = full_cov[1:dim_from, 1:dim_from]
                Σ12 = full_cov[1:dim_from, (dim_from+1):dim_to]
                Σ22 = full_cov[(dim_from+1):dim_to, (dim_from+1):dim_to]
                
                cond_cov = Σ22 - Σ12' * inv(Σ11) * Σ12
                cond_cov = (cond_cov + cond_cov') / 2  # Ensure symmetry
                
                cache.conditional_covs[(i,j)] = cond_cov
            end
        end
        
        new{T}(n_models, dims, full_cov, cache)
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
struct ConditionalCache{T<:Real}
    conditional_means::Dict{Tuple{Int,Int,Vector{T}}, Vector{T}}
    conditional_covs::Dict{Tuple{Int,Int}, Matrix{T}}
end

function ConditionalCache{T}() where T<:Real
    return ConditionalCache(
        Dict{Tuple{Int,Int,Vector{T}}, Vector{T}}(),
        Dict{Tuple{Int,Int}, Matrix{T}}()
    )
end