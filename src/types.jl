struct ConditionalCache{T<:Real}
    conditional_means::Dict{Tuple{Int,Int,Vector{T}}, Vector{T}}
    conditional_covs::Dict{Tuple{Int,Int}, Matrix{T}}
    conditional_chols::Dict{Tuple{Int,Int}, Cholesky{T, Matrix{T}}}  # Add this
end

function ConditionalCache{T}() where T<:Real
    return ConditionalCache(
        Dict{Tuple{Int,Int,Vector{T}}, Vector{T}}(),
        Dict{Tuple{Int,Int}, Matrix{T}}(),
        Dict{Tuple{Int,Int}, Cholesky{T, Matrix{T}}}()
    )
end

struct NestedModelStructure{T<:Real}
    n_models::Int  
    dims::Vector{Int}  
    full_cov::Matrix{T}
    full_chol::Cholesky{T, Matrix{T}}
    marginal_chols::Vector{Cholesky{T, Matrix{T}}}
    conditional_cache::ConditionalCache{T}
    
    function NestedModelStructure{T}(n_models::Int, dims::Vector{Int}, full_cov::Matrix{T}) where T<:Real
        @assert length(dims) == n_models 
        @assert issorted(dims) 
        @assert size(full_cov, 1) == size(full_cov, 2) == dims[end]
        
        # Add small regularization to ensure positive definiteness
        full_cov_reg = full_cov + √eps(T) * I
        
        # Compute full matrix Cholesky
        full_chol = cholesky(Hermitian(full_cov_reg))
        
        # Compute marginal Cholesky factors
        marginal_chols = Vector{Cholesky{T, Matrix{T}}}(undef, n_models)
        for i in 1:n_models
            dim = dims[i]
            marginal_covs = Hermitian(view(full_cov_reg, 1:dim, 1:dim))
            marginal_chols[i] = cholesky(marginal_covs)
        end
        
        # Initialize cache
        cache = ConditionalCache{T}()
        
        # Pre-compute conditional covariances using more stable method
        for i in 1:(n_models-1)
            for j in (i+1):n_models
                dim_from = dims[i]
                dim_to = dims[j]
                
                # Use block matrix form
                full_block = view(full_cov_reg, 1:dim_to, 1:dim_to)
                L = cholesky(Hermitian(full_block)).L
                
                # Extract blocks after Cholesky
                L11 = L[1:dim_from, 1:dim_from]
                L21 = L[(dim_from+1):dim_to, 1:dim_from]
                L22 = L[(dim_from+1):dim_to, (dim_from+1):dim_to]
                
                # Compute conditional covariance directly from Cholesky factors
                cond_cov = L22 * L22'
                
                cache.conditional_covs[(i,j)] = cond_cov
                cache.conditional_chols[(i,j)] = cholesky(Hermitian(cond_cov))
            end
        end
        
        new{T}(n_models, dims, full_cov, full_chol, marginal_chols, cache)
    end
end
# Main problem specification type
struct RJESSProblem{T<:Real,F<:Function}
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
    ) where {T<:Real,F<:Function}

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
) where {T<:Real,F<:Function}
    return RJESSProblem{T,F}(loglikelihood, full_covariance, model_dimensions, model_priors)
end
