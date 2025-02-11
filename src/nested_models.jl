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
# Get the marginalized log prior density for a parameter vector in a specific model
function log_prior_density(problem::RJESSProblem{T}, model_index::Int, params::Vector{T}) where T<:Real
    dim = problem.model_dimensions[model_index]
    if model_index == 1
        model_cov = problem.nested_structure.full_cov[1:dim, 1:dim]
        return logpdf(MvNormal(zeros(dim), model_cov), params)
    else
        # Split into conditional parts
        dim_prev = problem.model_dimensions[model_index-1]
        params_prev = params[1:dim_prev]
        params_new = params[(dim_prev+1):end]
        
        # Get marginal for first part
        model_cov_prev = problem.nested_structure.full_cov[1:dim_prev, 1:dim_prev]
        log_prob_prev = logpdf(MvNormal(zeros(dim_prev), model_cov_prev), params_prev)
        
        # Get conditional for second part
        cond_mean, cond_cov = get_conditional_distribution(problem.nested_structure, model_index-1, model_index, params_prev)
        log_prob_cond = logpdf(MvNormal(cond_mean, cond_cov), params_new)
        
        return log_prob_prev + log_prob_cond
    end
end

# Method to get marginalized covariance for a specific model
function get_model_covariance(nested::NestedModelStructure, model_index::Int)
    @assert 1 ≤ model_index ≤ nested.n_models
    dim = nested.dims[model_index]
    return nested.full_cov[1:dim, 1:dim]
end
function get_conditional_distribution(nested::NestedModelStructure, from_index::Int, to_index::Int, current_params::Vector{T}) where T<:Real
    @assert from_index < to_index ≤ nested.n_models "Invalid model transition"
    
    # Get cached conditional covariance
    cond_cov = nested.conditional_cache.conditional_covs[(from_index, to_index)]
    
    # Compute conditional mean (this still needs to be computed each time)
    dim_from = nested.dims[from_index]
    Σ11 = nested.full_cov[1:dim_from, 1:dim_from]
    Σ12 = nested.full_cov[1:dim_from, (dim_from+1):nested.dims[to_index]]
    
    cond_mean = Σ12' * inv(Σ11) * current_params
    
    return cond_mean, cond_cov
end
# Helper function to generate initial state for a given model
function sample_initial_state(
    problem::RJESSProblem{T}, 
    model_index::Int) where T<:Real
    
    cov = get_model_covariance(problem.nested_structure, model_index)
    dim = problem.model_dimensions[model_index]
    
    return rand(MvNormal(zeros(dim), cov))
end