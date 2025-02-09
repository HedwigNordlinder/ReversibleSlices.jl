struct ModelJumpProposal{T<:Real}
    from_index::Int
    to_index::Int
    current_params::Vector{T}
    proposed_params::Vector{T}
    log_jacobian::T
    log_ratio::T    # Will store the full log acceptance ratio
end

# Generate proposal for jumping up to larger model
function propose_up_jump(
    problem::RJESSProblem{T},
    from_index::Int,
    to_index::Int,
    current_params::Vector{T}) where T<:Real
    
    @assert from_index < to_index "Up jump must move to larger model"
    
    # Get dimensions
    dim_from = problem.model_dimensions[from_index]
    dim_to = problem.model_dimensions[to_index]
    
    # Get conditional distribution parameters
    cond_cov = get_conditional_covariance(problem.nested_structure, from_index, to_index)
    
    # Sample new parameters
    n_new_params = dim_to - dim_from
    new_params = rand(MvNormal(zeros(n_new_params), cond_cov))
    
    # Construct full parameter vector for larger model
    proposed_params = vcat(current_params, new_params)
    
    # Calculate components of acceptance ratio:
    # 1. Jacobian term
    log_jacobian = 0.5 * logdet(cond_cov)
    
    # 2. Likelihood ratio
    log_likelihood_ratio = problem.loglikelihood(proposed_params) - problem.loglikelihood(current_params)
    
    # 3. Model prior ratio
    log_prior_ratio = log(problem.model_priors[to_index]) - log(problem.model_priors[from_index])
    
    # Combine all terms
    log_ratio = log_likelihood_ratio + log_prior_ratio + log_jacobian
    
    return ModelJumpProposal(from_index, to_index, current_params, proposed_params, log_jacobian, log_ratio)
end

# Generate proposal for jumping down to smaller model
function propose_down_jump(
    problem::RJESSProblem{T},
    from_index::Int,
    to_index::Int,
    current_params::Vector{T}) where T<:Real
    
    @assert from_index > to_index "Down jump must move to smaller model"
    
    # Get dimensions
    dim_to = problem.model_dimensions[to_index]
    
    # For down moves, just take first dim_to parameters
    proposed_params = current_params[1:dim_to]
    
    # Get conditional distribution parameters for reverse move
    cond_cov = get_conditional_covariance(problem.nested_structure, to_index, from_index)
    
    # Calculate components of acceptance ratio:
    # 1. Jacobian term (negative of up move)
    log_jacobian = -0.5 * logdet(cond_cov)
    
    # 2. Likelihood ratio
    log_likelihood_ratio = problem.loglikelihood(proposed_params) - problem.loglikelihood(current_params)
    
    # 3. Model prior ratio
    log_prior_ratio = log(problem.model_priors[to_index]) - log(problem.model_priors[from_index])
    
    # Combine all terms
    log_ratio = log_likelihood_ratio + log_prior_ratio + log_jacobian
    
    return ModelJumpProposal(from_index, to_index, current_params, proposed_params, log_jacobian, log_ratio)
end