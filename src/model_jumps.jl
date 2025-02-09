struct ModelJumpProposal{T<:Real}
    from_index::Int
    to_index::Int
    current_params::Vector{T}
    proposed_params::Vector{T}
    log_jacobian::T  # Only the Jacobian of the transformation
end
# Double check the sign here
function propose_up_jump(problem::RJESSProblem{T}, from_index::Int, to_index::Int, current_params::Vector{T}) where T<:Real
    cond_mean, cond_cov = get_conditional_distribution(problem.nested_structure, from_index, to_index, current_params)
    
    # Sample new parameters from conditional
    n_new_params = problem.model_dimensions[to_index] - problem.model_dimensions[from_index]
    new_params = rand(MvNormal(cond_mean, cond_cov))
    
    # Construct full parameter vector
    proposed_params = vcat(current_params, new_params)
    
    # Add Jacobian term for dimension change
    # log_jacobian =  0.5 * logdet(cond_cov)
    log_jacobian = 0.0
    return ModelJumpProposal(from_index, to_index, current_params, proposed_params, log_jacobian)
end

function propose_down_jump(problem::RJESSProblem{T}, from_index::Int, to_index::Int, current_params::Vector{T}) where T<:Real
    dim_to = problem.model_dimensions[to_index]
    proposed_params = current_params[1:dim_to]
    
    # Get reverse conditional distribution for Jacobian calculation
    _, cond_cov = get_conditional_distribution(problem.nested_structure, to_index, from_index, proposed_params)
    
    # Add Jacobian term for dimension change (note positive sign here)
    # log_jacobian = -0.5 * logdet(cond_cov)
    log_jacobian = 0.0
    
    return ModelJumpProposal(from_index, to_index, current_params, proposed_params, log_jacobian)
end