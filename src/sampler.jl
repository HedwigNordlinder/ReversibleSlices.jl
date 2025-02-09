using EllipticalSliceSampling: ESSModel, ESSState, ESS

struct RJESSState{T<:Real}
    model_index::Int               # Current model index
    params::Vector{T}             # Current parameter values
    log_likelihood::T             # Cached log likelihood
    log_prior::T                  # Cached log prior density
end

struct RJESSOptions
    n_samples::Int               # Number of samples to draw
    prob_within_model::Float64   # Probability of within-model move vs jump
    jump_proposal_dist::Vector{Float64}  # Probability distribution for jump distances
end

function run_sampler(
    problem::RJESSProblem{T},
    options::RJESSOptions;
    initial_model::Int = 1
    ) where T<:Real
    
    # Initialize storage for samples
    model_samples = Vector{Int}(undef, options.n_samples)
    param_samples = Vector{Vector{T}}(undef, options.n_samples)
    logpost_samples = Vector{T}(undef, options.n_samples)  # New array for log posteriors
    
    # Initialize state
    initial_params = sample_initial_state(problem, initial_model)
    current = RJESSState(
        initial_model,
        initial_params,
        problem.loglikelihood(initial_params),
        log_prior_density(problem, initial_model, initial_params)
    )
    
    # Main sampling loop
    for i in 1:options.n_samples
        # Decide whether to do within-model move or between-model jump
        if rand() < options.prob_within_model
            # Within-model move using elliptical slice sampling
            current = elliptical_slice_sampling_step(problem, current)
        else
            # Between-model jump
            current = model_jump_step(problem, current, options.jump_proposal_dist)
        end
        
        # Store samples
        model_samples[i] = current.model_index
        param_samples[i] = copy(current.params)
        logpost_samples[i] = current.log_likelihood + current.log_prior  # Store the log posterior
    end
    
    return model_samples, param_samples, logpost_samples
end

function model_jump_step(
    problem::RJESSProblem{T}, 
    state::RJESSState{T},
    jump_proposal_dist::Vector{Float64}
    ) where T<:Real

    current_model = state.model_index

    # Generate allowed jumps (only nonzero moves that keep new model index valid)
    allowed_jumps_i = vcat(-1:-1:-(state.model_index - 1), 1:(problem.n_models - current_model))
    if isempty(allowed_jumps_i)
        return state
    end

    # Filter allowed jumps based on jump_proposal_dist availability.
    # Use the jump_proposal_dist weight for the target model.
    forward_jumps = Int[]
    forward_weights = Float64[]
    for jump in allowed_jumps_i
        new_model = current_model + jump
        if 1 <= new_model <= length(jump_proposal_dist)
            push!(forward_jumps, jump)
            push!(forward_weights, jump_proposal_dist[new_model])
        end
    end

    if isempty(forward_jumps)
        return state
    end

    # Normalize forward weights to build the proposal distribution q(i→j)
    forward_weights_norm = forward_weights / sum(forward_weights)

    # Propose a jump distance from the forward distribution
    jump_index = sample(1:length(forward_jumps), ProbabilityWeights(forward_weights_norm))
    jump_dist = forward_jumps[jump_index]
    new_model = current_model + jump_dist

    # Generate proposal for the parameter values using the appropriate function
    proposal = jump_dist > 0 ? 
                propose_up_jump(problem, current_model, new_model, state.params) :
                propose_down_jump(problem, current_model, new_model, state.params)

    #
    # Now compute the probability of the reverse move (from new_model back to current_model)
    #
    allowed_jumps_j = vcat(-1:-1:-(new_model - 1), 1:(problem.n_models - new_model))
    backward_jumps = Int[]
    backward_weights = Float64[]
    for rjump in allowed_jumps_j
        candidate = new_model + rjump
        if 1 <= candidate <= length(jump_proposal_dist)
            push!(backward_jumps, rjump)
            push!(backward_weights, jump_proposal_dist[candidate])
        end
    end

    # The reverse jump that would take us from new_model back to current_model
    reverse_jump = current_model - new_model  # note: reverse_jump = -jump_dist
    backward_index = findfirst(x -> x == reverse_jump, backward_jumps)
    if isnothing(backward_index)
        # no reverse move available: reject jump.
        return state
    end

    # Convert backward weights to a normalized probability distribution, q(j→i)
    backward_weights_norm = backward_weights / sum(backward_weights)
    p_forward = forward_weights_norm[jump_index]
    p_backward = backward_weights_norm[backward_index]

    # Compute the acceptance ratio,
    # including the log-ratio from the parameter transformation in the proposal.
    acceptance_ratio = exp(proposal.log_ratio) * (p_backward / p_forward)

    if log(rand()) < acceptance_ratio
        # Accept the jump move by using the proposed new model and parameters.
        return RJESSState(
            new_model,
            proposal.proposed_params,
            problem.loglikelihood(proposal.proposed_params),
            log_prior_density(problem, new_model, proposal.proposed_params)
        )
    else
        # Reject move: remain in the current state.
        return state
    end
end

function elliptical_slice_sampling_step(
    problem::RJESSProblem{T},
    rjess_state::RJESSState{T}
) where T<:Real
    # Get current model and parameters
    ll = ϕ -> problem.loglikelihood(ϕ)
    μ = zeros(problem.model_dimensions[rjess_state.model_index])
    Σ = get_model_covariance(problem.nested_structure, rjess_state.model_index)

    prior = MvNormal(μ, Σ)
    ess_model = ESSModel(prior, ll)
    initial_ll = ll(rjess_state.params)
    ess_state = ESSState(rjess_state.params, initial_ll)
    new_params = AbstractMCMC.step(Random.default_rng(), ess_model, ESS(), ess_state)[1]
    
    return RJESSState(
        rjess_state.model_index,  # Keep the same model index
        new_params,
        ll(new_params),
        logpdf(prior, new_params)
    )
end

function sample_initial_state(
    problem::RJESSProblem{T},
    model_index::Int
) where T<:Real
    # Get the dimension and covariance for the current model
    μ = zeros(problem.model_dimensions[model_index])
    Σ = get_model_covariance(problem.nested_structure, model_index)
    
    # Sample from multivariate normal prior
    prior = MvNormal(μ, Σ)
    return rand(prior)
end