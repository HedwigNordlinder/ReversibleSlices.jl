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
    end
    
    return model_samples, param_samples
end

function model_jump_step(
    problem::RJESSProblem{T}, 
    state::RJESSState{T},
    jump_proposal_dist::Vector{Float64}
    ) where T<:Real
    
    # Get possible jump distances (excluding 0)
    possible_jumps = vcat(
        (-state.model_index+1):-1,  # negative jumps
        1:(problem.n_models-state.model_index)  # positive jumps
    )
    
    # Create probabilities for each possible jump
    jump_probs = Float64[]
    for jump in possible_jumps
        target_index = state.model_index + jump
        if 1 <= target_index <= length(jump_proposal_dist)
            push!(jump_probs, jump_proposal_dist[target_index])
        end
    end
    
    # If no valid jumps are possible, return current state
    if isempty(jump_probs)
        return state
    end
    
    # Normalize probabilities
    jump_probs = jump_probs / sum(jump_probs)
    
    # Filter possible_jumps to match valid probabilities
    valid_jumps = Int[]
    for jump in possible_jumps
        target_index = state.model_index + jump
        if 1 <= target_index <= length(jump_proposal_dist)
            push!(valid_jumps, jump)
        end
    end
    
    # Propose jump distance (will never be 0)
    jump_dist = sample(valid_jumps, ProbabilityWeights(jump_probs))
    
    # Calculate new model index
    new_model_index = state.model_index + jump_dist
    
    # Generate proposal based on relative positions of current and new model
    if jump_dist > 0
        # Moving up to a larger model
        proposal = propose_up_jump(problem, state.model_index, new_model_index, state.params)
    else
        # Moving down to a smaller model
        @assert state.model_index > new_model_index "Invalid down jump: from $(state.model_index) to $(new_model_index)"
        proposal = propose_down_jump(problem, state.model_index, new_model_index, state.params)
    end
    
    # Accept/reject step
    if log(rand()) < proposal.log_ratio
        # Accept jump
        return RJESSState(
            proposal.to_index,
            proposal.proposed_params,
            problem.loglikelihood(proposal.proposed_params),
            log_prior_density(problem, proposal.to_index, proposal.proposed_params)
        )
    else
        # Reject jump - return current state
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