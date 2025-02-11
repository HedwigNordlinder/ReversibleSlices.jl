using EllipticalSliceSampling: ESSModel, ESSState, ESS
# Add this struct at the top level
struct JumpDiagnostics{T<:Real}
    from_model::Int
    to_model::Int
    current_params::Vector{T}
    proposed_params::Vector{T}
    log_prior_ratio::T
    log_jacobian::T
    acceptance_ratio::T
    conditional_mean::Union{Nothing,Vector{T}}
    conditional_cov::Union{Nothing,Matrix{T}}
end

# Add to RJESSProblem struct
mutable struct DiagnosticCollector{T<:Real}
    diagnostics::Vector{JumpDiagnostics{T}}
    max_stored::Int  # Maximum number of diagnostics to store
    current_idx::Int

    DiagnosticCollector{T}(max_stored::Int=1000) where {T} = new{T}(
        Vector{JumpDiagnostics{T}}(), max_stored, 0
    )
end

function debug_densities(problem::RJESSProblem, from_model::Int, to_model::Int,
    current_params::Vector{T}, proposed_params::Vector{T}) where {T<:Real}

    if to_model > from_model
        # Up jump case
        dim_from = problem.model_dimensions[from_model]
        Σ = problem.nested_structure.full_cov
        Σ11 = Σ[1:dim_from, 1:dim_from]

        # Calculate each term separately
        marg_density = logpdf(MvNormal(zeros(dim_from), Σ11), current_params)

        cond_mean, cond_cov = get_conditional_distribution(
            problem.nested_structure, from_model, to_model, current_params)
        cond_density = logpdf(MvNormal(cond_mean, cond_cov),
            proposed_params[(dim_from+1):end])

        joint_density = log_prior_density(problem, to_model, proposed_params)

        println("Up jump M_$(from_model) → M_$(to_model):")
        println("log p(θ₁) = ", marg_density)
        println("log p(θ₂|θ₁) = ", cond_density)
        println("log p(θ₁,θ₂) = ", joint_density)
        println("log p(θ₁) + log p(θ₂|θ₁) = ", marg_density + cond_density)
        println("Difference = ", joint_density - (marg_density + cond_density))
    else
        # Down jump case
        dim_to = problem.model_dimensions[to_model]
        proposed_subset = proposed_params[1:dim_to]
        removed_params = current_params[(dim_to+1):end]

        marg_density = log_prior_density(problem, to_model, proposed_subset)

        cond_mean, cond_cov = get_conditional_distribution(
            problem.nested_structure, to_model, from_model, proposed_subset)
        cond_density = logpdf(MvNormal(cond_mean, cond_cov), removed_params)

        joint_density = log_prior_density(problem, from_model, current_params)

        println("Down jump M_$(from_model) → M_$(to_model):")
        println("log p(θ₁) = ", marg_density)
        println("log p(θ₂|θ₁) = ", cond_density)
        println("log p(θ₁,θ₂) = ", joint_density)
        println("log p(θ₁) + log p(θ₂|θ₁) = ", marg_density + cond_density)
        println("Difference = ", joint_density - (marg_density + cond_density))
    end
end

function nested_normal_diagnostics(problem::RJESSProblem{T},
    from_model::Int,
    to_model::Int,
    current_params::Vector{T},
    proposal::ModelJumpProposal{T}) where {T<:Real}

    println("\nDiagnostic for move from M_$from_model to M_$to_model:")

    # Print full covariance matrix
    println("\nFull covariance matrix:")
    display(problem.nested_structure.full_cov)

    # Current state diagnostics
    println("\nCurrent state θ₁: ", current_params)

    # Get dimensions
    dim_from = problem.model_dimensions[from_model]
    dim_to = problem.model_dimensions[to_model]

    # Get blocks of covariance matrix
    Σ = problem.nested_structure.full_cov

    if to_model > from_model
        # Up jump case
        Σ11 = Σ[1:dim_from, 1:dim_from]
        Σ12 = Σ[1:dim_from, (dim_from+1):dim_to]
        Σ22 = Σ[(dim_from+1):dim_to, (dim_from+1):dim_to]

        # Get conditional distribution
        cond_mean, cond_cov = get_conditional_distribution(
            problem.nested_structure, from_model, to_model, current_params)

        println("\nConditional distribution used for proposal:")
        println("Mean: ", cond_mean)
        println("Covariance: ")
        display(cond_cov)

        # Proposed new parameters
        println("\nProposed θ₂: ", proposal.proposed_params[(dim_from+1):end])

        # Evaluate densities
        joint_density = logpdf(MvNormal(zeros(dim_to), Σ[1:dim_to, 1:dim_to]),
            proposal.proposed_params)
        marg_density = logpdf(MvNormal(zeros(dim_from), Σ11), current_params)
        cond_density = logpdf(MvNormal(cond_mean, cond_cov),
            proposal.proposed_params[(dim_from+1):end])
    else
        # Down jump case
        Σ11 = Σ[1:dim_to, 1:dim_to]
        proposed_subset = proposal.proposed_params

        # Get conditional distribution for reverse move
        cond_mean, cond_cov = get_conditional_distribution(
            problem.nested_structure, to_model, from_model, proposed_subset)

        println("\nConditional distribution for reverse move:")
        println("Mean: ", cond_mean)
        println("Covariance: ")
        display(cond_cov)

        # Evaluate densities
        joint_density = logpdf(MvNormal(zeros(dim_from), Σ[1:dim_from, 1:dim_from]),
            current_params)
        marg_density = logpdf(MvNormal(zeros(dim_to), Σ11), proposed_subset)
        cond_density = logpdf(MvNormal(cond_mean, cond_cov),
            current_params[(dim_to+1):end])
    end

    println("\nDensity evaluations:")
    println("Joint log-density: ", joint_density)
    println("Marginal log-density: ", marg_density)
    println("Conditional log-density: ", cond_density)

    # Theoretical vs actual marginal check
    target_dim = to_model > from_model ? dim_from : dim_to
    θ1_under_joint = zeros(T, 1000)
    θ1_direct = zeros(T, 1000)

    for i in 1:1000
        # Sample from joint and take first component
        full_sample = rand(MvNormal(zeros(target_dim), Σ[1:target_dim, 1:target_dim]))
        θ1_under_joint[i] = full_sample[1]  # Just take the first component

        # Sample directly from marginal
        θ1_direct[i] = rand(Normal(0, sqrt(Σ[1,1])))  # Sample from univariate normal
    end

    println("\nMarginal distribution check:")
    println("Mean of θ₁ under joint: ", mean(θ1_under_joint))
    println("Mean of θ₁ direct: ", mean(θ1_direct))
    println("Var of θ₁ under joint: ", var(θ1_under_joint))
    println("Var of θ₁ direct: ", var(θ1_direct))
end

function calculate_proposal_density(
    problem::RJESSProblem{T}, 
    from_model::Int, 
    to_model::Int,
    current_params::Vector{T},
    proposed_params::Vector{T}) where T<:Real
    
    dim_from = problem.model_dimensions[from_model]
    dim_to = problem.model_dimensions[to_model]
    
    if to_model > from_model
        # Up move - calculate q(θ₂|θ₁)
        cond_mean, cond_cov = get_conditional_distribution(
            problem.nested_structure, 
            from_model, 
            to_model, 
            current_params
        )
        # Get proposed θ₂ parameters
        proposed_part = proposed_params[(dim_from+1):dim_to]
        log_q = logpdf(MvNormal(cond_mean, cond_cov), proposed_part)
    else
        # Down move - calculate q(θ₂|θ₁) for reverse move
        cond_mean, cond_cov = get_conditional_distribution(
            problem.nested_structure, 
            to_model, 
            from_model, 
            proposed_params
        )
        # Get removed θ₂ parameters
        removed_part = current_params[(dim_to+1):dim_from]
        log_q = logpdf(MvNormal(cond_mean, cond_cov), removed_part)
    end
    
    return log_q
end

function rj_ess(problem::RJESSProblem{T}; n_samples::Int64=1000, n_burnin::Int64=100, model_switching_probability::Float64=0.3) where {T<:Real}
    # First we need to pick a random model to start in
    current_model = rand(1:problem.n_models)
    current_params = sample_initial_state(problem, current_model)

    samples = Vector{Vector{Float64}}(undef, n_samples + n_burnin)
    model_indices = Vector{Int}(undef, n_samples + n_burnin)

    # Initialize diagnostics
    n_models = length(problem.model_dimensions)
    proposed_jumps = zeros(Int, n_models, n_models)  # proposed_jumps[from,to]
    accepted_jumps = zeros(Int, n_models, n_models)  # accepted_jumps[from,to]
    diagnostic_collector = DiagnosticCollector{T}()
    for i in 1:(n_samples+n_burnin)
        if rand() < model_switching_probability
            # Propose jumps to adjacent models with higher probability
            available_models = setdiff(1:problem.n_models, current_model)
            proposed_model = rand(available_models)

            if proposed_model > current_model
                # Sample a new parameter vector
                proposal = propose_up_jump(problem, current_model, proposed_model, current_params)
            else
                # Sample a new parameter vector
                proposal = propose_down_jump(problem, current_model, proposed_model, current_params)
            end

            # Update diagnostics for proposal
            proposed_jumps[proposal.from_index, proposal.to_index] += 1

            proposal_log_likelihood = problem.loglikelihood(proposal.proposed_params)
            proposal_log_prior = log_prior_density(problem, proposed_model, proposal.proposed_params)

            # debug_densities(problem, current_model, proposed_model, current_params, proposal.proposed_params)

            current_log_likelihood = problem.loglikelihood(current_params)
            current_log_prior = log_prior_density(problem, current_model, current_params)

            log_q = calculate_proposal_density(problem, current_model, proposed_model, 
                                 current_params, proposal.proposed_params)


            # Calculate acceptance ratio
            acceptance_ratio = (proposal_log_likelihood - current_log_likelihood) +  # Likelihood ratio
                  (proposal_log_prior - current_log_prior) +             # Prior ratio
                  (proposed_model < current_model ? log_q : -log_q)      # Proposal density
            # Then in the jump proposal section, replace printing with:
            if proposal_log_likelihood == 0 && current_log_likelihood == 0

                # Only collect diagnostics when sampling from prior
                log_prior_ratio = proposal_log_prior - current_log_prior

                # Get conditional distribution info for up-jumps
                cond_mean = nothing
                cond_cov = nothing
                if proposal.to_index > proposal.from_index
                    cond_mean, cond_cov = get_conditional_distribution(
                        problem.nested_structure,
                        proposal.from_index,
                        proposal.to_index,
                        current_params
                    )
                end

                diagnostics = JumpDiagnostics(
                    proposal.from_index,
                    proposal.to_index,
                    copy(current_params),
                    copy(proposal.proposed_params),
                    log_prior_ratio,
                    proposal.log_jacobian,
                    acceptance_ratio,
                    cond_mean,
                    cond_cov
                )
                if rand() < 0.5  # Only print diagnostics occasionally
                    # nested_normal_diagnostics(problem, current_model, proposed_model, current_params, proposal)
                end

                if diagnostic_collector.current_idx < diagnostic_collector.max_stored
                    diagnostic_collector.current_idx += 1
                    push!(diagnostic_collector.diagnostics, diagnostics)
                end
            end


            if log(rand()) < acceptance_ratio
                # Update diagnostics for acceptance
                accepted_jumps[proposal.from_index, proposal.to_index] += 1
                current_model = proposed_model
                current_params = proposal.proposed_params
            end
            samples[i] = current_params
            model_indices[i] = current_model
        else
            # Perform a within-model update
            current_params = sample_within_model(problem, current_params, current_model)
            samples[i] = current_params
            model_indices[i] = current_model
        end
    end

    # Print diagnostics after burn-in
    println("\nCross-model move diagnostics:")
    for i in 1:n_models
        for j in 1:n_models
            if proposed_jumps[i, j] > 0
                acceptance_rate = accepted_jumps[i, j] / proposed_jumps[i, j]
                println("M_$i → M_$j: $(accepted_jumps[i,j])/$(proposed_jumps[i,j]) ($(round(acceptance_rate*100, digits=1))%)")
            end
        end
    end

    return samples, model_indices, diagnostic_collector
end

function sample_within_model(problem::RJESSProblem, current_params::Vector{T}, model_index::Int) where {T<:Real}
    ll = ϕ -> problem.loglikelihood(ϕ)
    μ = zeros(problem.model_dimensions[model_index])
    Σ = get_model_covariance(problem.nested_structure, model_index)
    prior = MvNormal(μ, Σ)
    ess_model = ESSModel(prior, ll)
    ess_state = ESSState(current_params, ll(current_params))
    return AbstractMCMC.step(Random.default_rng(), ess_model, ESS(), ess_state)[1]
end