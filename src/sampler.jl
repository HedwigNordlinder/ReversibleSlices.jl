using EllipticalSliceSampling: ESSModel, ESSState, ESS
struct JumpInfo{T<:Real}
    from_model::Int
    to_model::Int
    log_likelihood_ratio::T
    log_prior_ratio::T
    log_proposal_ratio::T  # log q term
    acceptance_ratio::T
    accepted::Bool
end


# Helper function to summarise jump diagnostics
function summarise_jumps(jump_history)
    n_models = maximum(max(j.from_model, j.to_model) for j in jump_history)
    
    # Initialize counters
    attempts = zeros(Int, n_models, n_models)
    accepts = zeros(Int, n_models, n_models)
    avg_acc_ratio = zeros(n_models, n_models)
    
    # Collect statistics
    for jump in jump_history
        attempts[jump.from_model, jump.to_model] += 1
        accepts[jump.from_model, jump.to_model] += jump.accepted
        avg_acc_ratio[jump.from_model, jump.to_model] += jump.acceptance_ratio
    end
    
    # Print summary
    println("\nJump Diagnostics:")
    println("-----------------")
    for i in 1:n_models
        for j in 1:n_models
            if attempts[i,j] > 0
                acc_rate = accepts[i,j] / attempts[i,j]
                mean_acc_ratio = avg_acc_ratio[i,j] / attempts[i,j]
                println("M$i → M$j: $(accepts[i,j])/$(attempts[i,j]) accepted " *
                       "($(round(acc_rate*100, digits=1))%) " *
                       "mean acc ratio: $(round(mean_acc_ratio, digits=3))")
            end
        end
    end
end
function plot_transition_graph(jump_history; min_attempts=5)
    # Get number of models
    n_models = maximum(max(j.from_model, j.to_model) for j in jump_history)
    
    # Initialize matrices
    attempts = zeros(Int, n_models, n_models)
    accepts = zeros(Int, n_models, n_models)
    
    # Collect statistics
    for jump in jump_history
        attempts[jump.from_model, jump.to_model] += 1
        accepts[jump.from_model, jump.to_model] += jump.accepted
    end
    
    # Create directed graph
    g = SimpleDiGraph(n_models)
    
    # Add edges where we have sufficient attempts
    edge_colors = []
    edge_widths = []
    edge_labels = String[]
    
    for i in 1:n_models
        for j in 1:n_models
            if attempts[i,j] ≥ min_attempts
                add_edge!(g, i, j)
                acc_rate = accepts[i,j] / attempts[i,j]
                
                # Color based on acceptance rate
                push!(edge_colors, acc_rate)
                
                # Width based on number of attempts
                push!(edge_widths, log(1 + attempts[i,j]))
                
                # Label with acceptance rate percentage
                push!(edge_labels, "$(round(acc_rate*100, digits=1))%")
            end
        end
    end
    
    # Create figure
    fig = Figure(resolution=(1000, 1000))
    ax = Axis(fig[1,1], aspect=DataAspect())
    
    # Plot graph
    graphplot!(ax, g, 
        layout=NetworkLayout.spring,  # Pass the layout function directly
        node_color=:lightblue,
        node_size=30,
        edge_color=edge_colors,
        edge_width=edge_widths,
        edge_label=edge_labels,
        edge_label_size=10,
        nlabels=["M$i" for i in 1:n_models],
        arrow_size=15)
    
    # Add colorbar
    Colorbar(fig[1,2], limits=(0,1), label="Acceptance Rate")
    
    hidedecorations!(ax)
    hidespines!(ax)
    
    return fig
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
        removed_part = current_params[(dim_to+1):dim_from]
        log_q = logpdf(MvNormal(cond_mean, cond_cov), removed_part)
    end
    
    return log_q
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

function rj_ess(problem::RJESSProblem{T}; n_samples::Int64=1000, n_burnin::Int64=100,
    model_switching_probability::Float64=0.3) where {T<:Real}
    current_model = rand(1:problem.n_models)
    current_params = sample_initial_state(problem, current_model)
    samples = Vector{Vector{Float64}}(undef, n_samples + n_burnin)
    model_indices = Vector{Int}(undef, n_samples + n_burnin)
    logposteriors = Vector{Float64}(undef, n_samples + n_burnin)
    
    # Store jump diagnostics
    jump_history = Vector{JumpInfo{T}}()
    
    # Initialize progress meter
    prog = Progress(n_samples + n_burnin, dt=0.5, desc="Running RJESS: ", barglyphs=BarGlyphs("[=> ]"))
    
    for i in 1:(n_samples+n_burnin)
        if rand() < model_switching_probability
            available_models = [min(current_model+1, problem.n_models), max(current_model-1, 1)]
            proposed_model = rand(available_models)
            proposal = if proposed_model > current_model
                propose_up_jump(problem, current_model, proposed_model, current_params)
            else
                propose_down_jump(problem, current_model, proposed_model, current_params)
            end
            
            # Calculate components for acceptance ratio
            proposal_log_likelihood = problem.loglikelihood(proposal.proposed_params)
            proposal_log_prior = log_prior_density(problem, proposed_model, proposal.proposed_params)
            current_log_likelihood = problem.loglikelihood(current_params)
            current_log_prior = log_prior_density(problem, current_model, current_params)
            log_q = calculate_proposal_density(problem, current_model, proposed_model,
                current_params, proposal.proposed_params)
            
            # Calculate acceptance ratio
            log_likelihood_ratio = proposal_log_likelihood - current_log_likelihood
            log_prior_ratio = proposal_log_prior - current_log_prior
            log_proposal_ratio = proposed_model < current_model ? log_q : -log_q
            acceptance_ratio = log_likelihood_ratio + log_prior_ratio + log_proposal_ratio
            
            original_model = current_model

            # Accept/reject step
            accepted = log(rand()) < acceptance_ratio
            if accepted
                current_model = proposed_model
                current_params = proposal.proposed_params
            end
            
            # Store jump info using original_model
            push!(jump_history, JumpInfo{T}(
                original_model,     # Use original model index
                proposed_model,
                log_likelihood_ratio,
                log_prior_ratio,
                log_proposal_ratio,
                acceptance_ratio,
                accepted
            ))
        else
            current_params = sample_within_model(problem, current_params, current_model)
        end
        
        samples[i] = current_params
        logposteriors[i] = problem.loglikelihood(current_params)
        model_indices[i] = current_model
        
        # Update progress bar
        next!(prog)
    end
    
    
    return samples, model_indices, logposteriors, jump_history
end

