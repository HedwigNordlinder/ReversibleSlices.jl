using ReversibleSlices
using Test
using LinearAlgebra
using Distributions
using Random

@testset "Linear Regression RJESS Tests" begin
    # Set random seed for reproducibility
    Random.seed!(123)

    # Generate synthetic data
    n_points = 100
    x = range(-1, 1, length=n_points)
    
    # True model: y = 2 + 3x + 1.5x² + ε, ε ~ N(0, 0.5²)
    true_y = 2.0 .+ 3.0 .* x .+ 1.5 .* x.^2 + rand(Normal(0, 0.5), n_points)

    # Define the three nested models:
    # M₁: y = β₀
    # M₂: y = β₀ + β₁x
    # M₃: y = β₀ + β₁x + β₂x²

    # Define log likelihood function
    function loglikelihood(params::Vector{Float64})
        if length(params) == 1
            y_pred = fill(params[1], n_points)
        elseif length(params) == 2
            y_pred = params[1] .+ params[2] .* x
        else # length == 3
            y_pred = params[1] .+ params[2] .* x .+ params[3] .* x.^2
        end
        
        return sum(logpdf.(Normal(0, 0.5), true_y .- y_pred))
    end

    # Setup RJESS problem
    model_dimensions = [1, 2, 3]  # Number of parameters in each model
    model_priors = [1/3, 1/3, 1/3]  # Equal prior probabilities

    # Define prior covariance for largest model
    full_covariance = [
        1.0 0.0 0.0
        0.0 1.0 0.0
        0.0 0.0 1.0
    ]

    # Create problem specification
    problem = RJESSProblem(
        loglikelihood,
        full_covariance,
        model_dimensions,
        model_priors
    )

    # Setup sampling options
    options = RJESSOptions(
        1000,           # Number of samples
        0.5,            # Probability of within-model move
        [0.5, 0.5]      # Jump proposal distribution (equal probability of ±1)
    )

    # Run sampler
    model_samples, param_samples = run_sampler(problem, options; initial_model=1)

    # Basic tests
    @test length(model_samples) == options.n_samples
    @test length(param_samples) == options.n_samples
    @test all(1 .<= model_samples .<= 3)

    # Check that the sampler spends most time in model 3 (true model)
    model_frequencies = [count(==(i), model_samples)/options.n_samples for i in 1:3]
    @test model_frequencies[3] > 0.4  # Should spend at least 40% time in true model
    
    # Check parameter estimates for model 3
    model3_samples = [params for (model, params) in zip(model_samples, param_samples) if model == 3]
    if !isempty(model3_samples)
        mean_params = mean(model3_samples)
        @test isapprox(mean_params[1], 2.0, atol=0.5)  # β₀
        @test isapprox(mean_params[2], 3.0, atol=0.5)  # β₁
        @test isapprox(mean_params[3], 1.5, atol=0.5)  # β₂
    end
end 