module ReversibleSlices

using Distributions
using LinearAlgebra
using Random
using EllipticalSliceSampling
using StatsBase
using AbstractMCMC

include("types.jl")           # New file with shared type definitions
include("nested_models.jl")
include("problem_specification.jl")
include("model_jumps.jl")
include("sampler.jl")

export RJESSProblem, RJESSOptions, run_sampler

end # module ReversibleSlices 