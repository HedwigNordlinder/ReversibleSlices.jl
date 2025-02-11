module ReversibleSlices

using Distributions
using LinearAlgebra
using Random
using EllipticalSliceSampling
using AbstractMCMC
using ProgressMeter

include("types.jl")           # New file with shared type definitions
include("nested_models.jl")
include("problem_specification.jl")
include("model_jumps.jl")
include("sampler.jl")

export RJESSProblem, RJESSOptions, rj_ess, JumpDiagnostics, DiagnosticCollector

end # module ReversibleSlices 