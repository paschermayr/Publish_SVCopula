################################################################################
using Distributions, ForwardDiff, ReverseDiff
import Distributions: cdf, logpdf, quantile, TDist
using ChainRulesCore
import ChainRulesCore: rrule
using StatsBase

# Reverse mode pullback rule for cdf(d,x) dx
function ChainRulesCore.rrule(::typeof(cdf), d::T, x
) where {T<:Distributions.TDist}
    ∇cdf(x) = Distributions.pdf(d, x) # Gradient w.r.t. x
    val  = cdf(d, x) #check_if_trackedarray(x) )
    function _pullback(Δy)
        # Only return differential w.r.t. x, keep d parameter as NoTangent
        return NoTangent(), NoTangent(), ∇cdf(x) * Δy
    end
    return val, _pullback
end
#!NOTE: Need to opt in for ReverseDiff in order to use rrule
#!NOTE - T Marginals with custom pullback for Reversediff makes using Cached ReverseDiff INCORRECT! Need to use untaped ReverseDiff
ReverseDiff.@grad_from_chainrules cdf(d::Main.Distributions.TDist, x::TrackedReal)

################################################################################
#Load libraries
import BaytesCore: ByRows, ByCols, ProposalTune
using ModelWrappers, BaytesFilters, BaytesMCMC, BaytesSMC, Baytes, BaytesInference
using Plots, Random, BenchmarkTools, UnPack, ArgCheck
using JLD2
using LinearAlgebra, Statistics
using Copulas

# Needed Gamma function for T-distribution Copula likelihood
using SpecialFunctions
function TDist(df::R) where {R<:ReverseDiff.TrackedReal}
    return TDist(ReverseDiff.value(df))
end
