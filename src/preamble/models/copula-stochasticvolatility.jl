##################################
#Base model
#=
param_stochvol = (;
    # SDE
    μₛ = Param(truncated(Normal(10.0, 10.0^4), -1.0, 1.0), 0.1, ),
    #!NOTE: This is the mean value that variance = exp(X) > 0 will shift around
    μᵥ = Param(truncated(Normal(10.0, 10.0^4), 0.0, 1.0), exp(data_real[begin].X), ), #exp(data_real[begin].X)  = 0.027
    #!NOTE: Mean reversion parameter for volatility parameter X
    κ = Param(truncated(Normal(10.0, 10.0^4), 0.0, 100.0), 1.00, ),
    #!NOTE Noise term > 0 for both S (in sqrt(σ)) and X (in σ)
    σ = Param(truncated(Normal(0.5, 10.0^4), 0.01, 2.0), 0.10),
# Standard Parametric Copulas
    copula = (;
        ρ = Param(truncated(Normal(0.0, 10^5), -1.0, 1.0), _rho, ),
        df = Param(Fixed(), 3)
    ),
    #Initial S_0 and X_0
    S₀ = Param(Fixed(), data_real[begin].S, ),
    X₀ = Param(Fixed(), data_real[begin].X,), #3.0
    #Time discretization parameter ; set to 1 if daily data
    δ = Param(Fixed(), 1.0/252.0, ),
)
stochvolcopula = ModelWrapper(StochasticVolatilityCopula(), param_stochvol,
    (; copulanames = TCop(), subcopulas = TCop(), rotation = _archimedeanrotation, marginals = _marginals)
)
=#

################################################################################
# Compute simulation and logpdf methods
function ModelWrappers.simulate(_rng::Random.AbstractRNG, model::ModelWrapper{<:StochasticVolatilityCopula}, Nsamples = 1000)
    # First sample from Copula
    @unpack marginals = model.arg
    if model.arg.copulanames isa FactorCopula
        latent, dataᵤ = simulate(_rng, model.arg.copulanames, model.arg.rotation, model.val.copula, model.arg.subcopulas, Nsamples)
    else
        dataᵤ = simulate(_rng, model.arg.copulanames, model.arg.rotation, model.val.copula, Nsamples)
    end
    #data is in uniform space - transform to real space with custom marginals.
    #!NOTE: Treat data[1,:] as ϵ and data[2,:] as ξ
    ϵ = quantile.(marginals[1], dataᵤ[1,:])
    ξ = quantile.(marginals[2], dataᵤ[2,:])
    # Then choose S_o and X_o
    @unpack S₀, X₀ = model.val
    Sₜ₋₁, Xₜ₋₁ = S₀, X₀
    # Preallocate S and X
    S = zeros(Float64, Nsamples)
    X = zeros(Float64, Nsamples)
    #Iterate over time
    @unpack μₛ, μᵥ, κ, σ, δ = model.val
    for iter in Base.OneTo(Nsamples)
        X[iter] = Xₜ₋₁ + δ * ( (κ * ( μᵥ - exp(Xₜ₋₁) ) - 0.5*σ^2 ) / exp(Xₜ₋₁) ) + σ * sqrt(δ) * exp(-Xₜ₋₁/2) * ξ[iter]
        S[iter] = Sₜ₋₁ + δ * ( μₛ - exp(Xₜ₋₁)/2) + sqrt(δ) * exp(Xₜ₋₁/2) * ϵ[iter]
        Xₜ₋₁ = X[iter]
        Sₜ₋₁ = S[iter]
    end
    #Return S, X as Vector of NamedTuple so can use it with PF and SMC
    if model.arg.copulanames isa FactorCopula
        return [(S = S[iter], X = X[iter]) for iter in eachindex(S)], dataᵤ, (ϵ, ξ), latent
    else
        return [(S = S[iter], X = X[iter]) for iter in eachindex(S)], dataᵤ, (ϵ, ξ)
    end
end

function Data_to_Error(θ::NamedTuple, data)
    @unpack μₛ, μᵥ, κ, σ, δ = θ
# first assign temperary container to define error terms from real data
    errorterms = zeros(eltype( promote(μₛ, μᵥ, κ, σ, δ) ), 2, length(data))
    @unpack S₀, X₀ = θ
    Sₜ₋₁, Xₜ₋₁ = S₀, X₀
# Obtain error terms given current parameter
    for t in 1:length(data)
        errorterms[2,t] = ( (data[t].X - Xₜ₋₁) - δ * ( (κ * ( μᵥ - exp(Xₜ₋₁) ) - 0.5*σ^2 ) / exp(Xₜ₋₁) ) ) / (σ * sqrt(δ) * exp(-Xₜ₋₁/2) )
        errorterms[1,t] = ( (data[t].S - Sₜ₋₁) - δ * ( μₛ - exp(Xₜ₋₁)/2) ) / ( sqrt(δ) * exp(Xₜ₋₁/2) )
        Xₜ₋₁ = data[t].X
        Sₜ₋₁ = data[t].S
    end
    return errorterms
end

function to_errorsᵤ(_marginal1::D, _marginal2::E, errors::Matrix{T}) where {D<:Distribution, E<:Distribution, T<:Real}
    errorsᵤ = zeros(T, size(errors))
    for iter in Base.OneTo(size(errors, 2))
        #ϵ
        errorsᵤ[1, iter] = cdf(_marginal1, errors[1, iter])
        #ξ
        errorsᵤ[2, iter] = cdf(_marginal2, errors[2, iter])
    end
    return errorsᵤ
end

function (objective::Objective{<:ModelWrapper{<:StochasticVolatilityCopula}})(θ::NamedTuple)
    @unpack model, data, tagged = objective
    @unpack marginals = model.arg
## Prior
    lp = log_prior(tagged.info.transform.constraint, ModelWrappers.subset(θ, tagged.parameter) )
## Obtain ϵ and ξ from S and X
    errors = Data_to_Error(θ, data)
## Take cdf of error terms to go to uniform space
    errorsᵤ = to_errorsᵤ(marginals[1], marginals[2], errors)
## Compute likelihood of copula
    ll = cumℓlikelihood(objective.model.arg.subcopulas, objective.model.arg.rotation, θ.copula, errorsᵤ)
    if (isnan(ll))
        return -Inf
    end
## Add Jacobian
    jacobian = 0.0
    jacobian += -length(data)*log(θ.σ)
    jacobian += sum(logpdf(marginals[1], errors[1, iter]) for iter in Base.OneTo( size(errors, 2) ) )
    jacobian += sum(logpdf(marginals[2], errors[2, iter]) for iter in Base.OneTo( size(errors, 2) ) )
## Return log density
    return ll + lp + jacobian
end

function ModelWrappers.predict(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{M}}) where {M<:StochasticVolatilityCopula}
    @unpack model, data = objective
    @unpack marginals = model.arg
    #Sample the error terms (in uniform dimension) given the Copula parameter
    if model.arg.copulanames isa FactorCopula
        latent, dataᵤ = simulate(_rng, model.arg.copulanames, model.arg.rotation, model.val.copula, model.arg.subcopulas, 1)
    else
        dataᵤ = simulate(_rng, model.arg.copulanames, model.arg.rotation, model.val.copula, 1)
    end
    #Transform to real dimension
    ϵ = quantile(marginals[1], dataᵤ[1])
    ξ = quantile(marginals[2], dataᵤ[2])
    # Then choose S_o and X_o
    Sₜ, Xₜ = data[end].S, data[end].X
    #Set X_t+1 and S_t+1 given the current parameter and error terms
    @unpack μₛ, μᵥ, κ, σ, δ = model.val
    Xₜ₊₁ = Xₜ + δ * ( (κ * ( μᵥ - exp(Xₜ) ) - 0.5*σ^2 ) / exp(Xₜ) ) + σ * sqrt(δ) * exp(-Xₜ/2) * ξ
    Sₜ₊₁ = Sₜ + δ * ( μₛ - exp(Xₜ)/2) + sqrt(δ) * exp(Xₜ/2) * ϵ
    return (S = Sₜ₊₁, X = Xₜ₊₁)
end

import BaytesSMC: SMCweight
function SMCweight(_rng::Random.AbstractRNG, algorithm, objective::Objective{<:ModelWrapper{M}}, proposaltune::P, cumweightsₜ₋₁) where {M<:StochasticVolatilityCopula, P<:ProposalTune}
    lpost = objective.temperature * objective(objective.model.val)
    #Remove prior computation
    lp = objective.temperature * log_prior(objective.tagged.info.transform.constraint, ModelWrappers.subset(objective.model.val, objective.tagged.parameter) )
    cumweightsₜ = lpost - lp
    #!NOTE: cumweightsₜ₋₁ already has correct temperature at t-1
    return cumweightsₜ, cumweightsₜ - cumweightsₜ₋₁
end
