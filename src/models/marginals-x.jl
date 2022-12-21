##################################
#Base model
struct VolatilityMarginal <: AbstractMarginals end

param_volatilitymarginal = (;
    μᵥ = Param(0.1, truncated(Normal(0.0, 10.0^1), 0.0, 1.0)),
    κ = Param(2.0, truncated(Normal(1.0, 10.0^2), 0.0, 10.0)),
    σ = Param(0.5, truncated(Normal(0.5, 10.0^2), 0.01, 5.0)),
    # T marginal parameter ~ reparametrized as log(ν-2)
    ℓν = Param(0.0,  truncated( Normal(0.0, 10.0^2), -10.0, 10.)),
    #Initial S_0 and X_0
    X₀ = Param(log(0.1), Fixed()),
    #Time discretization parameter ; set to 1 if daily data
    δ = Param(1.0/252.0, Fixed()),
    # Fixed volatility data
    S = Param( getindex.(data_real, 1), Fixed())
)
volatilitymarginal = ModelWrapper(VolatilityMarginal(), param_volatilitymarginal)

################################################################################
# Compute simulation and logpdf methods
function ModelWrappers.simulate(_rng::Random.AbstractRNG, model::ModelWrapper{<:VolatilityMarginal}, Nsamples = 1000)
    @unpack μᵥ, κ, σ, δ, ℓν, S, X₀ = model.val
    ν = exp(ℓν)+2
    # Sample error term
    marginal = Distributions.TDist.(ν)
    ξ = rand(marginal, Nsamples)
    # Then choose X_o
    Sₜ₋₁ = S[1]
    Xₜ₋₁ = X₀
    # Preallocate X
    X = zeros(Float64, Nsamples)
    #Iterate over time
    sqrt_σ_scaled = sqrt( δ*(ν-2)/ν )
    for iter in Base.OneTo(Nsamples)
        X[iter] = Xₜ₋₁ + δ * ( (κ * ( μᵥ - exp(Xₜ₋₁) ) - 0.5*σ^2 ) / exp(Xₜ₋₁) ) + σ * sqrt_σ_scaled * exp(-Xₜ₋₁/2) * ξ[iter]
        Xₜ₋₁ = X[iter]
        Sₜ₋₁ = S[iter]
    end
    #Return X and error terms
    return X, ξ
end

function Data_to_Error_VolatilityMarginal(θ::NamedTuple, data)
    @unpack μᵥ, κ, σ, δ, ℓν, S, X₀ = θ
    ν = exp(ℓν)+2
# first assign temperary container to define error terms from real data
    errorterms = zeros(eltype( promote(μᵥ, κ, σ, δ, ν) ), length(data))
    Sₜ₋₁ = S[1]
    Xₜ₋₁ = X₀
    sqrt_σ_scaled = sqrt( δ*(ν-2)/ν )
# Obtain error terms given current parameter
    for t in 1:length(data)
        errorterms[t] = ( (data[t] - Xₜ₋₁) - δ * ( (κ * ( μᵥ - exp(Xₜ₋₁) ) - 0.5*σ^2 ) / exp(Xₜ₋₁) ) ) / (σ * sqrt_σ_scaled * exp(-Xₜ₋₁/2) )
        Xₜ₋₁ = data[t]
        Sₜ₋₁ = S[t]
    end
    return errorterms
end

function (objective::Objective{<:ModelWrapper{<:VolatilityMarginal}})(θ::NamedTuple)
    @unpack model, data, tagged = objective
## Prior
    lp = log_prior(tagged.info.constraint, ModelWrappers.subset(θ, tagged.parameter) )
## Obtain ϵ from X
    errors = Data_to_Error_VolatilityMarginal(θ, data)
    ν = exp(θ.ℓν)+2
    dist = Distributions.TDist(ν)
## Add Jacobian for Marginal
    jacobian = sum(logpdf(dist, errors[iter]) for iter in eachindex(errors) ) + -length(data)*log(θ.σ)
## Return log density
    return lp + jacobian
end

function ModelWrappers.predict(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{M}}) where {M<:VolatilityMarginal}
    @unpack model, data = objective
    @unpack μᵥ, κ, σ, δ, ℓν, S, X₀ = model.val
    ν = exp(ℓν)+2
    # Sample error term
    marginal = Distributions.TDist.(ν)
    ξ = rand(marginal)
    # Then choose X_o
    Sₜ₋₁ = S[end]
    Xₜ₋₁ = data[end]
    sqrt_σ_scaled = sqrt( δ*(ν-2)/ν )
    X = Xₜ₋₁ + δ * ( (κ * ( μᵥ - exp(Xₜ₋₁) ) - 0.5*σ^2 ) / exp(Xₜ₋₁) ) + σ * sqrt_σ_scaled * exp(-Xₜ₋₁/2) * ξ
    #Return X and error terms
    return X, ξ
end
