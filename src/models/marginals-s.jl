##################################
#Base model
struct StockMarginal <: AbstractMarginals end

param_stockmarginal = (;
    μₛ = Param(0.1,  truncated(Normal(0.0, 10.0^1), 0.0, 1.0)),
    # T marginal parameter ~ reparametrized as log(ν-2)
    ℓν = Param(0.0,  truncated( Normal(0.0, 10.0^3), -10.0, 10.)),
    #Initial S_0 and X_0
    S₀ = Param(8.5, Fixed()),
    #Time discretization parameter ; set to 1 if daily data
    δ = Param(1.0/252.0, Fixed()),
    # Fixed volatility data
    X = Param( getindex.(data_real, 2), Fixed())
)
stockmarginal = ModelWrapper(StockMarginal(), param_stockmarginal)

################################################################################
# Compute simulation and logpdf methods
function ModelWrappers.simulate(_rng::Random.AbstractRNG, model::ModelWrapper{<:StockMarginal}, Nsamples = 1000)
    @unpack μₛ, δ, ℓν, X = model.val
    ν = exp(ℓν)+2
    # Sample error term
    marginal = Distributions.TDist.(ν)
    ϵ = rand(marginal, Nsamples)
    # Then choose S_o
    @unpack S₀= model.val
    Sₜ₋₁ = S₀
    Xₜ₋₁ = X[1]
    # Preallocate S
    S = zeros(Float64, Nsamples)
    #Iterate over time
    sqrt_σ_scaled = sqrt( δ*(ν-2)/ν )
    for iter in Base.OneTo(Nsamples)
        S[iter] = Sₜ₋₁ + δ * ( μₛ - exp(Xₜ₋₁)/2) + sqrt_σ_scaled * exp(Xₜ₋₁/2) * ϵ[iter]
        Xₜ₋₁ = X[iter]
        Sₜ₋₁ = S[iter]
    end
    #Return S and error terms
    return S, ϵ
end

function Data_to_Error_StockMarginal(θ::NamedTuple, data)
    @unpack μₛ, δ, ℓν, X = θ
    ν = exp(ℓν)+2
# first assign temperary container to define error terms from real data
    errorterms = zeros(eltype( promote(μₛ, δ, ν) ), length(data))
    @unpack S₀ = θ
    Sₜ₋₁ = S₀
    Xₜ₋₁ = X[1]
    sqrt_σ_scaled = sqrt( δ*(ν-2)/ν )
# Obtain error terms given current parameter
    for t in 1:length(data)
        errorterms[t] = ( (data[t] - Sₜ₋₁) - δ * ( μₛ - exp(Xₜ₋₁)/2) ) / ( sqrt_σ_scaled * exp(Xₜ₋₁/2) )
        Xₜ₋₁ = X[t]
        Sₜ₋₁ = data[t]
    end
    return errorterms
end

function (objective::Objective{<:ModelWrapper{<:StockMarginal}})(θ::NamedTuple)
    @unpack model, data, tagged = objective
## Prior
    lp = log_prior(tagged.info.constraint, ModelWrappers.subset(θ, tagged.parameter) )
## Obtain ϵ from S
    errors = Data_to_Error_StockMarginal(θ, data)
    ν = exp(θ.ℓν)+2
    dist = Distributions.TDist(ν)
## Add Jacobian for Marginal
    jacobian = sum(logpdf(dist, errors[iter]) for iter in eachindex(errors) ) #+ -length(data)*log(θ.σ)
## Return log density
    return lp + jacobian
end

function ModelWrappers.predict(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{M}}) where {M<:StockMarginal}
    @unpack model, data = objective
    @unpack S₀, ℓν, X, μₛ, δ = model.val
    ν = exp(ℓν)+2
    # Sample error term
    marginal = Distributions.TDist.(ν)
    ϵ = rand(marginal)
    # Then choose S_o and X_0
    Sₜ₋₁ = data[end]
    Xₜ₋₁ = X[end]
    sqrt_σ_scaled = sqrt( δ*(ν-2)/ν )
    S = Sₜ₋₁ + δ * ( μₛ - exp(Xₜ₋₁)/2) + sqrt_σ_scaled * exp(Xₜ₋₁/2) * ϵ
    #Return S and error terms
    return S, ϵ
end
