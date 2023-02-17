####################################################################
#BB1
struct BB1 <: Archimedean end
#=
param_BB1Copula = (;
    theta = Param(truncated(Normal(2.0, 10^5), 0.0001, 100.0), 1., ),
    delta = Param(truncated(Normal(2.0, 10^5), 1.0, 100.0), 2., )
)
bb1copula = ModelWrapper(BB1(), param_BB1Copula, (;reflection = _archimedeanreflection))
length(bb1copula)
# =#
struct BB1Copula{A<:Real}
    theta::A
    delta::A
    function BB1Copula(
        theta::A, delta::A
    ) where {A<:Real}
        ArgCheck.@argcheck theta > 0.0
        ArgCheck.@argcheck delta >= 1.0
        return new{A}(theta, delta)
    end
end
function toCopula(copula::BB1, θ)
    return BB1Copula(θ.theta, θ.delta)
end
function ℓlikelihood(copula::C, θ::NamedTuple, u::AbstractVector) where {C<:BB1}
    @unpack theta, delta = θ
    delta⁻¹ = 1/delta
    theta⁻¹ = 1/theta
    _u = u[1]
    _v = u[2]

    ut = (_u^(-theta) - 1)
    vt = (_v^(-theta) - 1)
    x = ut^delta
    y = vt^delta
    sm = x + y
    smd = sm^(delta⁻¹)
    tem = (-theta⁻¹ - 2) * log(1 + smd) + log(theta * (delta - 1) + (theta * delta + 1) * smd)
    val = tem + log(smd) + log(x) + log(y) + log(ut + 1) + log(vt + 1) - log(sm) - log(sm) - log(ut) - log(vt) - log(_u) - log(_v)

    return val
end
function cdf(copula::BB1Copula, u::AbstractVector)
    @unpack theta, delta = copula
    de1 = 1/delta
    th1 = 1/theta
    _u = u[1]
    _v = u[2]

    ut = (_u^(-theta) - 1)
    vt = (_v^(-theta) - 1)
    x = ut^delta
    y = vt^delta
    sm = x + y
    smd = sm^(de1)
    val = (1 + smd)^(-th1)
    return val
end

function get_copuladiagnostics(copula::BB1, θ::NamedTuple, dataᵤ::Matrix{F}) where {F<:Real}
    @argcheck size(dataᵤ, 1) < size(dataᵤ, 2) "Tail and Correlation statistics computer for bivariate data of size 2*n"
    @unpack theta, delta = θ
    # Compute
    λₗ = 2^(-1/(theta*delta))
    λᵤ = 2 - 2^(1/delta)
    τ_kendall = StatsBase.corkendall(dataᵤ[1,:], dataᵤ[2,:])
    ρ_spearman = StatsBase.corspearman(dataᵤ[1,:], dataᵤ[2,:])
    # Return Output
    return (;
        λₗ = λₗ,
        λᵤ = λᵤ,
        τ_kendall = τ_kendall,
        ρ_spearman = ρ_spearman
    )
end

function rand(_rng::Random.AbstractRNG, copula::BB1Copula, Nsamples::Integer)
    @unpack theta, delta = copula
    theta⁻¹ = 1/theta
    r = rand(_rng, Gamma(theta⁻¹, 1), Nsamples)

    gumbelcop = toCopula(Gumbel(), (; α = delta))
    xx = rand(_rng, gumbelcop, Nsamples)

    _u = reduce(hcat, [-log.(xx[:, iter]) ./ r[iter] for iter in eachindex(r)] )
    u = (_u .+ 1).^(-theta⁻¹)

    return u
end

####################################################################
#BB7
struct BB7 <: Archimedean end
#=
param_BB7Copula = (;
    theta = Param(truncated(Normal(2.0, 10^5), 1.0, 100.0), 2., ),
    delta = Param(truncated(Normal(2.0, 10^5), 0.0001, 100.0), 1., )
)
bb7copula = ModelWrapper(BB7(), param_BB7Copula, (;reflection = _archimedeanreflection))
length(bb7copula)
=#
struct BB7Copula{A<:Real}
    theta::A
    delta::A
    function BB7Copula(
        theta::A, delta::A
    ) where {A<:Real}
        ArgCheck.@argcheck theta >= 1.0
        ArgCheck.@argcheck delta > 0.0
        return new{A}(theta, delta)
    end
end
function toCopula(copula::BB7, θ)
    return BB7Copula(θ.theta, θ.delta)
end

function ℓlikelihood(copula::C, θ::NamedTuple, u::AbstractVector) where {C<:BB7}
    @unpack theta, delta = θ
    delta⁻¹ = 1/delta
    theta⁻¹ = 1/theta

    _u = u[1]
    _v = u[2]
    ut = 1 - (1 - _u)^theta
    vt = 1 - (1 - _v)^theta

    x = ut^(-delta) - 1
    y = vt^(-delta) - 1
    sm = x + y + 1
    smd = sm^(-delta⁻¹)

    _tem = (theta⁻¹ - 2) * log(1 - smd)
    tem = _tem + log(theta * (delta + 1) - (theta * delta + 1) * smd)
    val = tem + log(smd) + log(x + 1) + log(y + 1) + log(1 - ut) + log(1 - vt)  - log(sm) - log(sm) - log(ut) - log(vt) - log(1 - _u) - log(1 - _v)
    return val
end

function cdf(copula::BB7Copula, u::AbstractVector)
    @unpack theta, delta = copula
    de1 = 1/delta
    th1 = 1/theta
    _u = u[1]
    _v = u[2]

    ut = 1 - (1 - _u)^theta
    vt = 1 - (1 - _v)^theta
    x = ut^(-delta) - 1
    y = vt^(-delta) - 1
    sm = x + y + 1
    smd = sm^(-de1)
    val = 1 - (1 - smd)^th1
    return val
end

function get_copuladiagnostics(copula::BB7, θ::NamedTuple, dataᵤ::Matrix{F}) where {F<:Real}
    @argcheck size(dataᵤ, 1) < size(dataᵤ, 2) "Tail and Correlation statistics computer for bivariate data of size 2*n"
    @unpack theta, delta = θ
    # Compute
    λₗ = 2^(-1/delta)
    λᵤ = 2 - 2^(1/theta)
    τ_kendall = StatsBase.corkendall(dataᵤ[1,:], dataᵤ[2,:])
    ρ_spearman = StatsBase.corspearman(dataᵤ[1,:], dataᵤ[2,:])
    # Return Output
    return (;
        λₗ = λₗ,
        λᵤ = λᵤ,
        τ_kendall = τ_kendall,
        ρ_spearman = ρ_spearman
    )
end

function rsibuya(_rng::Random.AbstractRNG, theta⁻¹::R, Nsamples::Integer) where {R<:Real}
    s_exp = rand(_rng, Exponential(1), Nsamples)
    s_gamma1 = rand(_rng, Gamma(1 - theta⁻¹, 1), Nsamples)
    s_gamma2 = rand(_rng, Gamma(theta⁻¹, 1), Nsamples)
    mu = s_exp .* s_gamma1 ./ s_gamma2
    #!NOTE In rare cases some elements of mu would be larger than can be stored in biggest Int type which would cause segfault
    #!NOTE: This can happen if parameter are very far off the HPD region
    mu[mu .> typemax(Int64)] .= typemax(Int64)/1.000001
    x = 1.0 .+ rand.(_rng, Poisson.(mu))
    return x
end

function rand(_rng::Random.AbstractRNG, copula::BB7Copula, Nsamples::Integer)
    @unpack theta, delta = copula
    theta⁻¹ = 1/theta

    r = rsibuya(_rng, theta⁻¹, Nsamples)
    xx = reduce(hcat, [rand(_rng, toCopula(Clayton(), (; α = delta/r[iter])) ) for iter in eachindex(r)])

    _u = reduce(hcat, [xx[:, iter].^(1/r[iter]) for iter in eachindex(r)])
    u = 1 .- ( 1 .- _u).^(theta⁻¹)

    return u
end
