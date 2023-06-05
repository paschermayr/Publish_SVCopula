##################################
#Clayton
struct Clayton <: Archimedean end
#=
param_ClaytonCopula = (;
    α = Param(truncated(Normal(0.1, 10^5), 0.0, 100.0), _alpha, )
)
claytoncopula = ModelWrapper(Clayton(), param_ClaytonCopula, (;reflection = _archimedeanreflection))
length(param_ClaytonCopula)
=#
function toCopula(copula::Clayton, θ)
    return ClaytonCopula(2, θ.α)
end

function ℓlikelihood(copula::C, θ::NamedTuple, u::AbstractVector) where {C<:Clayton}
    @unpack α = θ
    return log((α+1)) +
        log(prod(u))*(-α-1) +
        log((sum(u[iter]^(-α) for iter in eachindex(u) ) - 1))*(-(2*α + 1)/α)
end
function rand_conditional(_rng::Random.AbstractRNG, copula::C, u1::T) where {C<:ClaytonCopula, T<:Real}
    t = rand(_rng)
    α = copula.θ
    u2 = ( ( t/u1^(-α-1) )^(-α/(1+α)) - u1^(-α) + 1 )^(-1/α)
    return u2
end

function get_tails(copula::Clayton, θ::NamedTuple)
    @unpack α = θ
    λₗ = 2^(-1/α)
    λᵤ = 0.0
    return λₗ, λᵤ
end
function get_copuladiagnostics(copula::Clayton, reflection::ArchimedeanReflection, θ::NamedTuple, dataᵤ::Matrix{F}) where {F<:Real}
    @argcheck size(dataᵤ, 1) < size(dataᵤ, 2) "Tail and Correlation statistics computer for bivariate data of size 2*n"
    #Compute tails - reflection takes into account lower and upper tails
    λₗ, λᵤ = get_tails(copula, reflection, θ)
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

##################################
#Frank
struct Frank <: Archimedean end
#=
param_FrankCopula = (;
    α = Param(truncated(Normal(0.1, 10^5), -30.0, 30.0), _alpha, )
)
frankcopula = ModelWrapper(Frank(), param_FrankCopula, (;reflection = _archimedeanreflection))
length(param_FrankCopula)
=#
function toCopula(copula::Frank, θ)
    return FrankCopula(2, θ.α)
end

function ℓlikelihood(copula::C, θ::NamedTuple, u::AbstractVector) where {C<:Frank}
    @unpack α = θ
    return ( log(α) + (-α * (sum(u))) + log(1 - exp(-α)) ) -
    2*( log( (1 - exp(-α)) - prod( (1 - exp(-α * u[iter])) for iter in eachindex(u)) ) )
end

function rand_conditional(_rng::Random.AbstractRNG, copula::C, u1::T) where {C<:FrankCopula, T<:Real}
    t = rand(_rng)
    α = copula.θ
    u2 = (1/-α) * log(1 + (t*(exp(-α) - 1) ) / ( 1 + (exp(-α*u1) - 1)*(1-t) ) )
    return u2
end

function get_tails(copula::Frank, θ::NamedTuple)
    return 0.0, 0.0
end
function get_copuladiagnostics(copula::Frank, reflection::ArchimedeanReflection, θ::NamedTuple, dataᵤ::Matrix{F}) where {F<:Real}
    @argcheck size(dataᵤ, 1) < size(dataᵤ, 2) "Tail and Correlation statistics computer for bivariate data of size 2*n"
    # Compute tails
    λₗ, λᵤ = get_tails(copula, reflection, θ)
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

##################################
#Gumbel
struct Gumbel <: Archimedean end
#=
param_GumbelCopula = (;
    α = Param(truncated(Normal(2.0, 10^5), 1.0, 100.0), _alpha, )
)
gumbelcopula = ModelWrapper(Gumbel(), param_GumbelCopula, (;reflection = _archimedeanreflection))
length(param_GumbelCopula)
=#
function toCopula(copula::Gumbel, θ)
    return GumbelCopula(2, θ.α)
end

function ℓlikelihood(copula::C, θ::NamedTuple, u::AbstractVector) where {C<:Gumbel}
    @unpack α = θ
    return ( - ( sum( (-log(u[iter]))^α for iter in eachindex(u) ) )^(1/α) - log( prod(u) ) ) +
    log(     ( sum( (-log(u[iter]))^α for iter in eachindex(u) )^( 2/α - 2 ) / (prod( log(u[iter]) for iter in eachindex(u) )^(1-α)) ) ) +
    log(     ( 1 + (α - 1) * ( sum( (-log(u[iter]))^α for iter in eachindex(u) ) )^(-1/α) ) )
end

function get_tails(copula::Gumbel, θ::NamedTuple)
    @unpack α = θ
    # Compute
    λₗ = 0.0
    λᵤ = 2 - 2^(1/α)
    return λₗ, λᵤ
end
function get_copuladiagnostics(copula::Gumbel, reflection::ArchimedeanReflection, θ::NamedTuple, dataᵤ::Matrix{F}) where {F<:Real}
    @argcheck size(dataᵤ, 1) < size(dataᵤ, 2) "Tail and Correlation statistics computer for bivariate data of size 2*n"
    # Compute tails
    λₗ, λᵤ = get_tails(copula, reflection, θ)

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

##################################
#Joe
struct Joe <: Archimedean end
#=
param_JoeCopula = (;
    α = Param(truncated(Normal(2.0, 10^5), 1.0, 100.0), _alpha, )
)
joecopula = ModelWrapper(Joe(), param_JoeCopula, (;reflection = _archimedeanreflection))
length(param_JoeCopula)
=#
function toCopula(copula::Joe, θ)
    return JoeCopula(2, θ.α)
end

function _joeldensity(θ::S, u::T) where {S<:Real, T<:Real}
    return (1 - u)^θ
end
function ℓlikelihood(copula::C, θ::NamedTuple, u::AbstractVector) where {C<:Joe}
    @unpack α = θ
    return (1/α - 2) * log( sum( _joeldensity(α, u[iter]) for iter in eachindex(u) ) - prod( _joeldensity(α, u[iter]) for iter in eachindex(u) ) ) +
        log(α - 1 + sum( _joeldensity(α, u[iter]) for iter in eachindex(u) ) - prod( _joeldensity(α, u[iter]) for iter in eachindex(u) ) ) +
        sum( (α-1)*log( (1 - u[iter]) ) for iter in eachindex(u) )
end

function get_tails(copula::Joe, θ::NamedTuple)
    @unpack α = θ
    # Compute
    λₗ = 0.0
    λᵤ = 2 - 2^(1/α)
    return λₗ, λᵤ
end
function get_copuladiagnostics(copula::Joe, reflection::ArchimedeanReflection, θ::NamedTuple, dataᵤ::Matrix{F}) where {F<:Real}
    @argcheck size(dataᵤ, 1) < size(dataᵤ, 2) "Tail and Correlation statistics computer for bivariate data of size 2*n"
    # Compute tails
    λₗ, λᵤ = get_tails(copula, reflection, θ)

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

################################################################################
function cumℓlikelihood(id::A, reflection::R, θ::NamedTuple, data) where {A<:Archimedean, R<:ArchimedeanReflection}
## If required, rotate data for copula
    U_rotated = rotatecopula(reflection, data)
## Compute ll
    ll = 0.0
    for dat in eachcol(U_rotated)
        ll += ℓlikelihood(id, θ, dat)
    end
    return ll
end
#!NOTE: not used in inference process
function _cumℓlikelihood(id::A, reflection::R, θ::NamedTuple, data) where {A<:Archimedean, R<:ArchimedeanReflection}
## If required, rotate data for copula
    U_rotated = rotatecopula(reflection, data)
## Compute ll
    ll = 0.0
    for dat in eachcol(U_rotated)
        _val = ℓlikelihood(id, θ, dat)
        ll += isfinite(_val) ? _val : 0.0
    end
    return ll
end

# Compute simulation and logpdf methods
function ModelWrappers.simulate(_rng::Random.AbstractRNG, model::ModelWrapper{<:Archimedean}, Nsamples = 1000)
    copula = toCopula(model.id, model.val)
    U =  rand(_rng, copula, Nsamples)
    U_rotated = rotatecopula(model.arg.reflection, U)
    return U_rotated
end
function ModelWrappers.simulate(_rng::Random.AbstractRNG, id::E, reflection::R, val::NamedTuple, Nsamples = 1000) where {E<:Archimedean, R<:ArchimedeanReflection}
    copula = toCopula(id, val)
    U =  rand(_rng, copula, Nsamples)
    U_rotated = rotatecopula(reflection, U)
    return U_rotated
end

function (objective::Objective{<:ModelWrapper{A}})(θ::NamedTuple) where {A<:Archimedean}
    @unpack model, data, tagged = objective
## Prior
    lp = log_prior(tagged.info.transform.constraint, ModelWrappers.subset(θ, tagged.parameter) )
## likelihood
    ll = cumℓlikelihood(objective.model.id, objective.model.arg.reflection, θ, data)
    return ll + lp
end

function ModelWrappers.predict(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{M}}) where {M<:Archimedean}
    #Sample the error terms (in uniform dimension) given the Copula parameter
    #!NOTE: This includes reflection already
    U_rotated = ModelWrappers.simulate(_rng, objective.model, 1)
    return U_rotated
end

#=
model
copula
dat = simulate(_rng, copula, 10000)
#scatter(dat[1,:], dat[2,:])
p1 = plotContour(copula, dat, )
=#
################################################################################
# Get Plots for it
function plotContour(model::ModelWrapper{<:Archimedean}, dataᵤ::D, copulaname = typeof(model.id);
    #!NOTE: assumes that model is already fitted with posterior mean
    #!NOTE: assumes data is from uniform domain
    marginal = Distributions.Normal(0.0, 1.0),
    x = -3:.01:3, y = -3:.01:3,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize,
    lims = (-6., 6.)
    ) where {D}
    #Relocate observations
    ux = Distributions.cdf.(marginal, x)
    vx = Distributions.cdf.(marginal, y)
    ℓdu = Distributions.logpdf.(marginal, x)
    ℓdv = Distributions.logpdf.(marginal, y)
    #Set likelihood function via model
    ll_contour = zeros(Float64, length(ux), length(vx) )
    u_vec = zeros(2)
    copula = toCopula(model.id, model.val)

    for rows in eachindex(ux)
        for cols in eachindex(vx)
            u_vec[1] = ux[rows]
            u_vec[2] = vx[cols]
            u_vec = rotatecopula(model.arg.reflection, u_vec)
            ll_contour[rows, cols] = exp( ℓdu[rows] + ℓdv[cols] + ℓlikelihood(model.id, model.val, u_vec) ) # logpdf(copula, u_vec) ) #ℓlikelihood(model.id, u_vec, model.val.α) ) #du[rows] * dv[cols] * exp( ℓlikelihood(model.id, u_vec, model.val.α) )#0.03733348434990828
        end
    end
    # Create Plot
    plot_copula = plot(layout=(1,1), size = plot_default_size, #legend=false,
        xlims = lims, ylims = lims,
        foreground_color_legend = nothing,
        background_color_legend = nothing,
        xguidefontsize=fontsize, yguidefontsize=fontsize, legendfontsize=fontsize,
        xtickfontsize=axissize, ytickfontsize=axissize
    )
    # Add sample data on the real line
    obs = reduce(hcat, quantile.(marginal, [dat for dat in eachcol(dataᵤ)]))
    plot!(view(obs, 1, :), view(obs, 2, :),
        label = copulaname,
        seriestype=:scatter,
        #legend=false,
        color="black", markersize=1.5
    )
    #Plot Copula density and data
    contour!(x, y, ll_contour)
end
