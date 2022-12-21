################################################################################
#Assign individual Copulas

##################################
#Gaussian
param_GaussianCopula = (;
    ρ = Param(_rho, truncated(Normal(0.0, 10^5), -1.0, 1.0))
)
struct Gaussian <: Elliptical end
gaussiancopula = ModelWrapper(Gaussian(), param_GaussianCopula, (; rotation = nothing))
length(param_GaussianCopula)

function toCopula(copula::Gaussian, θ)
    return GaussianCopula( ρToΣ(θ.ρ) )
end
function ℓlikelihood(copula::C, θ::NamedTuple, u::AbstractVector) where {C<:Gaussian}
    @unpack ρ = θ
    x = [ quantile(Normal(), u[iter]) for iter in eachindex(u) ]
    return - log(sqrt(1-ρ^2)) - (ρ^2*sum(x[iter]^2 for iter in eachindex(x)) - 2*ρ*prod(x)) / (2*(1-ρ^2))
end
function ℓlikelihood_conditional(copula::C, θ::NamedTuple, ϵ::S, ξ::T) where {C<:Gaussian, S<:Real, T<:Real}
    #!NOTE: cdf of v given u
    @unpack ρ = θ
    return logpdf(Normal(), (ϵ - ρ*ξ)/sqrt(1 - ρ^2)) + 0 - logpdf(Normal(), ϵ) + log(sqrt(1 - ρ^2) )
end

function rand_conditional(_rng::Random.AbstractRNG, copula::C, u1::T) where {C<:GaussianCopula, T<:Real}
    #"Sample u2 given fixed u1 ~ Conditional Sampling method"
    #Steps:
        # Derive C(u,v), take derivative wrt v, and solve for inverse (v = ...)
        # Sample P(U2=u2 | U1 = u1) = t ~ Uniform, then plug into v =
    t = rand(_rng)
    ρ = copula.Σ[2]
    u2 = cdf(Normal(), (quantile(Normal(), t) * sqrt(1-ρ^2) + ρ*quantile(Normal(), u1)) )
    return u2
end
function get_copuladiagnostics(copula::Gaussian, θ::NamedTuple, dataᵤ::Matrix{F}) where {F<:Real}
    @argcheck size(dataᵤ, 1) < size(dataᵤ, 2) "Tail and Correlation statistics computer for bivariate data of size 2*n"
    # Compute
    λₗ = 0.0
    λᵤ = 0.0
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
#T Copula:

import Distributions: quantile, cdf, logpdf

struct T2Distribution <: Distributions.DiscreteUnivariateDistribution end
function quantile(d::T2Distribution, u::R) where {R<:Real}
    return (2*u - 1) / ( sqrt(2*u*(1-u)) )
end
function cdf(d::T2Distribution, x::R) where {R<:Real}
    return 0.5 + x / ( 2 * sqrt(2) * sqrt(1 + x^2/2) )
end
function logpdf(d::T2Distribution, x::R) where {R<:Real}
    return - log(2) - log(sqrt(2)) - log( ( 1 + x^2/2 )^(3/2) )
end

struct T4Distribution <: Distributions.DiscreteUnivariateDistribution end
function quantile(d::T4Distribution, u::R) where {R<:Real}
    α = 4 * u * (1 - u)
    p = (4 / sqrt(α)) * cos(1/3 * acos( sqrt(α) ) )
    x = sign(u - 0.5) * sqrt(p - 4)
    return x
end
function cdf(d::T4Distribution, x::R) where {R<:Real}
    return 0.5 + 3/8 * ( x / sqrt(1 + x^2/4)  ) * (  1 - 1/12 * (x^2 / (1 + x^2/4) ) )
end
function logpdf(d::T4Distribution, x::R) where {R<:Real}
    return log(3) - log(8) - log( ( 1 + x^2/4 )^(5/2) )
end

struct MyTDistribution{T<:Real} <: Distributions.DiscreteUnivariateDistribution
    ν   ::  T
end
param_TCopula = (;
    ρ = Param(_rho, truncated(Normal(0.0, 10^5), -1.0, 1.0)),
    #!NOTE: Fix df to be either 1 or 2 so can use Cauchy or explicit formula for quantile computation
    df = Param(2, Fixed())
)

struct TCop <: Elliptical end
tcopula = ModelWrapper(TCop(), param_TCopula, (; rotation = nothing))
length(param_TCopula)

function toCopula(copula::TCop, θ)
    return TCopula( θ.df, ρToΣ(θ.ρ) )
end
function _Tdensity(ρ::R, df::S, x::AbstractArray{T}) where {R<:Real, S<:Real, T<:Real}
    return df*(1-ρ^2) + sum(x[iter]^2 for iter in eachindex(x)) - 2*ρ*prod(x)
end
function ℓlikelihood(copula::C, θ::NamedTuple, u::AbstractVector) where {C<:TCop}
    @unpack ρ, df = θ
    x = [ quantile(T2Distribution(), u[iter]) for iter in eachindex(u)]
    return -log(2) + (df+1)/2 * log(1 - ρ^2) - 2*log( SpecialFunctions.gamma((df+1)/2) ) + 2*log( SpecialFunctions.gamma((df)/2) ) -
     (df-2)/2*log(df) + (df+1)/2 * sum(log(df + x[iter]^2) for iter in eachindex(x)) - (df+2)/2 * log( _Tdensity(ρ, df, x) )
end
function get_copuladiagnostics(copula::TCop, θ::NamedTuple, dataᵤ::Matrix{F}) where {F<:Real}
    @argcheck size(dataᵤ, 1) < size(dataᵤ, 2) "Tail and Correlation statistics computer for bivariate data of size 2*n"
    @unpack df, ρ = θ
    # Compute
    λₗ = 2 * pdf(TDist(df+1), (-sqrt(df+1) * sqrt( (1-ρ)/(1+ρ) ) ) )
    λᵤ = 2 * pdf(TDist(df+1), (-sqrt(df+1) * sqrt( (1-ρ)/(1+ρ) ) ) )
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


function cumℓlikelihood(id::E, rotation, θ::NamedTuple, data) where {E<:Elliptical}
    ll = 0.0
    for dat in eachcol(data)
        ll += ℓlikelihood(id, θ, dat)
    end
    return ll
end

################################################################################
# Compute simulation and logpdf methods
function ModelWrappers.simulate(_rng::Random.AbstractRNG, model::ModelWrapper{<:Elliptical}, Nsamples = 1000)
    copula = toCopula(model.id, model.val)
    U =  rand(_rng, copula, Nsamples)
    return U
end
function ModelWrappers.simulate(_rng::Random.AbstractRNG, id::E, rotation, val::NamedTuple, Nsamples = 1000) where {E<:Elliptical}
    copula = toCopula(id, val)
    U =  rand(_rng, copula, Nsamples)
    return U
end

function (objective::Objective{<:ModelWrapper{E}})(θ::NamedTuple) where {E<:Elliptical}
    @unpack model, data, tagged = objective
## Prior
    lp = log_prior(tagged.info.constraint, ModelWrappers.subset(θ, tagged.parameter) )
##Likelihood
    ll = cumℓlikelihood(objective.model.id, objective.model.arg.rotation, θ, data)
    return ll + lp
end
function ModelWrappers.predict(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{M}}) where {M<:Elliptical}
    #Sample the error terms (in uniform dimension) given the Copula parameter
    u = ModelWrappers.simulate(_rng, objective.model, 1)
    return u
end

################################################################################
# Get Plots for it
function plotContour(model::ModelWrapper{<:Elliptical}, dataᵤ::D, copulaname = model.id;
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
            ll_contour[rows, cols] = exp( ℓdu[rows] + ℓdv[cols] + ℓlikelihood(model.id, model.val, u_vec)  ) # logpdf(copula, u_vec) ) #ℓlikelihood(model.id, u_vec, model.val.α) ) #du[rows] * dv[cols] * exp( ℓlikelihood(model.id, u_vec, model.val.α) )#0.03733348434990828
        end
    end
    # Create Plot
    plot_copula = plot(layout=(1,1), size = plot_default_size,# legend=false,
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
        seriestype=:scatter, color="black", markersize=1.5
    )
    #Plot Copula density and data
    contour!(x, y, ll_contour)
end
