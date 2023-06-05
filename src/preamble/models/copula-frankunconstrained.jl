##################################
#Frank defined in whole Reals
struct FrankUnconstrained <: AbstractCopulas end
#=
param_frankunconstrainedCopula = (;
    α = Param(truncated(Normal(0.0, 10^5), -20.0, 20.0), -5.0, )
)
frankunconstrainedcopula = ModelWrapper(FrankUnconstrained(), param_frankunconstrainedCopula, (; reflection = nothing))
length(frankunconstrainedcopula)
=#
struct FrankUnconstrainedCopula{A<:Real}
    α::A
    function FrankUnconstrainedCopula(
        α::A
    ) where {A<:Real}
        return new{A}(α)
    end
end
function toCopula(copula::FrankUnconstrained, θ)
    return FrankUnconstrainedCopula(θ.α)
end

function ℓlikelihood(copula::C, θ::NamedTuple, u::AbstractVector) where {C<:FrankUnconstrained}
    @unpack α = θ
    t1 = 1 - exp(-α)
    tem1 = exp(-α * u[1])
    tem2 = exp(-α * u[2])
    _ℓlik = α * tem1 * tem2 * t1
    temp = t1 - (1 - tem1) * (1 - tem2)
    ℓlik = log(_ℓlik) - log(temp * temp)
    return ℓlik
end
function get_copuladiagnostics(copula::FrankUnconstrained, reflection, θ::NamedTuple, dataᵤ::Matrix{F}) where {F<:Real}
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

function qcondfrk(p::AbstractVector, u::AbstractVector, α::Real)
    cpar0 = exp(-α)
    cpar1 = 1 - cpar0
    etem = exp.(-α .* u .+ log.(1 ./ p .- 1))
    tem = 1 .- cpar1 ./ (etem .+ 1)
    v = (-log.(tem)) ./ α

    _inf = isinf.(v)
    v[_inf] .= 1.0 - 1.e-10 #(-log.(cpar0 .+ etem[_inf])) ./ α
    return v
end
function rand(_rng::Random.AbstractRNG, copula::FrankUnconstrainedCopula, Nsamples::Integer)
    @unpack α = copula
    u1 = rand(Nsamples)
    p = rand(Nsamples)
    u2 = qcondfrk(p, u1, α)
    return hcat(u1, u2)'
end

function cumℓlikelihood(id::FrankUnconstrained, reflection, θ::NamedTuple, data)
    ll = 0.0
    for dat in eachcol(data)
        ll += ℓlikelihood(id, θ, dat)
    end
    return ll
end
function _cumℓlikelihood(id::FrankUnconstrained, reflection, θ::NamedTuple, data)
    cumℓlikelihood(id, reflection, θ, data)
end

################################################################################
# Compute simulation and logpdf methods
function ModelWrappers.simulate(_rng::Random.AbstractRNG, model::ModelWrapper{<:FrankUnconstrained}, Nsamples = 1000)
    copula = toCopula(model.id, model.val)
    U =  rand(_rng, copula, Nsamples)
    return U
end
function ModelWrappers.simulate(_rng::Random.AbstractRNG, id::FrankUnconstrained, reflection, val::NamedTuple, Nsamples = 1000)
    copula = toCopula(id, val)
    U =  rand(_rng, copula, Nsamples)
    return U
end

function (objective::Objective{<:ModelWrapper{E}})(θ::NamedTuple) where {E<:FrankUnconstrained}
    @unpack model, data, tagged = objective
## Prior
    lp = log_prior(tagged.info.transform.constraint, ModelWrappers.subset(θ, tagged.parameter) )
##Likelihood
    ll = cumℓlikelihood(objective.model.id, objective.model.arg.reflection, θ, data)
    return ll + lp
end
function ModelWrappers.predict(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{M}}) where {M<:FrankUnconstrained}
    #Sample the error terms (in uniform dimension) given the Copula parameter
    u = ModelWrappers.simulate(_rng, objective.model, 1)
    return u
end

################################################################################
# Get Plots for it
function plotContour(model::ModelWrapper{<:FrankUnconstrained}, dataᵤ::D, copulaname = typeof(model.id);
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
    #obs = reduce(hcat, quantile.(marginal, dataᵤ))
    obs = reduce(hcat, quantile.(marginal, [dat for dat in eachcol(dataᵤ)]))
    plot!(view(obs, 1, :), view(obs, 2, :),
        label = copulaname,
        seriestype=:scatter, color="black", markersize=1.5
    )
    #Plot Copula density and data
    contour!(x, y, ll_contour)
end

#=
N = 10^4
Nsamples=N
copula = FrankUnconstrained()
θ = frankunconstrainedcopula.val
u = [.3, .1]
ℓlikelihood(copula, θ, u)

#check with constrained frank
ℓlikelihood(Frank(), θ, u)
copula = toCopula(FrankUnconstrained(), θ)

s = rand(_rng, toCopula(FrankUnconstrained(), (α = -10.0,)), 10^4)
scatter(s[1,:], s[2,:])
fra = Frank()
@btime ℓlikelihood($fra, $θ, $u)

cumℓlikelihood(FrankUnconstrained(), Reflection180(), θ, s)
ModelWrappers.simulate(_rng, frankunconstrainedcopula, Nsamples)
ModelWrappers.simulate(_rng, FrankUnconstrained(), "!ha", θ, Nsamples)
obj = Objective(frankunconstrainedcopula, s)
obj(θ)
ModelWrappers.predict(_rng, obj)

param_frankunconstrainedCopula = (;
α = Param(truncated(Normal(0.0, 10^5), -10.0, 10.0), -5.0, )
)
frankunconstrainedcopula = ModelWrapper(FrankUnconstrained(), param_frankunconstrainedCopula, (; reflection = nothing))

s = ModelWrappers.simulate(_rng, frankunconstrainedcopula, Nsamples)
plotContour(frankunconstrainedcopula, s)
=#
