#=
NOTE: We cannot SAMPLE from factor copulas with different rotation structures, but can evaluate them correctly, i.e. Real data works as intended, simulated has to have same rotation.
=#
##################################
#Base model
param_factor = (;
    #latent Factor
    #!NOTE - has to be in same dimension as data ~ all in UNIFORM space! -> Careful if defined in real dimension in other models
    latent = Param( rand(n), [truncated(Normal(0.5, 1.0), 0.0, 1.0) for _ in Base.OneTo(n)]),
    #Normal
    factors = (;
        #Normal
        factor1 = (; ρ = Param(_rho, truncated(Normal(0.0, 10^5), -1.0, 1.0)) ),
        #Frank
        factor2 = (; α = Param(_alpha, truncated(Normal(0.1, 10^5), 0.0, 100.0)) ),
        #Clayton
        factor3 = (; α = Param(_alpha, truncated(Normal(0.1, 10^5), 0.0, 100.0))
        )
    )
)
factorcopula = ModelWrapper(FactorCopula(), param_factor,
    (;  copulanames = (Gaussian(), Frank(), Clayton()),
        rotation = (nothing, Rotation0(), Rotation0())
    )
)
length(factorcopula)

################################################################################
# Compute simulation and logpdf methods
function ModelWrappers.simulate(_rng::Random.AbstractRNG, model::ModelWrapper{<:FactorCopula}, Nsamples = 1000)
    #Preallocate data
    data = zeros(Float64, length(model.arg.copulanames), Nsamples)
    #First simulate latent u
    latent = rand(Nsamples)
    #For each  factor, sample u | latent
    for factor in Base.OneTo(size(data, 1))
        copula_conditional = toCopula(model.arg.copulanames[factor], model.val.factors[factor])
        for t in Base.OneTo(Nsamples)
            data[factor, t] = rand_conditional(_rng, copula_conditional, latent[t])
        end
    end
    return latent, data
end

function ModelWrappers.simulate(_rng::Random.AbstractRNG, id::E, rotation, val::NamedTuple, arg, Nsamples = 1000) where {E<:FactorCopula}
    #Preallocate data
    data = zeros(Float64, length(arg), Nsamples)
    #First simulate latent u
    latent = rand(Nsamples)
    #For each  factor, sample u | latent
    for factor in Base.OneTo(size(data, 1))
        copula_conditional = toCopula(arg[factor], val.factors[factor])
        for t in Base.OneTo(Nsamples)
            data[factor, t] = rand_conditional(_rng, copula_conditional, latent[t])
        end
    end
    return latent, data
end

function cumℓlikelihood(copulanames::T, rotation, θ::NamedTuple, data) where {T<:Tuple}
    ##Likelihood
    ll = 0.0
    #Assign buffer variable and assign latent state trajectory as u1
    _u = zeros(eltype(θ.latent), 2, size(data, 2))
    for factor in Base.OneTo( size(data, 1) )
        #Set latent factor as common first data dimension
        _u[1,:] .= θ.latent
        #assign current data as u2
        _u[2,:] .= data[factor, :]
        ## If required, unrotate copula to original angle
        if copulanames[factor] isa Archimedean
            _u[1,:] .= θ.latent
            _u = unrotatecopula(rotation[factor], _u)
        end
        #Add current copula to likelihood
        ll += cumℓlikelihood(copulanames[factor], rotation[factor], θ.factors[factor], _u)
    end
    return ll
end

function (objective::Objective{<:ModelWrapper{F}})(θ::NamedTuple) where {F<:FactorCopula}
    @unpack model, data, tagged = objective
## Prior
    lp = log_prior(tagged.info.constraint, ModelWrappers.subset(θ, tagged.parameter) )
##Likelihood
    ll = cumℓlikelihood(model.arg.copulanames, model.arg.rotation, θ, data)
    return ll + lp
end
function ModelWrappers.predict(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{M}}) where {M<:FactorCopula}
    #Sample the error terms (in uniform dimension) given the Copula parameter
    latentᵤ, u = ModelWrappers.simulate(_rng, objective.model, 1)
    return latentᵤ, u
end

################################################################################
# Get Plots for it
function plotContour(model::ModelWrapper{<:FactorCopula}, dataᵤ::D, factor::Integer;
    #!NOTE: assumes that model is already fitted with posterior mean
    #!NOTE: assumes data is from uniform domain
    marginal = Distributions.Normal(0.0, 1.0),
    x = -3:.01:3, y = -3:.01:3,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize,
    lims = (-4., 4.)
    ) where {D}
    #Relocate observations
    ux = Distributions.cdf.(marginal, x)
    vx = Distributions.cdf.(marginal, y)
    ℓdu = Distributions.logpdf.(marginal, x)
    ℓdv = Distributions.logpdf.(marginal, y)
    #Set likelihood function via model
    ll_contour = zeros(Float64, length(ux), length(vx) )
    u_vec = zeros(2)
    for rows in eachindex(ux)
        for cols in eachindex(vx)
            u_vec[1] = ux[rows]
            u_vec[2] = vx[cols]
            if model.arg.copulanames[factor] isa Archimedean
                u_vec = rotatecopula(model.arg.rotation[factor], u_vec)
            end
            ll_contour[rows, cols] = exp( ℓdu[rows] + ℓdv[cols] + ℓlikelihood(model.arg.copulanames[factor], model.val.factors[factor], u_vec) ) # logpdf(copula, u_vec) ) #ℓlikelihood(model.id, u_vec, model.val.α) ) #du[rows] * dv[cols] * exp( ℓlikelihood(model.id, u_vec, model.val.α) )#0.03733348434990828
        end
    end
    # Create Plot
    plot_copula = plot(layout=(1,1), size = plot_default_size, legend=false,
        xlims = lims, ylims = lims,
        xguidefontsize=fontsize, yguidefontsize=fontsize, legendfontsize=fontsize,
        xtickfontsize=axissize, ytickfontsize=axissize
    )
    # Add sample data on the real line
    data_merged = hcat(model.val.latent, dataᵤ[factor,:])'
    obs = reduce(hcat, quantile.(marginal, [dat for dat in eachcol(data_merged)]))
    plot!(view(obs, 1, :), view(obs, 2, :), seriestype=:scatter, legend=false, color="orange", markersize=1.5)
    #Plot Copula density and data
    contour!(x, y, ll_contour)
end
