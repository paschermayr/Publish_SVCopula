######################################
# Define rotations for copula
function rotatecopula(rotation::Rotation0, u::AbstractVector{T}) where {T<:Real}
    return u
end
function rotatecopula(rotation::Rotation90, u::AbstractVector{T}) where {T<:Real}
    u_rotated = zeros(eltype(u), size(u))
    u_rotated[1] = 1 - u[2]
    u_rotated[2] = u[1]
    return u_rotated
end
function rotatecopula(rotation::Rotation180, u::AbstractVector{T}) where {T<:Real}
    u_rotated = zeros(eltype(u), size(u))
    u_rotated[1] = 1 - u[1]
    u_rotated[2] = 1 - u[2]
    return u_rotated
end
function rotatecopula(rotation::Rotation270, u::AbstractVector{T}) where {T<:Real}
    u_rotated = zeros(eltype(u), size(u))
    u_rotated[1] = u[2]
    u_rotated[2] = 1 - u[1]
    return u_rotated
end
function unrotatecopula(rotation::Rotation0, u::AbstractVector{T}) where {T<:Real}
    return u
end
function unrotatecopula(rotation::Rotation90, u_rotated::AbstractVector{T}) where {T<:Real}
    u = zeros(eltype(u_rotated), size(u_rotated))
    u[1] = u_rotated[2]
    u[2] = 1 - u_rotated[1]
    return u
end
function unrotatecopula(rotation::Rotation180, u_rotated::AbstractVector{T}) where {T<:Real}
    u = zeros(eltype(u_rotated), size(u_rotated))
    u[1] = 1 - u_rotated[1]
    u[2] = 1 - u_rotated[2]
    return u
end
function unrotatecopula(rotation::Rotation270, u_rotated::AbstractVector{T}) where {T<:Real}
    u = zeros(eltype(u_rotated), size(u_rotated))
    u[1] = 1 - u_rotated[2]
    u[2] = u_rotated[1]
    return u
end

######################################
function rotatecopula(rotation::Rotation0, u::AbstractMatrix{T}) where {T<:Real}
    return u
end
function rotatecopula(rotation::Rotation90, u::AbstractMatrix{T}) where {T<:Real}
    u_rotated = zeros(eltype(u), size(u))
    for iter in Base.OneTo(size(u, 2))
        u_rotated[1, iter] = 1 - u[2, iter]
        u_rotated[2, iter] = u[1, iter]
    end
    return u_rotated
end
function rotatecopula(rotation::Rotation180, u::AbstractMatrix{T}) where {T<:Real}
    u_rotated = zeros(eltype(u), size(u))
    for iter in Base.OneTo(size(u, 2))
        u_rotated[1, iter] = 1 - u[1, iter]
        u_rotated[2, iter] = 1 - u[2, iter]
    end
    return u_rotated
end
function rotatecopula(rotation::Rotation270, u::AbstractMatrix{T}) where {T<:Real}
    u_rotated = zeros(eltype(u), size(u))
    for iter in Base.OneTo(size(u, 2))
        u_rotated[1, iter] = u[2, iter]
        u_rotated[2, iter] = 1 - u[1, iter]
    end
    return u_rotated
end

function unrotatecopula(rotation::Rotation0, u::AbstractMatrix{T}) where {T<:Real}
    return u
end
function unrotatecopula(rotation::Rotation90, u_rotated::AbstractMatrix{T}) where {T<:Real}
    u = zeros(eltype(u_rotated), size(u_rotated))
    for iter in Base.OneTo(size(u, 2))
        u[1, iter] = u_rotated[2, iter]
        u[2, iter] = 1 - u_rotated[1, iter]
    end
    return u
end
function unrotatecopula(rotation::Rotation180, u_rotated::AbstractMatrix{T}) where {T<:Real}
    u = zeros(eltype(u_rotated), size(u_rotated))
    for iter in Base.OneTo(size(u, 2))
        u[1, iter] = 1 - u_rotated[1, iter]
        u[2, iter] = 1 - u_rotated[2, iter]
    end
    return u
end
function unrotatecopula(rotation::Rotation270, u_rotated::AbstractMatrix{T}) where {T<:Real}
    u = zeros(eltype(u_rotated), size(u_rotated))
    for iter in Base.OneTo(size(u, 2))
        u[1, iter] = 1 - u_rotated[2, iter]
        u[2, iter] = u_rotated[1, iter]
    end
    return u
end

##################################
#Clayton
param_ClaytonCopula = (;
    ?? = Param(_alpha, truncated(Normal(0.1, 10^5), 0.0, 100.0))
)
struct Clayton <: Archimedean end
claytoncopula = ModelWrapper(Clayton(), param_ClaytonCopula, (;rotation = _archimedeanrotation))
length(param_ClaytonCopula)

function toCopula(copula::Clayton, ??)
    return ClaytonCopula(2, ??.??)
end

function ???likelihood(copula::C, ??::NamedTuple, u::AbstractVector) where {C<:Clayton}
    @unpack ?? = ??
    return log((??+1)) +
        log(prod(u))*(-??-1) +
        log((sum(u[iter]^(-??) for iter in eachindex(u) ) - 1))*(-(2*?? + 1)/??)
end
function rand_conditional(_rng::Random.AbstractRNG, copula::C, u1::T) where {C<:ClaytonCopula, T<:Real}
    t = rand(_rng)
    ?? = copula.??
    u2 = ( ( t/u1^(-??-1) )^(-??/(1+??)) - u1^(-??) + 1 )^(-1/??)
    return u2
end
function get_copuladiagnostics(copula::Clayton, ??::NamedTuple, data???::Matrix{F}) where {F<:Real}
    @argcheck size(data???, 1) < size(data???, 2) "Tail and Correlation statistics computer for bivariate data of size 2*n"
    @unpack ?? = ??
    # Compute
    ????? = 2^(-1/??)
    ????? = 0.0
    ??_kendall = StatsBase.corkendall(data???[1,:], data???[2,:])
    ??_spearman = StatsBase.corspearman(data???[1,:], data???[2,:])
    # Return Output
    return (;
        ????? = ?????,
        ????? = ?????,
        ??_kendall = ??_kendall,
        ??_spearman = ??_spearman
    )
end

##################################
#Frank
param_FrankCopula = (;
    ?? = Param(_alpha, truncated(Normal(0.1, 10^5), 0.0, 30.0))
)
struct Frank <: Archimedean end
frankcopula = ModelWrapper(Frank(), param_FrankCopula, (;rotation = _archimedeanrotation))
length(param_FrankCopula)

function toCopula(copula::Frank, ??)
    return FrankCopula(2, ??.??)
end
function ???likelihood(copula::C, ??::NamedTuple, u::AbstractVector) where {C<:Frank}
    @unpack ?? = ??
    return ( log(??) + (-?? * (sum(u))) + log(1 - exp(-??)) ) -
    2*( log( (1 - exp(-??)) - prod( (1 - exp(-?? * u[iter])) for iter in eachindex(u)) ) )
end
function rand_conditional(_rng::Random.AbstractRNG, copula::C, u1::T) where {C<:FrankCopula, T<:Real}
    t = rand(_rng)
    ?? = copula.??
    u2 = (1/-??) * log(1 + (t*(exp(-??) - 1) ) / ( 1 + (exp(-??*u1) - 1)*(1-t) ) )
    return u2
end
function get_copuladiagnostics(copula::Frank, ??::NamedTuple, data???::Matrix{F}) where {F<:Real}
    @argcheck size(data???, 1) < size(data???, 2) "Tail and Correlation statistics computer for bivariate data of size 2*n"
    # Compute
    ????? = 0.0
    ????? = 0.0
    ??_kendall = StatsBase.corkendall(data???[1,:], data???[2,:])
    ??_spearman = StatsBase.corspearman(data???[1,:], data???[2,:])
    # Return Output
    return (;
        ????? = ?????,
        ????? = ?????,
        ??_kendall = ??_kendall,
        ??_spearman = ??_spearman
    )
end

##################################
#Gumbel
param_GumbelCopula = (;
    ?? = Param(_alpha, truncated(Normal(2.0, 10^5), 1.0, 100.0))
)
struct Gumbel <: Archimedean end
gumbelcopula = ModelWrapper(Gumbel(), param_GumbelCopula, (;rotation = _archimedeanrotation))
length(param_GumbelCopula)

function toCopula(copula::Gumbel, ??)
    return GumbelCopula(2, ??.??)
end

function ???likelihood(copula::C, ??::NamedTuple, u::AbstractVector) where {C<:Gumbel}
    @unpack ?? = ??
    return ( - ( sum( (-log(u[iter]))^?? for iter in eachindex(u) ) )^(1/??) - log( prod(u) ) ) +
    log(     ( sum( (-log(u[iter]))^?? for iter in eachindex(u) )^( 2/?? - 2 ) / (prod( log(u[iter]) for iter in eachindex(u) )^(1-??)) ) ) +
    log(     ( 1 + (?? - 1) * ( sum( (-log(u[iter]))^?? for iter in eachindex(u) ) )^(-1/??) ) )
end
function get_copuladiagnostics(copula::Gumbel, ??::NamedTuple, data???::Matrix{F}) where {F<:Real}
    @argcheck size(data???, 1) < size(data???, 2) "Tail and Correlation statistics computer for bivariate data of size 2*n"
    @unpack ?? = ??
    # Compute
    ????? = 0.0
    ????? = 2 - 2^(1/??)
    ??_kendall = StatsBase.corkendall(data???[1,:], data???[2,:])
    ??_spearman = StatsBase.corspearman(data???[1,:], data???[2,:])
    # Return Output
    return (;
        ????? = ?????,
        ????? = ?????,
        ??_kendall = ??_kendall,
        ??_spearman = ??_spearman
    )
end

##################################
#Joe
param_JoeCopula = (;
    ?? = Param(_alpha, truncated(Normal(2.0, 10^5), 1.0, 100.0))
)
struct Joe <: Archimedean end
joecopula = ModelWrapper(Joe(), param_JoeCopula, (;rotation = _archimedeanrotation))
length(param_JoeCopula)

function toCopula(copula::Joe, ??)
    return JoeCopula(2, ??.??)
end

function _joeldensity(??::S, u::T) where {S<:Real, T<:Real}
    return (1 - u)^??
end
function ???likelihood(copula::C, ??::NamedTuple, u::AbstractVector) where {C<:Joe}
    @unpack ?? = ??
    return (1/?? - 2) * log( sum( _joeldensity(??, u[iter]) for iter in eachindex(u) ) - prod( _joeldensity(??, u[iter]) for iter in eachindex(u) ) ) +
        log(?? - 1 + sum( _joeldensity(??, u[iter]) for iter in eachindex(u) ) - prod( _joeldensity(??, u[iter]) for iter in eachindex(u) ) ) +
        sum( (??-1)*log( (1 - u[iter]) ) for iter in eachindex(u) )
end
function get_copuladiagnostics(copula::Joe, ??::NamedTuple, data???::Matrix{F}) where {F<:Real}
    @argcheck size(data???, 1) < size(data???, 2) "Tail and Correlation statistics computer for bivariate data of size 2*n"
    @unpack ?? = ??
    # Compute
    ????? = 0.0
    ????? = 2 - 2^(1/??)
    ??_kendall = StatsBase.corkendall(data???[1,:], data???[2,:])
    ??_spearman = StatsBase.corspearman(data???[1,:], data???[2,:])
    # Return Output
    return (;
        ????? = ?????,
        ????? = ?????,
        ??_kendall = ??_kendall,
        ??_spearman = ??_spearman
    )
end

################################################################################
function cum???likelihood(id::A, rotation::R, ??::NamedTuple, data) where {A<:Archimedean, R<:ArchimedeanRotation}
## If required, unrotate copula to original angle
    U_rotated = unrotatecopula(rotation, data)
## If required, rotate copula to specified angle
## Compute ll
    ll = 0.0
    for dat in eachcol(U_rotated)
        ll += ???likelihood(id, ??, dat)
    end
    return ll
end

# Compute simulation and logpdf methods
function ModelWrappers.simulate(_rng::Random.AbstractRNG, model::ModelWrapper{<:Archimedean}, Nsamples = 1000)
    copula = toCopula(model.id, model.val)
    U =  rand(_rng, copula, Nsamples)
#    return U
    U_rotated = rotatecopula(model.arg.rotation, U)
    return U_rotated
end
function ModelWrappers.simulate(_rng::Random.AbstractRNG, id::E, rotation::R, val::NamedTuple, Nsamples = 1000) where {E<:Archimedean, R<:ArchimedeanRotation}
    copula = toCopula(id, val)
    U =  rand(_rng, copula, Nsamples)
#    return U
    U_rotated = rotatecopula(rotation, U)
    return U_rotated
end

function (objective::Objective{<:ModelWrapper{A}})(??::NamedTuple) where {A<:Archimedean}
    @unpack model, data, tagged = objective
## Prior
    lp = log_prior(tagged.info.constraint, ModelWrappers.subset(??, tagged.parameter) )
## likelihood
    ll = cum???likelihood(objective.model.id, objective.model.arg.rotation, ??, data)
    return ll + lp
end

function ModelWrappers.predict(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{M}}) where {M<:Archimedean}
    #Sample the error terms (in uniform dimension) given the Copula parameter
    u = ModelWrappers.simulate(_rng, objective.model, 1)
#    return u
    U_rotated = rotatecopula(objective.model.arg.rotation, u)
    return U_rotated
end

function plotContour(model::ModelWrapper{<:Archimedean}, data???::D, copulaname = model.id;
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
    ???du = Distributions.logpdf.(marginal, x)
    ???dv = Distributions.logpdf.(marginal, y)
    #Set likelihood function via model
    ll_contour = zeros(Float64, length(ux), length(vx) )
    u_vec = zeros(2)
    copula = toCopula(model.id, model.val)

    for rows in eachindex(ux)
        for cols in eachindex(vx)
            u_vec[1] = ux[rows]
            u_vec[2] = vx[cols]
            u_vec = rotatecopula(model.arg.rotation, u_vec)
#            u_vec = unrotatecopula(model.arg.rotation, u_vec)
            ll_contour[rows, cols] = exp( ???du[rows] + ???dv[cols] + ???likelihood(model.id, model.val, u_vec) ) # logpdf(copula, u_vec) ) #???likelihood(model.id, u_vec, model.val.??) ) #du[rows] * dv[cols] * exp( ???likelihood(model.id, u_vec, model.val.??) )#0.03733348434990828
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
    obs = reduce(hcat, quantile.(marginal, [dat for dat in eachcol(data???)]))
#    obs = reduce(hcat, quantile.(marginal, [rotatecopula(model.arg.rotation, dat) for dat in eachcol(data???)]))
    plot!(view(obs, 1, :), view(obs, 2, :),
        label = copulaname,
        seriestype=:scatter,
        #legend=false,
        color="black", markersize=1.5
    )
    #Plot Copula density and data
    contour!(x, y, ll_contour)
end
