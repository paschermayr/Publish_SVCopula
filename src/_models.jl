################################################################################
# Define copula structure

####################################### Standard Parametric Copulas
if _modelname isa Gaussian
    #Normal Copula Names
    copulanames = Gaussian()
    subcopulas = Gaussian()
    # Parameter
    _copula = (;
        ρ = Param(_rho, truncated(Normal(0.0, 10^5), -1.0, 1.0))
    )
    # Model
    copula = ModelWrapper(
        copulanames, deepcopy(_copula), (; rotation = _archimedeanrotation)
    )

elseif _modelname isa TCop
    #T Copula Names
    copulanames = TCop()
    subcopulas = TCop()
    # Parameter
    _copula = (;
        ρ = Param(_rho, truncated(Normal(0.0, 10^5), -1.0, 1.0)),
        #!NOTE: set df = 2 so have formula for quantile, else need to change copula-elliptical.jl code
        df = Param(2.0, truncated( Normal(2.0, 10.0^3), 0.1, 100.))
    )
    # Model
    copula = ModelWrapper(
        copulanames, deepcopy(_copula), (; rotation = _archimedeanrotation)
    )

elseif _modelname isa Clayton
    #Clayton Copula Names
    copulanames = Clayton()
    subcopulas = Clayton()
    # Parameter
    _copula = (;
        α = Param(_alpha, truncated(Normal(1.0, 10^5), 0.0, 50.0))
    )
    # Model
    copula = ModelWrapper(
        copulanames, deepcopy(_copula), (; rotation = _archimedeanrotation)
    )

elseif _modelname isa Frank
    #Frank Copula Names
    copulanames = Frank()
    subcopulas = Frank()
    # Parameter
    _copula = (;
        α = Param(_alpha, truncated(Normal(1.0, 10^5), 0.0, 30.0))
    )
    # Model
    copula = ModelWrapper(
        copulanames, deepcopy(_copula), (; rotation = _archimedeanrotation)
    )

elseif _modelname isa Gumbel
    #Gumbel Copula Names
    copulanames = Gumbel()
    subcopulas = Gumbel()
    # Parameter
    _copula = (;
        α = Param(_alpha, truncated(Normal(1.0, 10^5), 1.0, 50.0))
    )
    # Model
    copula = ModelWrapper(
        copulanames, deepcopy(_copula), (; rotation = _archimedeanrotation)
    )

elseif _modelname isa Joe
    #Joe Copula Names
    copulanames = Joe()
    subcopulas = Joe()
    # Parameter
    _copula = (;
        α = Param(_alpha, truncated(Normal(1.0, 10^5), 1.0, 50.0))
    )
    # Model
    copula = ModelWrapper(
        copulanames, deepcopy(_copula), (; rotation = _archimedeanrotation)
    )

elseif _modelname isa FactorCopula
####################################### Factor Copula - need to custom define individual subcopulas
    # Factor Copula Names
    copulanames = FactorCopula()
    subcopulas = (Joe(), Joe())
    # Parameter
    _copula = (;
            #!NOTE - has to be in same dimension as data ~ all in UNIFORM space! -> Careful if defined in real dimension in other example
            latent = Param( rand(n), [truncated(Normal(0.5, 1.0), 0.0, 1.0) for _ in Base.OneTo(n)]),
            #Normal
            factors = (;
                #Double Joe
                factor1 = (;α = Param(_alpha, truncated(Normal(1.0, 10^5), 1.0, 50.0))),
                factor2 = (;α = Param(_alpha, truncated(Normal(1.0, 10^5), 1.0, 50.0))),
            )
    )
    # Model
    copula = ModelWrapper(
        copulanames, deepcopy(_copula), (; copulanames = subcopulas, rotation = _archimedeanrotation )
    )
else
    println("Copula name not defined")
end

################################################################################
# Define (separate) marginal models

if _marginalname isa StockMarginal
    # Parameter
    param_marginal1 = (;
        μₛ = Param(0.1, truncated(Normal(0.0, 10.0^4), -1.0, 1.0)),
        # T marginal parameter ~ reparametrized as log(ν-2)
        ℓν = Param(0.0,  truncated( Normal(0.0, 10.0^3), -10.0, 25.)),
        #Initial S_0 and X_0
        S₀ = Param(data_real[begin].S, Fixed()),
        #Time discretization parameter ; set to 1 if daily data
        δ = Param(1.0/252.0, Fixed()),
        # Fixed volatility data
        X = Param( getindex.(data_real, 2), Fixed())
    )
    # Model
    model_marginal = ModelWrapper(StockMarginal(), param_marginal1)
elseif _marginalname isa VolatilityMarginal

    param_marginal2 = (;
        μᵥ = Param(exp(data_real[begin].X), truncated(Normal(10.0, 10.0^4), 0.0, 1.0)), #exp(data_real[begin].X)  = 0.027
        κ = Param(1.00, truncated(Normal(10.0, 10.0^4), 0.0, 100.0)),
        σ = Param(0.10, truncated(Normal(0.5, 10.0^4), 0.01, 2.0)),
        # T marginal parameter ~ reparametrized as log(ν-2)
        ℓν = Param(0.0,  truncated( Normal(0.0, 10.0^3), -10.0, 25.)),
        #Initial S_0 and X_0
        X₀ = Param(data_real[begin].X, Fixed()),
        #Time discretization parameter ; set to 1 if daily data
        δ = Param(1.0/252.0, Fixed()),
        # Fixed volatility data
        S = Param( getindex.(data_real, 1), Fixed())
    )
    model_marginal = ModelWrapper(VolatilityMarginal(), param_marginal2)
else
    println("")
end
################################################################################
# Define Stochastic Volatility Model

# Define Parameter
param_stochvol = (;
    # T marginal parameter ~ reparametrized as log(ν-2)
    #!NOTE: This is the drift (upwards+/downward-) of log(S&P500)
    μₛ = Param(0.0, truncated(Normal(0.0, 10.0^2), -1.0, 1.0)), #Normal(0.0, 1.0^1)
    #!NOTE: This is the mean value that variance = exp(X) > 0 will shift around
    μᵥ = Param(exp(data_real[begin].X), truncated(Normal(0.0, 1.0^2), 0.0, 1.0)), #Normal(0.0, 1.0^1) exp(data_real[begin].X)  = 0.027
    #!NOTE: Mean reversion parameter for volatility parameter X
    κ = Param(20.00, truncated(Normal(20.0, 10.0^4), 0.0, 100.0)), #10
    #!NOTE Noise term > 0 for both S (in sqrt(σ)) and X (in σ)
    σ = Param(0.50, truncated(Normal(0.5, 10.0^4), 0.01, 2.0)), #1.0
    copula = _copula,
    #Initial S_0 and X_0
    S₀ = Param(data_real[begin].S, Fixed()),
    X₀ = Param(data_real[begin].X, Fixed()), #3.0
    #Time discretization parameter ; set to 1 if daily data
    δ = Param(1.0/252.0, Fixed()), #1.0/252.0
)
model = ModelWrapper(StochasticVolatilityCopula(), param_stochvol,
    (; copulanames = copulanames, subcopulas = subcopulas, rotation = _archimedeanrotation, marginals = _marginals)
)
