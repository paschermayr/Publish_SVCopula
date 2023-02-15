_rho₀ = -.75
_alpha₀ = 4.0
################################################################################
#Assign Structures
if _modelname isa Gaussian
    #Normal Copula Names
    copulanames = Gaussian()
    subcopulas = Gaussian()
    # Parameter
    _copula = (;
        ρ = Param(truncated(Normal(0.0, 10^5), -1.0, 1.0), _rho₀, )
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
        ρ = Param(truncated(Normal(0.0, 10^5), -1.0, 1.0), _rho₀, ),
        #!NOTE: set df = 2 so have formula for quantile, else need to change copula-elliptical.jl code
#        df = Param(2, Fixed())
        df = Param(truncated( Normal(2.0, 10.0^3), 0.1, 100.), 2.0, )
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
        α = Param(truncated(Normal(1.0, 10^5), 0.0, 50.0), _alpha₀, )
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
        α = Param(truncated(Normal(1.0, 10^5), 0.0, 30.0), _alpha₀, )
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
        α = Param(truncated(Normal(1.0, 10^5), 1.0, 50.0), _alpha₀, )
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
        α = Param(truncated(Normal(1.0, 10^5), 1.0, 50.0), _alpha₀, )
    )
    # Model
    copula = ModelWrapper(
        copulanames, deepcopy(_copula), (; rotation = _archimedeanrotation)
    )
    println("Copula name not defined")
end

################################################################################
# Define Stochastic Volatility Model

# Define Parameter
param_stochvol = (;
    #!NOTE: This is the drift (upwards+/downward-) of log(S&P500)
    μₛ = Param(truncated(Normal(0.0, 10.0^2), -1.0, 1.0), 0.0, ), #Normal(0.0, 1.0^1)
    #!NOTE: This is the mean value that variance = exp(X) > 0 will shift around
    μᵥ = Param(truncated(Normal(0.0, 1.0^2), 0.0, 1.0), exp(data_real[begin].X), ), #Normal(0.0, 1.0^1) exp(data_real[begin].X)  = 0.027
    #!NOTE: Mean reversion parameter for volatility parameter X
    κ = Param(truncated(Normal(20.0, 10.0^4), 0.0, 100.0), 20.00, ), #10
    #!NOTE Noise term > 0 for both S (in sqrt(σ)) and X (in σ)
    σ = Param(truncated(Normal(0.5, 10.0^4), 0.01, 2.0), 0.50, ), #1.0
    copula = _copula,
    #Initial S_0 and X_0
    S₀ = Param(Fixed(), data_real[begin].S, ),
    X₀ = Param(Fixed(), data_real[begin].X, ), #3.0
    #Time discretization parameter ; set to 1 if daily data
    δ = Param(Fixed(), 1.0/252.0, ), #1.0/252.0
)
model = ModelWrapper(StochasticVolatilityCopula(), param_stochvol,
    (; copulanames = copulanames, subcopulas = subcopulas, rotation = _archimedeanrotation, marginals = _marginals)
)
