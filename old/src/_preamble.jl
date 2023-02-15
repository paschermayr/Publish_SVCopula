################################################################################
#constants
data_delay = 1000 #Number of days before using current data
n = 1000 #number of data points
cvg = 1.0 #pf coverage
_rng = Random.Xoshiro(1)
_N1 = 2000 # MCMC iterations
_burnin = 1000 #max(0, Int(round(_N1/10)))
mcmcchains = 4
# SMC specific variations
smcchains = 100 #100
smcinit = 250

plot_default_color = :rainbow_bgyrm_35_85_c71_n256
plot_default_size = (1000,1000)
_fontsize = 16
_axissize = 16

################################################################################
#Load prepared data in real domain
include("_data.jl")

################################################################################
#Load Models
include("models/core/core.jl")

abstract type AbstractCopulas <: ModelName end
abstract type Archimedean <: AbstractCopulas end
abstract type Elliptical <: AbstractCopulas end
struct FactorCopula <: AbstractCopulas end
struct StochasticVolatilityCopula <: AbstractCopulas end

abstract type AbstractMarginals <: ModelName end

################################################################################
#Assign individual copula rotation
abstract type ArchimedeanRotation end
struct Rotation0 <: ArchimedeanRotation end
struct Rotation90 <: ArchimedeanRotation end
struct Rotation180 <: ArchimedeanRotation end
struct Rotation270 <: ArchimedeanRotation end

################################################################################
# Explicit Copulas - only Archimedean covered
_alpha = 4.0
_archimedeanrotation = Rotation90()
include(string("models/copula-", "archimedean", ".jl"))

# Implicit Copulas - Normal and T covered
_rho = -.75
include(string("models/copula-", "elliptical", ".jl"))

# Factor Copulas - need to call all explicit/implicit copulas - can be arbitrary combination of copulas
include(string("models/copula-", "factor", ".jl"))

# Stochastic Volatility framework
_marginals = (Distributions.Normal(), Distributions.Normal())

include(string("models/copula-", "stochasticvolatility", ".jl"))

# Marginals - S
include(string("models/marginals-", "s", ".jl"))
# Marginals - X
include(string("models/marginals-", "x", ".jl"))
