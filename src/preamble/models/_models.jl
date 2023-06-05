################################################################################
#Define Models
abstract type AbstractCopulas <: ModelName end

abstract type Archimedean <: AbstractCopulas end
abstract type Elliptical <: AbstractCopulas end

struct FactorCopula <: AbstractCopulas end
struct StochasticVolatilityCopula <: AbstractCopulas end

abstract type AbstractMarginals <: ModelName end

################################################################################
#Assign individual copula Reflection
abstract type ArchimedeanReflection end
struct Reflection0 <: ArchimedeanReflection end
struct Reflection90 <: ArchimedeanReflection end
struct Reflection180 <: ArchimedeanReflection end
struct Reflection270 <: ArchimedeanReflection end

function get_tails(copula::AbstractCopulas, reflection::ArchimedeanReflection, θ::NamedTuple)
    λₗ, λᵤ = get_tails(copula, θ)
    return λₗ, λᵤ
end
function get_tails(copula::Archimedean, reflection::Reflection90, θ::NamedTuple)
    #Compute tails and then switch lower and upper tails
    λₗ, λᵤ = get_tails(copula, θ)
    return λᵤ, λₗ
end
function get_tails(copula::Archimedean, reflection::Reflection270, θ::NamedTuple)
    #Compute tails and keep that fixed
    λₗ, λᵤ = get_tails(copula, θ)
    return λₗ, λᵤ
end

################################################################################
# Add utility functions
include(string("_utility.jl"))

################################################################################
# Implicit Copulas - Normal and T covered
include(string("copula-", "elliptical", ".jl"))

# Explicit Copulas - only Archimedean covered
include(string("copula-", "archimedean", ".jl"))
include(string("copula-", "bb", ".jl"))
#Frank version that is defined for all Reals, and hence does not need rotation
include(string("copula-", "frankunconstrained", ".jl"))

# Stochastic Volatility framework
#!NOTE: Marginal associated to ϵ (_marginal1) and ξ (_marginal2)
include(string("copula-", "stochasticvolatility", ".jl"))
