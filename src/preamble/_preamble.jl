################################################################################
#plotting defaults
plot_default_color = :rainbow_bgyrm_35_85_c71_n256
plot_default_size = (1000,1000)
_fontsize = 16
_axissize = 16

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

################################################################################
#Load prepared data in real domain
include("_data.jl")

################################################################################
# Define all models and load corresponding functions
include(string("models/_models.jl"))

################################################################################
# Settings for MCMC/SMC run
_modelname = Frank() #Gaussian() #TCop() #Clayton() #Frank() #Joe() #Gumbel()
_archimedeanreflection = Reflection0()
_marginals = (Distributions.Normal(), Distributions.Normal())
_realdata = true
#!NOTE - T Marginals with custom pullback for Reversediff makes using Cached ReverseDiff INCORRECT! Need to use untaped ReverseDiff if ReverseMode is used
adbackend = :ForwardDiff #:ForwardDiff #:ReverseDiffUntaped
