adbackend_copula = adbackend #length(copula) > 50 ? :ReverseDiff : :ForwardDiff

################################################################################
#Define MCMC Kernels
_mcmc = NUTS(keys(tagged.parameter);
    GradientBackend = adbackend,
    init = PriorInitialization(1000) #NoInitialization()
)

_mcmc_copula = NUTS(keys(tagged_copula.parameter);
    GradientBackend = adbackend_copula,
    init = PriorInitialization(1000) #NoInitialization()
)

################################################################################
#Define IBIS Kernels
_ibis = SMC(_mcmc;
    jittermin = 1, jittermax = 5,
    jitterfun = maximum, jitterthreshold = 0.9, jitteradaption = UpdateTrue(),
    resamplingthreshold = .90,
    Ntuning = 35,
)
_ibis_copula = SMC(_mcmc_copula;
    jittermin = 1, jittermax = 5,
    jitterfun = maximum, jitterthreshold = 0.9, jitteradaption = UpdateTrue(),
    resamplingthreshold = .90,
    Ntuning = 35,
)
