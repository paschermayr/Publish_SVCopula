################################################################################
import Pkg
cd(@__DIR__)
Pkg.activate(".")
Pkg.status()
include("src/_packages.jl");
Pkg.status()
#Pkg.instantiate()

################################################################################
# Set Preamble
include("src/_preamble.jl");

################################################################################
# Define setting
_modelname = TCop() #Gaussian() #TCop() #Clayton() #Frank() #Joe() #Gumbel()  #FactorCopula
_marginalname = nothing
_realdata = false
#!NOTE - T Marginals with custom pullback for Reversediff makes using Cached ReverseDiff INCORRECT! Need to use untaped ReverseDiff
adbackend = :ReverseDiffUntaped #:ForwardDiff #:ReverseDiffUntaped

################################################################################
# Set Model
include("src/_models.jl");

################################################################################
_SVsetting = string(
    "StochasticVolatility Errors - Copula-", typeof(_modelname),
    ", Marginals-", typeof(_marginals),
    _modelname isa FactorCopula ? string(", SubCopulas-", typeof.(model.arg.subcopulas)) : "",
    _modelname isa Archimedean ? string(", Rotation-", _archimedeanrotation) : "",
)
_Copsetting = string(
    "Copula - ", typeof(_modelname),
    ", Marginals-", typeof(_marginals),
    _modelname isa FactorCopula ? string(", SubCopulas-", typeof.(model.arg.subcopulas)) : "",
    _modelname isa Archimedean ? string(", Rotation-", _archimedeanrotation) : "",
)
length(model)
keys(model.val)
tagged = Tagged(model, (:μₛ, :μᵥ, :κ, :σ, :copula) ) #:ℓν,
keys(tagged.parameter)

length(copula)
keys(copula.val)
tagged_copula = Tagged(copula)#, (:ρ,) )
keys(tagged_copula.parameter)

################################################################################
# Set Algorithm
include("src/_algorithm.jl");

################################################################################
# Set Data
model.val
if _realdata
    data = data_real
elseif !_realdata && _modelname isa FactorCopula
    data, errorterms, _, latent = ModelWrappers.simulate(_rng, model, n)
    latent_copula, data_copula = ModelWrappers.simulate(_rng, copula, n)
elseif !_realdata && !(_modelname isa FactorCopula)
    data, errorterms, _ = ModelWrappers.simulate(_rng, model, n)
    data_copula = ModelWrappers.simulate(_rng, copula, n)
else
    println("Check settings for data and modelname.")
end
plot( getindex.(data, 1) )
################################################################################
# Create initial model and sample from prior
model_initial = deepcopy(model)
copula_initial = deepcopy(copula)
Objective(model, data, tagged)(model.val)
_modelinit = ModelWrappers.PriorInitialization(10^5)
_modelinit(_rng, "-", Objective(model_initial, data, tagged))
Objective(model_initial, data, tagged)(model_initial.val)

################################################################################
################################################################################
################################################################################
_N1 = 2000
_burnin = 1000
# MCMC - Sample Stochastic Volatility model
trace_mcmc, algorithm_mcmc = sample(_rng, model_initial, data, _mcmc;
    default = SampleDefault(;
#        report = ProgressReport(; bar = true, log = ConsoleLog()),
        iterations = _N1, chains = mcmcchains, burnin = _burnin,
        printoutput = true, safeoutput = false
    )
)
Baytes.savetrace(trace_mcmc, model, algorithm_mcmc, string("MCMC - ", _SVsetting," - Trace"))
####################################
# Basic Plots and output
plotChain(trace_mcmc, tagged; burnin = _burnin)
Plots.savefig( string("MCMC - ", _SVsetting," - Chain.png") )

transform_mcmc = Baytes.TraceTransform(trace_mcmc, model, tagged,
    TransformInfo(collect(1:trace_mcmc.summary.info.Nchains), [trace_mcmc.summary.info.Nalgorithms], (_burnin+1):1:trace_mcmc.summary.info.iterations)
)
summary(trace_mcmc, algorithm_mcmc, transform_mcmc, PrintDefault())
BaytesInference.plotChains(trace_mcmc, transform_mcmc)

postmean_vec, postmean_tup = trace_to_posteriormean(trace_mcmc, transform_mcmc)
model_new = deepcopy(model)
fill!(model_new, postmean_tup)
model_new.val
####################################
# Contour plots
copula_contour = deepcopy(copula)
fill!(copula_contour, model_new.val.copula)
copula_contour.val

#Real data Get error terms based on posterior parameter of model
errors = Data_to_Error(model_new.val, data)
errorsᵤ = cdf.(Normal(), errors)
scatter(errorsᵤ[1,:], errorsᵤ[2,:])
plotContour(copula_contour, errorsᵤ; marginal = Cauchy())
Plots.savefig( string("MCMC - ", _SVsetting," - Contour posterior parameter - Cauchy Marginals.png") )
plotContour(copula_contour, errorsᵤ)
Plots.savefig( string("MCMC - ", _SVsetting," - Contour posterior parameter - Normal Marginals.png") )

#Simulate data with posterior samples
_Nsimulations = 5
p = plot(
    layout = (_Nsimulations,2),
    size(1000, 1000),
    foreground_color_legend = :transparent,
    background_color_legend = :transparent,
    label=false
)
for iter in Base.OneTo(_Nsimulations)
    if _modelname isa FactorCopula
        data_sim, _ = ModelWrappers.simulate(_rng, model_new, n)
    elseif !(_modelname isa FactorCopula)
        data_sim, _ = ModelWrappers.simulate(_rng, model_new, n)
    end
    Nplot = iter * 2
    plot!(getindex.(data_sim, 1), color="black", subplot=Nplot-1,  label=false)
    plot!(getindex.(data_sim, 2), color="red", subplot=Nplot,label=false)
end
plot!(ylabel="Log(S&P500)", subplot=9)
plot!(ylabel="X=log(V)", subplot=10)
p
Plots.savefig( string("MCMC - ", _SVsetting," - Simulated data.png") )

#save data and model parameter use for simulated data
using JLD2
JLD2.jldsave(
    join((_SVsetting, " - Simulated Data and Parameter.jld2"));
    model=model,
    data=data
)

################################################################################
################################################################################
################################################################################
# IBIS - Sample Stochastic Volatility model
trace_smcIBIS, algorithm_smcIBIS = sample(_rng, model_initial, data, _ibis;
    default = SampleDefault(;
        dataformat = Expanding(smcinit),
        iterations = n, chains = smcchains, burnin = 0,
        printoutput = false,
    )
)
Baytes.savetrace(trace_smcIBIS, model, algorithm_smcIBIS, string("IBIS - ", _SVsetting," - Trace"))

plotChain(trace_smcIBIS, tagged; burnin = 10)
Plots.savefig( string("IBIS - ", _SVsetting," - Chain.png") )

BaytesInference.plotDiagnostics(trace_smcIBIS.diagnostics, algorithm_smcIBIS)
Plots.savefig( string("IBIS - ", _SVsetting," - Diagnostics.png") )

transform_ibis = TraceTransform(
    trace_smcIBIS, model_initial, tagged,
    TransformInfo(
        collect(Base.OneTo(trace_smcIBIS.summary.info.Nchains)),
        collect(Base.OneTo(trace_smcIBIS.summary.info.Nalgorithms)),
        Int(round(trace_smcIBIS.summary.info.iterations/1.25)),
        trace_smcIBIS.summary.info.thinning,
        trace_smcIBIS.summary.info.iterations
    )
)
summary(trace_smcIBIS, algorithm_smcIBIS, transform_ibis)
postmean_vec, postmean_tup = trace_to_posteriormean(trace_smcIBIS, transform_ibis)
model_new = deepcopy(model)
fill!(model_new, postmean_tup)
model_new.val
####################################
# Contour plots
copula_contour = deepcopy(copula)
fill!(copula_contour, model_new.val.copula)
copula_contour.val

#Real data Get error terms based on posterior parameter of model
errors = Data_to_Error(model_new.val, data)
errorsᵤ = cdf.(Normal(), errors)
plotContour(copula_contour, errorsᵤ; marginal = Cauchy())
Plots.savefig( string("IBIS - ", _SVsetting," - Contour posterior parameter - Cauchy Marginals.png") )
plotContour(copula_contour, errorsᵤ)
Plots.savefig( string("IBIS - ", _SVsetting," - Contour posterior parameter - Gaussian Marginals.png") )

#Simulate data with posterior samples
_Nsimulations = 5
p = plot(
    layout = (_Nsimulations,2),
    size(1000, 1000),
    foreground_color_legend = :transparent,
    background_color_legend = :transparent,
    label=false
)
for iter in Base.OneTo(_Nsimulations)
    if _modelname isa FactorCopula
        data_sim, _ = ModelWrappers.simulate(_rng, model_new, n)
    elseif !(_modelname isa FactorCopula)
        data_sim, _ = ModelWrappers.simulate(_rng, model_new, n)
    end
    Nplot = iter * 2
    plot!(getindex.(data_sim, 1), color="black", subplot=Nplot-1,  label=false)
    plot!(getindex.(data_sim, 2), color="red", subplot=Nplot,label=false)
end
plot!(ylabel="Log(S&P500)", subplot=9)
plot!(ylabel="Log(VIX)", subplot=10)
p
Plots.savefig( string("IBIS - ", _SVsetting," - Simulated Data.png") )

################################################################################
#Check predictions vs actual data
using BaytesCore, StatsBase
_effective_iter = 1:length(trace_smcIBIS.diagnostics)
_fonttemp = 16
ℓweights = [exp.( trace_smcIBIS.diagnostics[iter].ℓweights .- BaytesCore.logsumexp(trace_smcIBIS.diagnostics[iter].ℓweights) ) for iter in _effective_iter]

all_pred = [trace_smcIBIS.diagnostics[_effective_iter][iter].base.prediction for iter in eachindex(trace_smcIBIS.diagnostics[_effective_iter])]
S_pred = [getindex.(all_pred[iter], 1) for iter in eachindex(all_pred)]
X_pred = [getindex.(all_pred[iter], 2) for iter in eachindex(all_pred)]
all_pred

#!NOTE: +2 because we start with ( size(latent, 1) - length(predictionᵛ) ) for warmup, then predict for +2 at +1 after warmup
#!NOTE: Burnin included here in all_pred
Npred = length(S_pred[begin])
smcstart = length(data) - length(all_pred) + 2
start_date = length(data) - length(all_pred) + 2
using Test
@test start_date >= smcstart

plot_prediction = plot(;
    layout=(2, 1),
    size=(1000,1000),
    legend=:topright,
    xguidefontsize=_fonttemp,
    yguidefontsize=_fonttemp,
    legendfontsize=_fonttemp,
    xtickfontsize=_fonttemp,
    ytickfontsize=_fonttemp,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
)

plot!(dates_real[start_date:end], mean.(S_pred)[(start_date - smcstart + 1):end-1],
    ylim = (8,9),
    label="Prediction", ylabel = "S = log(S&P500)", subplot=1, color="black"
)
plot!(dates_real[start_date:end], mean.(X_pred)[(start_date - smcstart + 1):end-1],
    ylim = (-6, -1),
    label="Prediction", ylabel = "X = log((VIX/100)²)", subplot=2, color="black"
)

plot!(dates_real[start_date:end], [ mean(S_pred[iter], StatsBase.weights(ℓweights[iter])) for iter in (start_date - smcstart + 1):(length(S_pred)-1) ],
    label="Weigthed Prediction", subplot=1
)
plot!(dates_real[start_date:end], [ mean(X_pred[iter], StatsBase.weights(ℓweights[iter])) for iter in (start_date - smcstart + 1):(length(X_pred)-1) ],
    label="Weigthed Prediction", subplot=2
)

for t in (start_date - smcstart + 1):(length(S_pred)-1)
    scatter!(repeat([dates_real[start_date+t-1]], Npred), S_pred[t], label=false, subplot=1,
        markerstrokewidth=0.0, alpha=0.25, shape=:o, markerstrokecolor="grey", markersize=1, color="grey"
    )
    scatter!(repeat([dates_real[start_date+t-1]], Npred), X_pred[t], label=false, subplot=2,
        markerstrokewidth=0.0, alpha=0.25, shape=:o, markerstrokecolor="grey", markersize=1, color="grey"
    )
end
plot_prediction
plot!(dates_real[start_date:end], getindex.(data, 1)[start_date:end], label="Observed Data", subplot=1, color="blue")
plot!(dates_real[start_date:end], getindex.(data, 2)[start_date:end], label="Observed Data", subplot=2, color="blue")
plot_prediction
Plots.savefig( string("IBIS - ", _SVsetting," - IBIS prediction.png") )
