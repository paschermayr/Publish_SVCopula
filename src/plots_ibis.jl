################################################################################
################################################################################
################################################################################
#Individual analysis

import Pkg
cd(@__DIR__)
Pkg.activate(".")
Pkg.status()
#If environment activated for first time, uncomment next line to install all libraries used in project
#Pkg.instantiate()
include("preamble/_packages.jl");

################################################################################
# Set Preamble
include("preamble/_preamble.jl");
include("preamble/_initialmodel.jl");

################################################################################
# Load saved model of choice
_subfolder ="/saved/real/ibis/"
_savedtrace = "IBIS - StochasticVolatility Errors - Copula-Clayton, Marginals-Tuple{Normal{Float64}, Normal{Float64}}, Reflection-Reflection90() - Trace.jld2"

f_model   =   jldopen(string(pwd(), _subfolder, _savedtrace))
trace_smcIBIS = read(f_model, "trace")
model = read(f_model, "model")
algorithm_smcIBIS = read(f_model, "algorithm")
tagged = algorithm_smcIBIS.tune.tagged
data = data_real

################################################################################
#overwrite preamble for custom plots
_modelname = model.arg.subcopulas
_archimedeanreflection = model.arg.reflection

_Nchains = collect( 1:length(trace_smcIBIS.val) )
_Nalgorithms = [1]
_burnin = 500
_maxiter = trace_smcIBIS.summary.info.iterations
_effective_iter = (_burnin+1):1:_maxiter
_fonttemp = 16

_SVsetting = string(
    "StochasticVolatility Errors - Copula-", typeof(_modelname),
    ", Marginals-", typeof(_marginals),
    _modelname isa FactorCopula ? string(", SubCopulas-", typeof.(model.arg.subcopulas)) : "",
    _modelname isa Archimedean ? string(", Reflection-", _archimedeanreflection) : "",
)
_Copsetting = string(
    "Copula - ", typeof(_modelname),
    ", Marginals-", typeof(_marginals),
    _modelname isa FactorCopula ? string(", SubCopulas-", typeof.(model.arg.subcopulas)) : "",
    _modelname isa Archimedean ? string(", Reflection-", _archimedeanreflection) : "",
)

################################################################################
transform_ibis = TraceTransform(
    trace_smcIBIS, model, tagged,
    TransformInfo(_Nchains, _Nalgorithms, _effective_iter)
)

################################################################################
# Basic Plots and output

plotChain(trace_smcIBIS, tagged; burnin = 10)
Plots.savefig( string("Chp5_SMCTrace_WinningModel.pdf") )

BaytesInference.plotDiagnostics(trace_smcIBIS.diagnostics, algorithm_smcIBIS)
Plots.savefig( string("IBIS - ", _SVsetting," - Diagnostics.png") )

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

################################################################################
################################################################################
################################################################################
# Joint Analysis: Check cumulative log predictive likelihood

import Pkg
cd(@__DIR__)
Pkg.activate(".")
Pkg.status()
#If environment activated for first time, uncomment next line to install all libraries used in project
#Pkg.instantiate()
include("preamble/_packages.jl");

################################################################################
# Set Preamble
include("preamble/_preamble.jl");
include("preamble/_initialmodel.jl");

################################################################################
# Load saved model of choice
_subfolder ="/saved/real/ibis/"

_savedtraces = [
    "IBIS - StochasticVolatility Errors - Copula-BB1, Marginals-Tuple{Normal{Float64}, Normal{Float64}}, Reflection-Reflection90() - Trace.jld2",
    "IBIS - StochasticVolatility Errors - Copula-BB1, Marginals-Tuple{Normal{Float64}, Normal{Float64}}, Reflection-Reflection270() - Trace.jld2",
    "IBIS - StochasticVolatility Errors - Copula-BB7, Marginals-Tuple{Normal{Float64}, Normal{Float64}}, Reflection-Reflection270() - Trace.jld2",
    "IBIS - StochasticVolatility Errors - Copula-Clayton, Marginals-Tuple{Normal{Float64}, Normal{Float64}}, Reflection-Reflection90() - Trace.jld2",
    "IBIS - StochasticVolatility Errors - Copula-Clayton, Marginals-Tuple{Normal{Float64}, Normal{Float64}}, Reflection-Reflection270() - Trace.jld2",
    "IBIS - StochasticVolatility Errors - Copula-Frank, Marginals-Tuple{Normal{Float64}, Normal{Float64}}, Reflection-Reflection90() - Trace.jld2",
    "IBIS - StochasticVolatility Errors - Copula-Gaussian, Marginals-Tuple{Normal{Float64}, Normal{Float64}} - Trace.jld2",
    "IBIS - StochasticVolatility Errors - Copula-Gumbel, Marginals-Tuple{Normal{Float64}, Normal{Float64}}, Reflection-Reflection90() - Trace.jld2",
    "IBIS - StochasticVolatility Errors - Copula-Gumbel, Marginals-Tuple{Normal{Float64}, Normal{Float64}}, Reflection-Reflection270() - Trace.jld2",
    "IBIS - StochasticVolatility Errors - Copula-TCop, Marginals-Tuple{Normal{Float64}, Normal{Float64}} - Trace.jld2"
]

################################################################################
# Assign copulas and names based on _savedtrace
_copulas = []
_copulas_names = []

for _modelname in _savedtraces
    f_model   =   jldopen(string(pwd(), _subfolder, _modelname))
    model = read(f_model, "model")

    _tagged = Tagged(model, :copula)
    copula_temp = ModelWrapper(model.val.copula, (; reflection = model.arg.reflection), _tagged.info, model.arg.subcopulas)
    name_temp = get_copula_name(model.arg.subcopulas, model.arg.reflection)

    push!(_copulas_names, name_temp)
    push!(_copulas, copula_temp)
end
length(_copulas_names) == length(_copulas) == length(_savedtraces)
_copulas_names

################################################################################
marginal_lik = []

for (_counter, savedtrace) in enumerate(_savedtraces)
    println("Computing model ", _counter)

    f_model   =   jldopen(string(pwd(), _subfolder, savedtrace))
    trace_smcIBIS = read(f_model, "trace")
    model = read(f_model, "model")
    algorithm_smcIBIS = read(f_model, "algorithm")
    tagged = algorithm_smcIBIS.tune.tagged
    # Compute cumulative incremental likelihood
    push!(marginal_lik, cumsum([trace_smcIBIS.diagnostics[chain].ℓincrement for chain in eachindex(trace_smcIBIS.diagnostics)]))
end

plot_cumℓincrement(
    marginal_lik,
    dates_real[(end-length(marginal_lik[begin])+1):end],
    String.(_copulas_names),
    6
)
ylabel!("Cumulative Log Predictive Likelihood", subplot=1)
ylabel!(string("CLPBF of ", _copulas_names[6]), subplot=2)
Plots.savefig( string("Chp5_SMC_ModelComparison.pdf") )


################################################################################
function plot_clpbf(
    #A Vector for different cumulative log likelihood increments, see 'compute_cumℓincrement'
    cumℓincrement::Vector{T},
    #X axis Vector (with dates) to specify timeframe in plots
    dates = 1:length(cumℓincrement[begin]),
    #Modelnames
    modelnames::Vector{String} = [string("Model ", iter) for iter in eachindex(cumℓincrement)],
    benchmarkmodel::Int64 = 1;
    plotsize=(1000,1000),
    param_color=:rainbow_bgyrm_35_85_c71_n256,
    fontsize=16,
    axissize=16,
) where {T}
    ArgCheck.@argcheck length(cumℓincrement) == length(modelnames)

    plot_score = plot(;
        layout=(1, 1),
        foreground_color_legend = :transparent,
        background_color_legend = :transparent,
        size=plotsize,
        xguidefontsize=fontsize,
        yguidefontsize=fontsize,
        legendfontsize=fontsize,
        xtickfontsize=axissize,
        ytickfontsize=axissize,
    )
#Plot Log Bayes Factor of all Models against chosen benchmark
    ℓbayes = [cumℓincrement[benchmarkmodel] .- cumℓincrement[iter] for iter in eachindex(cumℓincrement)]
    for iter in eachindex(cumℓincrement)
        Plots.plot!(
            dates, ℓbayes[iter],
            label= modelnames[iter], legend=:topleft,
            ylabel= string("CLPBF of ", modelnames[benchmarkmodel]),
            color=Plots.palette(param_color, length(cumℓincrement)+1)[iter],
            subplot=1
        )
    end
    return plot_score
end

plot_clpbf(
    marginal_lik,
    dates_real[(end-length(marginal_lik[begin])+1):end],
    String.(_copulas_names),
    6
)
#ylabel!("CLPBF of Frank Copula", subplot=1)
Plots.savefig( string("Chp5_SVM_SMC_ModelComparison.pdf") )
