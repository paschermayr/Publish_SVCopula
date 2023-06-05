################################################################################
################################################################################
################################################################################
#MCMC analysis

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
_subfolder ="/output/Simulation_Benchmark/"

_savedtraces_mcmc = [
    "MCMC - StochasticVolatility Errors - Copula-BB1, Marginals-Tuple{Normal{Float64}, Normal{Float64}}, Reflection-Reflection270() - Trace.jld2",
    "MCMC - StochasticVolatility Errors - Copula-BB7, Marginals-Tuple{Normal{Float64}, Normal{Float64}}, Reflection-Reflection270() - Trace.jld2",

    "MCMC - StochasticVolatility Errors - Copula-Clayton, Marginals-Tuple{Normal{Float64}, Normal{Float64}}, Reflection-Reflection270() - Trace.jld2",
    "MCMC - StochasticVolatility Errors - Copula-FrankUnconstrained, Marginals-Tuple{Normal{Float64}, Normal{Float64}} - Trace.jld2",
    "MCMC - StochasticVolatility Errors - Copula-Gaussian, Marginals-Tuple{Normal{Float64}, Normal{Float64}} - Trace.jld2",
    "MCMC - StochasticVolatility Errors - Copula-Gumbel, Marginals-Tuple{Normal{Float64}, Normal{Float64}}, Reflection-Reflection270() - Trace.jld2",
    "MCMC - StochasticVolatility Errors - Copula-Joe, Marginals-Tuple{Normal{Float64}, Normal{Float64}}, Reflection-Reflection270() - Trace.jld2",
    "MCMC - StochasticVolatility Errors - Copula-TCop, Marginals-Tuple{Normal{Float64}, Normal{Float64}} - Trace.jld2",
]
################################################################################
# Assign copulas and names based on _savedtrace
_copulas = []
_copulas_names = []
for _modelname in _savedtraces_mcmc
    f_model   =   jldopen(string(pwd(), _subfolder, _modelname))
    model = read(f_model, "model")

    _tagged = Tagged(model, :copula)
    copula_temp = ModelWrapper(model.val.copula, (; reflection = model.arg.reflection), _tagged.info, model.arg.subcopulas)
    name_temp = get_copula_name(model.arg.subcopulas, model.arg.reflection)

    push!(_copulas_names, name_temp)
    push!(_copulas, copula_temp)
end
length(_copulas_names) == length(_copulas) == length(_savedtraces_mcmc)
_copulas_names

################################################################################
#WAIC and DIC computations
# function to go from real data to uniform errors errorsᵤ
function data_to_errorᵤ(marginals, θ, data)
    ## Obtain ϵ and ξ from S and X
    errors = Data_to_Error(θ, data)
    ## Take cdf of error terms to go to uniform space
    errorsᵤ = cdf.(marginals, errors)
    return errorsᵤ
end
function data_to_errors(marginals, θ, data)
    ## Obtain ϵ and ξ from S and X
    errors = Data_to_Error(θ, data)
    ## Take cdf of error terms to go to uniform space
    errorsᵤ = cdf.(marginals, errors)
    return errors, errorsᵤ
end

# Set Base Model
f_modelbase = jldopen(string(pwd(), _subfolder, _savedtraces_mcmc[begin]))
trace_mcmcbase = read(f_modelbase, "trace")
_Nchains = collect( 1:length(trace_mcmcbase.val) )
_Nalgorithms = [1]
_burnin = 1000
_maxiter = trace_mcmcbase.summary.info.iterations
_effective_iter = (_burnin+1):1:_maxiter
_fonttemp = 16

f_modelbenchmark = jldopen(string(pwd(), _subfolder,
    "StochasticVolatility Errors - Copula-FrankUnconstrained, Marginals-Tuple{Normal{Float64}, Normal{Float64}} - FrankUnconstrained_SimulationBenchmark.jld2"
    )
)
model = read(f_modelbenchmark, "model")
data = read(f_modelbenchmark, "data")
errors = read(f_modelbenchmark, "errors")

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

transform_mcmc = Baytes.TraceTransform(trace_mcmcbase, model, Tagged(model, trace_mcmcbase.summary.info.printedparam.tagged),
    TransformInfo(_Nchains, _Nalgorithms, _effective_iter)
)

################################################################################
# Preallocate results
dic = []
dic_alternative = []
waic = []

dic_all = []
dic_all_alternative = []
waic_all = []

cop_plots = []
cop_statistics = []

################################################################################
#_counter=2
#savedtrace = _savedtraces[_counter]

for (_counter, savedtrace) in enumerate(_savedtraces_mcmc)
#for savedtrace in _savedtraces
    println("Computing model ", _counter)

## Load trace
    f_model   =   jldopen(string(pwd(), _subfolder, savedtrace))
    trace_mcmc = read(f_model, "trace")
    model = read(f_model, "model")
    algorithm_mcmc = read(f_model, "algorithm")
    tagged = Tagged(model, trace_mcmc.summary.info.printedparam.tagged)
    model_copula = _copulas[_counter]

## Necessary mcmc diagnostics
    # First get all relevant parameter, and remove burnin
    transform_all = Baytes.TraceTransform(trace_mcmc, model, Tagged(model),
        TransformInfo(_Nchains, _Nalgorithms, _effective_iter)
    )
    savechainsummary(trace_mcmc, transform_all, PrintDefault(;Ndigits=2), string("output/MCMC - ", _copulas_names[_counter]," - Chaindiagnostics"))

    chain_posterior = merge_chainvals(trace_mcmc, transform_all)
    #Then obtain posterior mean
    chain_posterior = merge_chainvals(trace_mcmc, transform_all)
    postmean_vec, postmean_tup = trace_to_posteriormean(trace_mcmc, transform_all)
    #From data to errors
    _errs = map(θ -> data_to_errors(model.arg.marginals, θ, data), chain_posterior)
    _errors = getindex.(_errs, 1)
    _errorsᵤ = getindex.(_errs, 2)
    # Compute DIC relevant errors
    _err, _errᵤ = data_to_errors(model.arg.marginals, postmean_tup, data)
    N_errors = length(_err)
## Compute Copula specific statistics
    _cop_stats = [ get_copuladiagnostics(model_copula.id, model.arg.reflection, chain_posterior[iter].copula, _errorsᵤ[iter]) for iter in eachindex(chain_posterior) ]
    push!(cop_statistics, _cop_stats)

## Compute WAIC relevant statistics - Incremental likelihood for data_t for t=1:T for each posterior sample
    # Compute likelihood from copula part
    incremental_ℓlikᵥ = map(iter ->
        map(dat ->
            cumℓlikelihood(model.arg.subcopulas, model.arg.reflection, chain_posterior[iter].copula, dat),
            eachcol(_errorsᵤ[iter])
        ), eachindex(chain_posterior)
    )
    # Compute Jacobian part
    incremental_ℓjac = map(iter ->
        map(dat ->
            logpdf(model.arg.marginals[1], dat[1]) + logpdf(model.arg.marginals[2], dat[2]) + -length(dat)*log(chain_posterior[iter].σ),
            eachcol(_errors[iter])
        ), eachindex(chain_posterior)
    )
    # Compute sum of it
    incremental_ℓpost = [ [incremental_ℓlikᵥ[iter][idx] + incremental_ℓjac[iter][idx] for idx in eachindex(incremental_ℓlikᵥ[iter])] for iter in eachindex(incremental_ℓlikᵥ)]
    #WAIC - copula part
    model_waic = compute_waic(incremental_ℓlikᵥ)
    push!(waic, model_waic)
    #WAIC - everything
    model_waic_all = compute_waic(incremental_ℓpost)
    push!(waic_all, model_waic_all)

## Compute DIC relevant statistics - Incremental likelihood for data_t for t=1:T for each posterior sample
    # Compute p(data | theta_posteriormean)
    ℓlikθ = _cumℓlikelihood(model.arg.subcopulas, model.arg.reflection, postmean_tup.copula, _errᵤ)
    ℓlikᵥ = map( iter ->
        _cumℓlikelihood(model.arg.subcopulas, model.arg.reflection, chain_posterior[iter].copula,
        _errorsᵤ[iter]
        ), eachindex(chain_posterior)
    )
    # Jacobian part
    ℓjacθ = sum( logpdf(model.arg.marginals[1], _errₜ[1]) + logpdf(model.arg.marginals[2], _errₜ[2]) for _errₜ in eachcol(_err) ) + -N_errors*log(postmean_tup.σ)
    ℓjacᵥ = map( iter ->
        sum( logpdf(model.arg.marginals[1], _errₜ[1]) + logpdf(model.arg.marginals[2], _errₜ[2]) for _errₜ in eachcol(_errors[iter]) ) + -N_errors*log(chain_posterior[iter].σ),
        eachindex(chain_posterior)
    )
    # posterior
    ℓpostθ = ℓlikθ + ℓjacθ
    ℓpostᵥ = ℓlikᵥ .+ ℓjacᵥ
    #Compute p_DIC
    model_dic = compute_dic(ℓlikθ, ℓlikᵥ)
    model_dic_alternative = compute_dic(ℓlikθ, ℓlikᵥ, var(ℓlikᵥ)/2 )
    push!(dic, model_dic)
    push!(dic_alternative, model_dic_alternative)

    model_dic_all = compute_dic(ℓpostθ, ℓpostᵥ)
    model_dic_all_alternative = compute_dic(ℓpostθ, ℓpostᵥ, var(ℓpostᵥ)/2 )
    push!(dic_all, model_dic_all)
    push!(dic_all_alternative, model_dic_all_alternative)

# Compute Contour plots
    #Set copula parameter to posterior mean
    fill!(model_copula, postmean_tup.copula)
    @unpack marginals = model.arg
    postmean_errors =  Data_to_Error(postmean_tup, data)
    postmean_errorsᵤ = hcat(
    cdf.(marginals[1], postmean_errors[1,:]),
    cdf.(marginals[2], postmean_errors[2,:])
    )'
    p = plotContour(model_copula, postmean_errorsᵤ, _copulas_names[_counter]; marginal = Normal())
#    push!(cop_plots, p)
    push!(cop_plots,
        (;  name = _copulas_names[_counter],
            plot = p,
            copula = model_copula,
            dataᵤ = postmean_errorsᵤ
        )
    )
end

################################################################################
# Print contour plots

allcopulaplots = [cop_plots[iter].plot for iter in eachindex(cop_plots)]
_fonttemp2 = 10
plot_cop = plot(
#    allcopulaplots[1], allcopulaplots[2], allcopulaplots[3];
    allcopulaplots...;
#    allcopulaplots[1], allcopulaplots[2], allcopulaplots[3], allcopulaplots[4], allcopulaplots[5], allcopulaplots[6];
    #label = false, #_copulas_names,
    layout=(6, 2),
    size=(1000,1000),
#    legend=:topleft,
    xguidefontsize=_fonttemp2,
    yguidefontsize=_fonttemp2,
    legendfontsize=_fonttemp2,
    xtickfontsize=_fonttemp2,
    ytickfontsize=_fonttemp2,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
)
Plots.savefig( string("Chp5_ContourPlots.pdf") )

################################################################################
# Print copula diagnostics - Compute Quantiles for each statistic and each copula
_copula_statistics = keys(cop_statistics[begin][begin])
_copula_quantiles = [.025, .25, .5, .75, .975]
_copula_table = []
for (iter, name) in enumerate(_copulas_names)
    _copstats = cop_statistics[iter]
    stats = [ quantile(getfield.(_copstats, stat), _copula_quantiles) for stat in _copula_statistics ]
    nt = merge((; name = _copulas_names[iter]),
        NamedTuple{_copula_statistics}([ quantile(getfield.(_copstats, stat), _copula_quantiles) for stat in _copula_statistics ])
    )
    push!(_copula_table, nt )
end
_copula_table
round.( reduce(hcat, getfield.(_copula_table, :λₗ) )' ; digits = 2)

using PrettyTables
outputbackend = :text
println("## Lower-Upper Tail Dependence: ")
PrettyTables.pretty_table(
    round.( reduce(hcat, getfield.(_copula_table, :λₗ) )'; digits = 3), backend = Val(outputbackend), row_labels = _copulas_names, header = string.(["Q2.5", "Q25.0", "Q50.0", "Q75.0", "Q97.5"])
)

println("## Upper-Lower Tail Dependence: ")
PrettyTables.pretty_table(
    round.( reduce(hcat, getfield.(_copula_table, :λᵤ) )'; digits = 3), backend = Val(outputbackend), row_labels = _copulas_names, header = string.(["Q2.5", "Q25.0", "Q50.0", "Q75.0", "Q97.5"])
)

println("## Spearman's rho: ")
PrettyTables.pretty_table(
    round.( reduce(hcat, getfield.(_copula_table, :ρ_spearman) )'; digits = 3), backend = Val(outputbackend), row_labels = _copulas_names, header = string.(["Q2.5", "Q25.0", "Q50.0", "Q75.0", "Q97.5"])
)

println("## Kendall's tau: ")
PrettyTables.pretty_table(
    round.( reduce(hcat, getfield.(_copula_table, :τ_kendall) )'; digits = 3), backend = Val(outputbackend), row_labels = _copulas_names, header = string.(["Q2.5", "Q25.0", "Q50.0", "Q75.0", "Q97.5"])
)

################################################################################
#Return WAIC and DIC as table
using PrettyTables

dic
dic_names = collect(keys(dic[begin]))
dic_table = reduce(hcat, [ getindex.(dic, iter) for iter in eachindex(dic_names)])
dic_ranking = sortperm(dic_table[:,1])

dic_all
dic_all_names = collect(keys(dic_all[begin]))
dic_all_table = reduce(hcat, [ getindex.(dic_all, iter) for iter in eachindex(dic_all_names)])
dic_all_ranking = sortperm(dic_all_table[:,1])

dic_all_alternative
dic_all_alternative_names = collect(keys(dic_all_alternative[begin]))
dic_all_alternative_table = reduce(hcat, [ getindex.(dic_all_alternative, iter) for iter in eachindex(dic_all_alternative_names)])
dic_all_alternative_ranking = sortperm(dic_all_alternative_table[:,1])

waic
waic_names = collect(keys(waic[begin]))
waic_table = reduce(hcat, [ getindex.(waic, iter) for iter in eachindex(waic_names)])
waic_ranking = sortperm(waic_table[:,1])

waic_all
waic_all_names = collect(keys(waic_all[begin]))
waic_all_table = reduce(hcat, [ getindex.(waic_all, iter) for iter in eachindex(waic_all_names)])
waic_all_ranking = sortperm(waic_all_table[:,1])

################################################################################
_prettytableoutput = :text
noranking = collect(1:length(dic_ranking))
dic_ranking = dic_all_ranking = waic_ranking = waic_all_ranking = noranking

println("## DIC - Copula and Marginal Part: ")
PrettyTables.pretty_table(
    round.(dic_all_table; digits=2)[dic_all_ranking,:], backend = Val(_prettytableoutput), row_labels = _copulas_names[dic_all_ranking], header = dic_all_names
)
println("## WAIC - Copula and Marginal Part: ")
PrettyTables.pretty_table(
    round.(waic_all_table; digits=2)[waic_all_ranking,:], backend = Val(_prettytableoutput), row_labels = _copulas_names[waic_all_ranking], header = waic_all_names
)

println("## DIC (alternative)- Copula and Marginal Part: ")
PrettyTables.pretty_table(
    round.(dic_all_alternative_table; digits=2)[dic_all_alternative_ranking,:], backend = Val(_prettytableoutput), row_labels = _copulas_names[dic_all_alternative_ranking], header = dic_all_alternative_names
)




################################################################################
################################################################################
################################################################################
# Sequential stuff


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
_subfolder ="/output/Simulation_Benchmark/"
_savedtraces = [
    "IBIS - StochasticVolatility Errors - Copula-BB1, Marginals-Tuple{Normal{Float64}, Normal{Float64}}, Reflection-Reflection270() - Trace.jld2",
    "IBIS - StochasticVolatility Errors - Copula-BB7, Marginals-Tuple{Normal{Float64}, Normal{Float64}}, Reflection-Reflection270() - Trace.jld2",

    "IBIS - StochasticVolatility Errors - Copula-Clayton, Marginals-Tuple{Normal{Float64}, Normal{Float64}}, Reflection-Reflection270() - Trace.jld2",
    "IBIS - StochasticVolatility Errors - Copula-FrankUnconstrained, Marginals-Tuple{Normal{Float64}, Normal{Float64}} - Trace.jld2",
    "IBIS - StochasticVolatility Errors - Copula-Gaussian, Marginals-Tuple{Normal{Float64}, Normal{Float64}} - Trace.jld2",
    "IBIS - StochasticVolatility Errors - Copula-Gumbel, Marginals-Tuple{Normal{Float64}, Normal{Float64}}, Reflection-Reflection270() - Trace.jld2",
    "IBIS - StochasticVolatility Errors - Copula-Joe, Marginals-Tuple{Normal{Float64}, Normal{Float64}}, Reflection-Reflection270() - Trace.jld2",
    "IBIS - StochasticVolatility Errors - Copula-TCop, Marginals-Tuple{Normal{Float64}, Normal{Float64}} - Trace.jld2",
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
    8
)
ylabel!("Cumulative Log Predictive Likelihood", subplot=1)
ylabel!(string("CLPBF of ", _copulas_names[8]), subplot=2)

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
