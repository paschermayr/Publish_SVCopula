################################################################################
import Pkg
Pcd(@__DIR__)
kg.activate(".")
Pkg.status()
include("src/_packages.jl");
Pkg.instantiate()

################################################################################
# Set Preamble
include("src/_preamble.jl");

################################################################################
# Define setting
_modelname = Gaussian() #Gaussian() #TCop() #Clayton() #Frank() #Joe() #Gumbel()  #FactorCopula
_marginalname = VolatilityMarginal() #StockMarginal()
_realdata = true

#################################################################################
# Set Model
include("src/_models.jl");
model = model_marginal

################################################################################
length(model)
keys(model.val)
tagged = Tagged(model, (:μᵥ, :κ, :σ, :ℓν) )
keys(tagged.parameter)

################################################################################
# Set Algorithm
_mcmc = NUTS(keys(tagged.parameter);
    GradientBackend = :ForwardDiff,
    init = PriorInitialization(1000) #NoInitialization()
)

################################################################################
# Set Data
if _realdata
    if _marginalname isa StockMarginal
        data = getindex.(data_real, 1)
    elseif _marginalname isa VolatilityMarginal
        data = getindex.(data_real, 2)
    end
elseif !_realdata
else
    println("Check settings for data and modelname.")
end

################################################################################
# Create initial model and sample from prior
model_initial = deepcopy(model)
Objective(model, data, tagged)(model.val)
_modelinit = ModelWrappers.PriorInitialization(10^5)
_modelinit(_rng, "-", Objective(model_initial, data, tagged))

################################################################################
################################################################################
################################################################################
# MCMC - Sample Stochastic Volatility model
trace_mcmc, algorithm_mcmc = sample(_rng, model_initial, data, _mcmc;
    default = SampleDefault(;
#        report = ProgressReport(; bar = true, log = ConsoleLog()),
        iterations = _N1, chains = mcmcchains, burnin = 1000,
        printoutput = true, safeoutput = false
    )
)
Baytes.savetrace(trace_mcmc, model, algorithm_mcmc, string("MCMC - ", _marginalname," - Trace"))
####################################
# Basic Plots and output
plotChain(trace_mcmc, tagged; burnin = 1000) #, chains=[1,3, 4])
Plots.savefig( string("MCMC - ", _marginalname," - Chain.png") )

transform_mcmc = Baytes.TraceTransform(trace_mcmc, model, tagged,
    TransformInfo(collect(1:trace_mcmc.summary.info.Nchains), [trace_mcmc.summary.info.Nalgorithms], (_burnin+1):1:trace_mcmc.summary.info.iterations)
)
summary(trace_mcmc, algorithm_mcmc, transform_mcmc, PrintDefault())

postmean_vec, postmean_tup = trace_to_posteriormean(trace_mcmc, transform_mcmc)
model_new = deepcopy(model)
fill!(model_new, postmean_tup)
model_new.val

postmean_tup_rounded = merge(postmean_tup, (; ℓν = round(postmean_tup.ℓν)) )
exp(postmean_tup_rounded.ℓν)+2
model_new_rounded = deepcopy(model)
fill!(model_new_rounded, postmean_tup_rounded)
model_new_rounded.val

if _marginalname isa StockMarginal
    errors = Data_to_Error_StockMarginal(model_new.val, data)
elseif _marginalname isa VolatilityMarginal
    errors = Data_to_Error_VolatilityMarginal(model_new.val, data)
end
histogram(errors, label=false, xlabel=string("MCMC - ", _marginalname," - histogram of T(df=postmean) Marginal errors.png"))
Plots.savefig( string("MCMC - ", _marginalname," - histogram of T(df=postmean) Marginal errors.png") )

#Make uniform errors based on df of marginal
errorsᵤ = cdf.(TDist(exp(model_new.val.ℓν)+2), errors)
errorsᵤ_rounded = cdf.(TDist(exp(postmean_tup_rounded.ℓν)+2), errors)
errorsᵤ_normal = cdf.(Normal(), errors)

histogram(errorsᵤ, label=false, xlabel=string("MCMC - ", _marginalname," - histogram of T(df=postmean) Marginal errors in uniform dim.png"))
Plots.savefig( string("MCMC - ", _marginalname," - histogram of T(df=postmean) Marginal errors in uniform dim.png") )
histogram(errorsᵤ_normal, label=false, xlabel=string("MCMC - ", _marginalname," - histogram of Normal Marginal errors in uniform dim.png"))
Plots.savefig( string("MCMC - ", _marginalname," - histogram of Normal Marginal errors in uniform dim.png") )

#Simulate data with posterior samples
_Nsimulations = 5
p = plot(
    layout = (_Nsimulations,1),
    size(1000, 1000),
    foreground_color_legend = :transparent,
    background_color_legend = :transparent,
    label=false
)
for iter in Base.OneTo(_Nsimulations)
    data_sim, _ = ModelWrappers.simulate(_rng, model_new, n)
    plot!(getindex.(data_sim, 1), color="black", subplot=iter,  label=false)
end
plot!(xlabel="Log(S&P500)", subplot=5)
p
Plots.savefig( string("MCMC - ", _marginalname," - Simulated data.png") )

# Save errors as JLD2
JLD2.jldsave(
    string("MCMC - ", _marginalname," - Errordata.jld2");
    # Posterior mean as tuple
    postmean_tup = postmean_tup,
    # Error data
    errors=errors,
    # Error data in uniform dimension with given Marginal with posterior mean
    errorsᵤ=errorsᵤ,
    # Error data in uniform dimension with given Marginal with nu nearest integer from posterior mean
    errorsᵤ_rounded=errorsᵤ_rounded,
    # Error data with standard Normal Marginal
    errorsᵤ_normal = errorsᵤ_normal
)

############################################################################################################
############################################################################################################
############################################################################################################
# use Posterior mean model
model_new_nucheck = deepcopy(model_new)
model_new_nucheck.val

#set possible range for
ℓνᵥ = collect(model_new_nucheck.info.constraint.ℓν.lower:model_new_nucheck.info.constraint.ℓν.upper)

#Fill model with range and compute log posterior
logpostᵥ = Float64[]
for ℓν_current in ℓνᵥ
    fill!(model_new_nucheck, (; ℓν = ℓν_current) )
    logpost = Objective(model_new_nucheck, data, tagged)(model_new_nucheck.val)
    push!(logpostᵥ, logpost)
end
p1 = plot(ℓνᵥ, logpostᵥ, xlabel = "log(ν - 2)", ylabel = "Logposterior given ν", label=false)
p2 = plot(ℓνᵥ[10:20], logpostᵥ[10:20], xlabel = "log(ν - 2)", ylabel = "Logposterior given ν", label=false)
plot(p1, p2, layout=(2,1))
Plots.savefig( string("MCMC - ", _marginalname," - Logposterior given nu.png") )

############################################################################################################
############################################################################################################
############################################################################################################
# Save both errors as text file for Aris
import Pkg
cd(@__DIR__)
Pkg.activate(".")
Pkg.status()

using JLD2, Plots, DelimitedFiles

############################################################################################################
# Load data
s_file = jldopen(string(pwd(), "\\_marginals\\MCMC - StockMarginal() - Errordata.jld2"))
x_file = jldopen(string(pwd(), "\\_marginals\\MCMC - VolatilityMarginal() - Errordata.jld2"))

s_postmean_tup = read(s_file, "postmean_tup")
s_errors = read(s_file, "errors")
s_errorsᵤ = read(s_file, "errorsᵤ")
s_errorsᵤ_rounded = read(s_file, "errorsᵤ_rounded")
s_errorsᵤ_normal = read(s_file, "errorsᵤ_normal")
exp(s_postmean_tup.ℓν)+2


x_postmean_tup = read(x_file, "postmean_tup")
x_errors = read(x_file, "errors")
x_errorsᵤ = read(x_file, "errorsᵤ")
x_errorsᵤ_rounded = read(x_file, "errorsᵤ_rounded")
x_errorsᵤ_normal = read(x_file, "errorsᵤ_normal")
exp(x_postmean_tup.ℓν)+2

############################################################################################################
# Save data as text

# Errors
errors_output = hcat(s_errors, x_errors)
errors_header = "s_errors x_errors\n"
open("errors.txt"; write=true) do f
         write(f, errors_header)
         writedlm(f, errors_output)
end

#Errors_uniform - T marginal posterior
errors_output = hcat(s_errorsᵤ_rounded, x_errorsᵤ_rounded)
errors_header = "s_errors_uniform x_errors_uniform\n"
open("errors_uniform_T.txt"; write=true) do f
         write(f, errors_header)
         writedlm(f, errors_output)
end

#Errors_uniform - Standard Normal marginal
errors_output = hcat(s_errorsᵤ_normal, x_errorsᵤ_normal)
errors_header = "s_errors_uniform x_errors_uniform\n"
open("errors_uniform_Normal.txt"; write=true) do f
         write(f, errors_header)
         writedlm(f, errors_output)
end
