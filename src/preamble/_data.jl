################################################################################
#Load prepared data in real domain
data_raw = jldopen( string(pwd(), "/data/data.jld2" ) )
data_dates = read(data_raw, "dates")
data_sp500 = read(data_raw, "sp500")
data_vix = read(data_raw, "vix")

#Transform as discussed
_timeperiod = (length(data_dates)-n-data_delay+1):(length(data_dates)-data_delay)
_multiplier = 1 #sqrt(10)
data_real = [
    (S = log(data_sp500[iter]),
    X = log( _multiplier * (data_vix[iter] / 100 )^2 ), #-log( _multiplier * (data_vix[iter] / 100 )^2 ), #log( _multiplier * (data_vix[iter] / 100 )^2 )
    V = _multiplier * (data_vix[iter] / 100 )^2 #exp( - log( _multiplier * (data_vix[iter] / 100 )^2 ) ) #_multiplier * (data_vix[iter] / 100 )^2
    ) for iter in _timeperiod
]
dates_real = data_dates[_timeperiod]

#=
S = getindex.( data_real, 1)
X = getindex.(data_real, 2)
V = getindex.(data_real, 3)
p = plot(layout=(3,1), label=false)
S_diff = [(S[iter] - S[iter-1])^2 for iter in 2:length(S)]
plot!(dates_real[2:end], S_diff, xlabel = string("Mean: ", mean(S_diff) ), ylabel="(Sₜ - Sₜ₋₁)²", label=false, subplot=1)
plot!(dates_real, model.val.δ .* (V), xlabel = string("Mean: ", mean(model.val.δ .* (V.^2) ) ), ylabel="δ * V", label=false, subplot=2)
plot!(dates_real, model.val.δ .* (X.^2), xlabel = string("Mean: ", mean(model.val.δ .* (X.^2) ) ), ylabel="δ * X²", label=false, subplot=3)

p = plot(layout=(2,1))
plot!(dates_real[2:end], S_diff, xlabel = string("Mean: ", mean(S_diff) ), ylabel="(Sₜ - Sₜ₋₁)²", label=false, subplot=1)
plot!(dates_real, model.val.δ .* exp.(X), xlabel = string("Mean: ", mean(model.val.δ .* exp.(X) ) ), ylabel="δ * exp X", label=false, subplot=2)
=#
