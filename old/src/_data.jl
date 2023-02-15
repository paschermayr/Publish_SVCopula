################################################################################
#Load prepared data in real domain
data_raw = jldopen( string(pwd(), "/src/data/data.jld2" ) )
data_dates = read(data_raw, "dates")
data_sp500 = read(data_raw, "sp500")
data_vix = read(data_raw, "vix")

#Transform as discussed
_timeperiod = (length(data_dates)-n-data_delay+1):(length(data_dates)-data_delay)
_multiplier = 1 #sqrt(10)
data_real = [
    (S = log(data_sp500[iter]),
    X = log( _multiplier * (data_vix[iter] / 100 )^2 ),
    V = _multiplier * (data_vix[iter] / 100 )^2 ) for iter in _timeperiod
]
dates_real = data_dates[_timeperiod]
