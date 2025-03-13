# 1 - Sem aux, lag = 4, basic timefeats
# 2 - Sem aux, lag = [4, 8, 13, 26, 52], basic timefeats
# 3 - Sem aux, lag = [4, 5, 6, 7, 8], basic timefeats
# 4 - Sem aux, lag = [4, 52], basic timefeats
# 5 - Sem aux, lag = [4, 52], ciclycal time
# 6 - Sem aux, lag = [4], ciclycal time
# 7 - Sem aux, lag = [4], ciclycal time + index
# 8 - total_us, "aux_lags": [4, 8, 13, 26, 52], lag = [4], ciclycal time + index
# 9 - total_us, "aux_lags": [4], lag = [4], ciclycal time + index
# 10 - Sem aux, lag = 4, basic timefeats + index
# 11 - Sem aux, lag = 4, basic timefeats + rolling
# 11 - Sem aux, lag = 4, basic timefeats + rolling
# 12 - Sem aux, lag = [4], ciclycal time + index + rolling
# 13 - Sem aux, lag = [4], ciclycal time + index + rolling12


# + lags antigos piorou
#  Lags recentes pior que antigos
configs = {
    "target_name": "AveragePrice",
    "lags": [4],
    "target_regions": [
        "Albany", 
        "Atlanta", 
        # "BaltimoreWashington", 
        "Boise", 
        # "Boston",
        # "BuffaloRochester", "California", "Charlotte", "Chicago",
        # "CincinnatiDayton", "Columbus", "DallasFtWorth", "Denver",
        # "Detroit", "GrandRapids", "GreatLakes", "HarrisburgScranton",
        # "HartfordSpringfield", "Houston", "Indianapolis", "Jacksonville",
        # "LasVegas", "LosAngeles", "Louisville", "MiamiFtLauderdale",
        # "Midsouth", "Nashville", "NewOrleansMobile", "NewYork",
        # "Northeast", "NorthernNewEngland", "Orlando", "Philadelphia",
        # "PhoenixTucson", "Pittsburgh", "Plains", "Portland",
        # "RaleighGreensboro", "RichmondNorfolk", "Roanoke", "Sacramento",
        # "SanDiego", 
        # "SanFrancisco",
        #  "Seattle", "SouthCarolina",
        # "SouthCentral", "Southeast", "Spokane", "StLouis", "Syracuse",
        # "Tampa", 
        # "TotalUS", 
        # "West", 
        # "WestTexNewMexico"
    ],
    'rolling_window_sizes':[12],
    "aux_regions": [
        "TotalUS", 
    #     # "West", "Midsouth", "Northeast", "Southeast", "SouthCentral"
        ],
    "aux_features": [
        "AveragePrice_combined", 
        # "TotalVolume_combined", 
        # "4046_combined", "4225_combined", "4770_combined", 
        # "TotalBags_combined", "SmallBags_combined", 
        # "LargeBags_combined", "XLargeBags_combined"
    ],
    "aux_lags": [4]
}