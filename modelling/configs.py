configs = {
    "target_name": "AveragePrice",
    "lags": [4, 8, 13, 26, 52],
    "target_regions": [
        # "Albany", 
        "Atlanta", 
        # "BaltimoreWashington", "Boise", "Boston",
        # "BuffaloRochester", "California", "Charlotte", "Chicago",
        # "CincinnatiDayton", "Columbus", "DallasFtWorth", "Denver",
        # "Detroit", "GrandRapids", "GreatLakes", "HarrisburgScranton",
        # "HartfordSpringfield", "Houston", "Indianapolis", "Jacksonville",
        # "LasVegas", "LosAngeles", "Louisville", "MiamiFtLauderdale",
        # "Midsouth", "Nashville", "NewOrleansMobile", "NewYork",
        # "Northeast", "NorthernNewEngland", "Orlando", "Philadelphia",
        # "PhoenixTucson", "Pittsburgh", "Plains", "Portland",
        # "RaleighGreensboro", "RichmondNorfolk", "Roanoke", "Sacramento",
        # "SanDiego", "SanFrancisco", "Seattle", "SouthCarolina",
        # "SouthCentral", "Southeast", "Spokane", "StLouis", "Syracuse",
        # "Tampa", "TotalUS", "West", 
        # "WestTexNewMexico"
    ],
    "aux_regions": [
        "TotalUS", "West", "Midsouth", "Northeast", "Southeast", "SouthCentral"
    ],
    "aux_features": [
        "AveragePrice_combined", "TotalVolume_combined", 
        "4046_combined", "4225_combined", "4770_combined", 
        "TotalBags_combined", "SmallBags_combined", 
        "LargeBags_combined", "XLargeBags_combined"
    ],
    "aux_lags": [4, 8, 13, 26, 52]
}