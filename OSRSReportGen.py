## Originally published at https://poignanttech.com/projects/
## the primary function of this script is to rapidly generate reports leveraging preprepared market statistics in-addition to data provided by the /latest market data endpoint of the OSRS Wiki API (https://oldschool.runescape.wiki/w/RuneScape:Real-time_Prices)
## This script must be executed in the same directory as the SQLite database 'osrsmarketdata.sqlite', which is generated by the OSRSBuildDatabase.py companion script.
## report results are saved within a SQLite database 'outputdb.sqlite' in the same directory where the script is executed. From here, it can be published using an API or other means for consumption.
## This script has been tested on Debian GNU/Linux 12 with Python 3.11.2 and is scheduled using run-one/cron to execute once every minute, which corresponds with the /latest endpoint refresh interval of the OSRS Wiki API

## Before using this script, you must uncomment the user-agent line below and enter contact information within the string. See the above URL for more information.
headers = {'User-Agent': 'Data collection for ML training - gianfrancoameri2002@gmail.com',}

## import modules
import pandas as pd
from functools import reduce
import requests
import json
import time
from io import StringIO
import sqlite3
import csv
import glob
from glob import glob; from os.path import expanduser
import os
import io

## change working directory to script location
pathtorunfile = os.path.dirname(__file__)
os.chdir(pathtorunfile)

## this function returns a generic response if a report returns no valid results
def nullresult():
	ResultData = pd.DataFrame([['No valid result at this time. Please check again in a few minutes.']], columns=['Result'])
	return(ResultData)

### runtime
if __name__ == '__main__':

	## initialize temp in-memory sqlite connection and attach main database to read preprepared market statistics
	tempdb = sqlite3.connect(":memory:")
	tempdb.execute('pragma journal_mode=wal')
	cur = tempdb.cursor()
	cur.execute(f"ATTACH DATABASE 'osrsmarketdata.sqlite' AS marketdatadb;")
	
	## define and connect to output database to publish results
	outputdb = sqlite3.connect("outputdb.sqlite")
	tempdb.execute('pragma journal_mode=wal')

	## fetch latest market data from osrs wiki api
	url = 'https://prices.runescape.wiki/api/v1/osrs/latest'
	r = requests.get(url, headers=headers)
	latestdata = json.dumps(r.json())
	# caveman code for weird json
	latestcomplete = latestdata.replace('"', '').replace('{data:{', 'data').replace('},', '\n').replace('}', '').replace('high', '').replace('low', '').replace('Time', '').replace(':', '').replace('{', ',').replace(' ', '')
	latestcomplete = latestcomplete.replace(',data,', 'id,high,hightime,low,lowtime\n')
	# save to temp db
	dflatest = pd.read_table(StringIO(latestcomplete), sep=",")
	dflatest.to_sql('latest', tempdb, if_exists='replace')	
	
	# This large block flattens-out our data, with resulting rows being comprised of unique item types.
	cur.execute("CREATE TABLE MasterTable(id TEXT, mappinglimit INT, mappingname TEXT, mappinghighalch INT, high INT, low INT, WeeklyMeanLow INT, WeeklyMeanHigh INT, WeeklyMeanVolumeLow TEXT, WeeklyMeanVolumeHigh TEXT, WeeklyMedianLow INT, WeeklyMedianHigh INT, WeeklyMedianVolumeLow INT, WeeklyMedianVolumeHigh INT, WeeklyMinLow INT, WeeklyMinHigh INT, WeeklyMaxLow INT, WeeklyMaxHigh INT, MonthlyMeanLow INT, MonthlyMeanHigh INT, MonthlyMeanVolumeLow TEXT, MonthlyMeanVolumeHigh TEXT, MonthlyMedianLow INT, MonthlyMedianHigh INT, MonthlyMedianVolumeLow INT, MonthlyMedianVolumeHigh INT, MonthlyMinLow INT, MonthlyMinHigh INT, MonthlyMaxLow INT, MonthlyMaxHigh INT, YearlyMeanLow INT, YearlyMeanHigh INT, YearlyMeanVolumeLow TEXT, YearlyMeanVolumeHigh TEXT, YearlyMedianLow INT, YearlyMedianHigh INT, YearlyMedianVolumeLow INT, YearlyMedianVolumeHigh INT, YearlyMinLow INT, YearlyMinHigh INT, YearlyMaxLow INT, YearlyMaxHigh INT, GranularDailyMeanLow INT, GranularDailyMeanHigh INT, GranularDailyMeanVolumeLow TEXT, GranularDailyMeanVolumeHigh TEXT, GranularDailyMedianLow INT, GranularDailyMedianHigh INT, GranularDailyMedianVolumeLow INT, GranularDailyMedianVolumeHigh INT, GranularDailyMinLow INT, GranularDailyMinHigh INT, GranularDailyMaxLow INT, GranularDailyMaxHigh INT, GranularBiweeklyMeanLow INT, GranularBiweeklyMeanHigh INT, GranularBiweeklyMeanVolumeLow TEXT, GranularBiweeklyMeanVolumeHigh TEXT, GranularBiweeklyMedianLow INT, GranularBiweeklyMedianHigh INT, GranularBiweeklyMedianVolumeLow INT, GranularBiweeklyMedianVolumeHigh INT, GranularBiweeklyMinLow INT, GranularBiweeklyMinHigh INT, GranularBiweeklyMaxLow INT, GranularBiweeklyMaxHigh INT, GranularMonthlyMeanLow INT, GranularMonthlyMeanHigh INT, GranularMonthlyMeanVolumeLow TEXT, GranularMonthlyMeanVolumeHigh TEXT, GranularMonthlyMedianLow INT, GranularMonthlyMedianHigh INT, GranularMonthlyMedianVolumeLow INT, GranularMonthlyMedianVolumeHigh INT, GranularMonthlyMinLow INT, GranularMonthlyMinHigh INT, GranularMonthlyMaxLow INT, GranularMonthlyMaxHigh INT, VeryGranularFiveMinuteMeanLow INT, VeryGranularFiveMinuteMeanHigh INT, VeryGranularFiveMinuteMeanVolumeLow TEXT, VeryGranularFiveMinuteMeanVolumeHigh TEXT, VeryGranularFiveMinuteMedianLow INT, VeryGranularFiveMinuteMedianHigh INT, VeryGranularFiveMinuteMedianVolumeLow INT, VeryGranularFiveMinuteMedianVolumeHigh INT, VeryGranularFiveMinuteMinLow INT, VeryGranularFiveMinuteMinHigh INT, VeryGranularFiveMinuteMaxLow INT, VeryGranularFiveMinuteMaxHigh INT, VeryGranularHourlyMeanLow INT, VeryGranularHourlyMeanHigh INT, VeryGranularHourlyMeanVolumeLow TEXT, VeryGranularHourlyMeanVolumeHigh TEXT, VeryGranularHourlyMedianLow INT, VeryGranularHourlyMedianHigh INT, VeryGranularHourlyMedianVolumeLow INT, VeryGranularHourlyMedianVolumeHigh INT, VeryGranularHourlyMinLow INT, VeryGranularHourlyMinHigh INT, VeryGranularHourlyMaxLow INT, VeryGranularHourlyMaxHigh INT, VeryGranularDailyMeanLow INT, VeryGranularDailyMeanHigh INT, VeryGranularDailyMeanVolumeLow TEXT, VeryGranularDailyMeanVolumeHigh TEXT, VeryGranularDailyMedianLow INT, VeryGranularDailyMedianHigh INT, VeryGranularDailyMedianVolumeLow INT, VeryGranularDailyMedianVolumeHigh INT, VeryGranularDailyMinLow INT, VeryGranularDailyMinHigh INT, VeryGranularDailyMaxLow INT, VeryGranularDailyMaxHigh INT, ProductName TEXT, RecipeType TEXT, QtyProduced INT, ProcessingCost INT, ingredient1id INT, ingredient1Qty TEXT, ingredient2id INT, ingredient2Qty TEXT, ingredient3id INT, ingredient3Qty TEXT);")
	cur.execute("INSERT INTO MasterTable(id, mappinglimit, mappingname, mappinghighalch, high, low) SELECT typeid, buylimit, name, highalch, high, low FROM Mapping RIGHT JOIN latest ON Mapping.typeid = latest.id;")
	cur.execute("CREATE TABLE BlackMarketRate AS SELECT exchangerate AS BlackMarketRate FROM BlackMarket;")
	cur.execute("UPDATE MasterTable SET WeeklyMeanLow = MeanLow, WeeklyMeanHigh = MeanHigh, WeeklyMeanVolumeLow = MeanVolumeLow, WeeklyMeanVolumeHigh = MeanVolumeHigh, WeeklyMedianLow = MedianLow, WeeklyMedianHigh = MedianHigh, WeeklyMedianVolumeLow = MedianVolumeLow, WeeklyMedianVolumeHigh = MedianVolumeHigh, WeeklyMinLow = MinLow, WeeklyMinHigh = MinHigh, WeeklyMaxLow = MaxLow, WeeklyMaxHigh = MaxHigh FROM marketstats WHERE Type = 'Weekly' AND id = typeid;")
	cur.execute("UPDATE MasterTable SET MonthlyMeanLow = MeanLow, MonthlyMeanHigh = MeanHigh, MonthlyMeanVolumeLow = MeanVolumeLow, MonthlyMeanVolumeHigh = MeanVolumeHigh, MonthlyMedianLow = MedianLow, MonthlyMedianHigh = MedianHigh, MonthlyMedianVolumeLow = MedianVolumeLow, MonthlyMedianVolumeHigh = MedianVolumeHigh, MonthlyMinLow = MinLow, MonthlyMinHigh = MinHigh, MonthlyMaxLow = MaxLow, MonthlyMaxHigh = MaxHigh FROM marketstats WHERE Type = 'Monthly' AND id = typeid;")
	cur.execute("UPDATE MasterTable SET YearlyMeanLow = MeanLow, YearlyMeanHigh = MeanHigh, YearlyMeanVolumeLow = MeanVolumeLow, YearlyMeanVolumeHigh = MeanVolumeHigh, YearlyMedianLow = MedianLow, YearlyMedianHigh = MedianHigh, YearlyMedianVolumeLow = MedianVolumeLow, YearlyMedianVolumeHigh = MedianVolumeHigh, YearlyMinLow = MinLow, YearlyMinHigh = MinHigh, YearlyMaxLow = MaxLow, YearlyMaxHigh = MaxHigh FROM marketstats WHERE Type = 'Yearly' AND id = typeid;")	
	cur.execute("UPDATE MasterTable SET GranularDailyMeanLow = MeanLow, GranularDailyMeanHigh = MeanHigh, GranularDailyMeanVolumeLow = MeanVolumeLow, GranularDailyMeanVolumeHigh = MeanVolumeHigh, GranularDailyMedianLow = MedianLow, GranularDailyMedianHigh = MedianHigh, GranularDailyMedianVolumeLow = MedianVolumeLow, GranularDailyMedianVolumeHigh = MedianVolumeHigh, GranularDailyMinLow = MinLow, GranularDailyMinHigh = MinHigh, GranularDailyMaxLow = MaxLow, GranularDailyMaxHigh = MaxHigh FROM marketstats WHERE Type = 'GranularDaily' AND id = typeid;")
	cur.execute("UPDATE MasterTable SET GranularBiweeklyMeanLow = MeanLow, GranularBiweeklyMeanHigh = MeanHigh, GranularBiweeklyMeanVolumeLow = MeanVolumeLow, GranularBiweeklyMeanVolumeHigh = MeanVolumeHigh, GranularBiweeklyMedianLow = MedianLow, GranularBiweeklyMedianHigh = MedianHigh, GranularBiweeklyMedianVolumeLow = MedianVolumeLow, GranularBiweeklyMedianVolumeHigh = MedianVolumeHigh, GranularBiweeklyMinLow = MinLow, GranularBiweeklyMinHigh = MinHigh, GranularBiweeklyMaxLow = MaxLow, GranularBiweeklyMaxHigh = MaxHigh FROM marketstats WHERE Type = 'GranularBiweekly' AND id = typeid;")
	cur.execute("UPDATE MasterTable SET GranularMonthlyMeanLow = MeanLow, GranularMonthlyMeanHigh = MeanHigh, GranularMonthlyMeanVolumeLow = MeanVolumeLow, GranularMonthlyMeanVolumeHigh = MeanVolumeHigh, GranularMonthlyMedianLow = MedianLow, GranularMonthlyMedianHigh = MedianHigh, GranularMonthlyMedianVolumeLow = MedianVolumeLow, GranularMonthlyMedianVolumeHigh = MedianVolumeHigh, GranularMonthlyMinLow = MinLow, GranularMonthlyMinHigh = MinHigh, GranularMonthlyMaxLow = MaxLow, GranularMonthlyMaxHigh = MaxHigh FROM marketstats WHERE Type = 'GranularMonthly' AND id = typeid;")
	cur.execute("UPDATE MasterTable SET VeryGranularFiveMinuteMeanLow = MeanLow, VeryGranularFiveMinuteMeanHigh = MeanHigh, VeryGranularFiveMinuteMeanVolumeLow = MeanVolumeLow, VeryGranularFiveMinuteMeanVolumeHigh = MeanVolumeHigh, VeryGranularFiveMinuteMedianLow = MedianLow, VeryGranularFiveMinuteMedianHigh = MedianHigh, VeryGranularFiveMinuteMedianVolumeLow = MedianVolumeLow, VeryGranularFiveMinuteMedianVolumeHigh = MedianVolumeHigh, VeryGranularFiveMinuteMinLow = MinLow, VeryGranularFiveMinuteMinHigh = MinHigh, VeryGranularFiveMinuteMaxLow = MaxLow, VeryGranularFiveMinuteMaxHigh = MaxHigh FROM marketstats WHERE Type = 'VeryGranularFiveMinute' AND id = typeid;")
	cur.execute("UPDATE MasterTable SET VeryGranularHourlyMeanLow = MeanLow, VeryGranularHourlyMeanHigh = MeanHigh, VeryGranularHourlyMeanVolumeLow = MeanVolumeLow, VeryGranularHourlyMeanVolumeHigh = MeanVolumeHigh, VeryGranularHourlyMedianLow = MedianLow, VeryGranularHourlyMedianHigh = MedianHigh, VeryGranularHourlyMedianVolumeLow = MedianVolumeLow, VeryGranularHourlyMedianVolumeHigh = MedianVolumeHigh, VeryGranularHourlyMinLow = MinLow, VeryGranularHourlyMinHigh = MinHigh, VeryGranularHourlyMaxLow = MaxLow, VeryGranularHourlyMaxHigh = MaxHigh FROM marketstats WHERE Type = 'VeryGranularHourly' AND id = typeid;")
	cur.execute("UPDATE MasterTable SET VeryGranularDailyMeanLow = MeanLow, VeryGranularDailyMeanHigh = MeanHigh, VeryGranularDailyMeanVolumeLow = MeanVolumeLow, VeryGranularDailyMeanVolumeHigh = MeanVolumeHigh, VeryGranularDailyMedianLow = MedianLow, VeryGranularDailyMedianHigh = MedianHigh, VeryGranularDailyMedianVolumeLow = MedianVolumeLow, VeryGranularDailyMedianVolumeHigh = MedianVolumeHigh, VeryGranularDailyMinLow = MinLow, VeryGranularDailyMinHigh = MinHigh, VeryGranularDailyMaxLow = MaxLow, VeryGranularDailyMaxHigh = MaxHigh FROM marketstats WHERE Type = 'VeryGranularDaily' AND id = typeid;")
	cur.execute("UPDATE MasterTable SET ProductName = Recipes.ProductName, RecipeType = Recipes.RecipeType, QtyProduced = Recipes.QtyProduced, ProcessingCost = Recipes.ProcessingCost, ingredient1id = Recipes.ingredient1id, ingredient1Qty = Recipes.ingredient1Qty, ingredient1Qty = Recipes.ingredient1Qty, ingredient2id = Recipes.ingredient2id, ingredient2Qty = Recipes.ingredient2Qty, ingredient3id = Recipes.ingredient3id, ingredient3Qty = Recipes.ingredient3Qty FROM Recipes WHERE MasterTable.id = Recipes.id;")

	### Generate and export reports
	## Report #1 - Dip Detection
	cur.execute('''CREATE TABLE MasterTableTax AS SELECT * FROM MasterTable;''')
	cur.execute('''ALTER TABLE MasterTableTax ADD COLUMN Tax;''')
	cur.execute('''CREATE TABLE MaxTax AS SELECT * FROM MasterTableTax WHERE round(VeryGranularDailyMeanLow) > 500000000;''')
	cur.execute('''CREATE TABLE MinTax AS SELECT * FROM MasterTableTax WHERE round(VeryGranularDailyMeanLow) <= 500000000;''')
	cur.execute('''UPDATE MaxTax SET Tax = 5000000;''')
	cur.execute('''UPDATE MinTax SET Tax = round((VeryGranularDailyMeanLow * 0.01) - 0.5);''')
	cur.execute('''CREATE TABLE DailyCSVwithTax AS SELECT * FROM MaxTax UNION SELECT * FROM MinTax;''')
	cur.execute('''CREATE TABLE NoBuyLimit AS SELECT *, ((VeryGranularDailyMeanLow - low - Tax) * 24 * MIN(GranularDailyMeanVolumeLow, GranularDailyMeanVolumeHigh)) AS NoBuyLimitProfit FROM DailyCSVwithTax;''')
	cur.execute('''CREATE TABLE WithBuyLimit AS SELECT id, ((VeryGranularDailyMeanLow - low - Tax) * mappinglimit) AS WithBuyLimitProfit FROM DailyCSVwithTax;''')
	cur.execute('''CREATE TABLE DailyCSVwithProfit AS SELECT *, MIN(NoBuyLimit.NoBuyLimitProfit, COALESCE(WithBuyLimit.WithBuyLimitProfit, 'NONE')) AS AdjustedPotentialDailyProfit FROM NoBuyLimit, WithBuyLimit WHERE NoBuyLimit.id = WithBuyLimit.id;''')
	cur.execute('''CREATE TABLE FinalOutput AS SELECT mappingname AS ItemName, low AS LowPrice, VeryGranularDailyMeanLow AS AvgLow, mappinglimit AS BuyLimit, AdjustedPotentialDailyProfit, (VeryGranularDailyMeanLow - low - Tax) AS ProfitPerUnit, ((VeryGranularDailyMeanLow - low - Tax) / low) * 100 AS pctROI FROM DailyCSVwithProfit WHERE (VeryGranularHourlyMeanLow > (low * 1.02)) AND (GranularBiweeklyMinHigh + GranularBiweeklyMinLow) / 2 > low AND (MonthlyMinLow + MonthlyMinHigh) / 2 > low AND pctROI > 0 AND AdjustedPotentialDailyProfit > 100000 AND MonthlyMaxHigh > MonthlyMaxLow AND (high - low - tax) > 0 AND MonthlyMedianVolumeHigh > 0 AND MonthlyMedianVolumeLow > 0 AND GranularDailyMedianVolumeHigh > 0 AND GranularDailyMedianVolumeLow > 0 ORDER BY AdjustedPotentialDailyProfit DESC;''')

	# export DipDetectReport to outputdb sqlite DB and cleanup
	dfrows = pd.read_sql('SELECT ItemName, LowPrice, AvgLow, BuyLimit, printf("%.2f", pctROI) AS pctROI FROM FinalOutput', tempdb)
	if dfrows.empty:
		dfrows = nullresult()
	dfrows.to_sql('DipDetectReport', outputdb, if_exists='replace', index=False)
	cur.executescript('''DROP TABLE IF EXISTS JagexExchangeRate; DROP TABLE IF EXISTS NatureRunePrice; DROP TABLE IF EXISTS PriceFloor; DROP TABLE IF EXISTS MasterTableTax; DROP TABLE IF EXISTS MaxTax; DROP TABLE IF EXISTS MinTax; DROP TABLE IF EXISTS DailyCSVwithTax; DROP TABLE IF EXISTS NoBuyLimit; DROP TABLE IF EXISTS WithBuyLimit; DROP TABLE IF EXISTS DailyCSVwithProfit; DROP TABLE IF EXISTS FinalOutput;''')

	## Report #2 - Alch Reporting
	cur.execute('''CREATE TABLE JagexExchangeRate AS SELECT ((WeeklyMeanLow + WeeklyMeanHigh) / 2) AS BondPrice, CAST((7.99 * 1000000) / ((WeeklyMeanLow + WeeklyMeanHigh) / 2) AS REAL) AS JagexExchangeRate FROM MasterTable WHERE id=13190;''')
	cur.execute('''CREATE TABLE NatureRunePrice AS SELECT (GranularDailyMeanLow + GranularDailyMeanHigh) / 2 AS NatureRunePrice FROM MasterTable WHERE id=561;''')
	cur.execute('''CREATE TABLE PriceFloor AS SELECT id, round(((MasterTable.mappinghighalch - NatureRunePrice.NatureRunePrice - (JagexExchangeRate.BondPrice / (403200 * (BlackMarketRate / JagexExchangeRate.JagexExchangeRate)))) * 0.99) + 0.5) AS PriceFloor FROM MasterTable, NatureRunePrice, JagexExchangeRate, BlackMarketRate;''')
	cur.execute('''CREATE TABLE MasterTableTax AS SELECT * FROM MasterTable INNER JOIN PriceFloor ON MasterTable.id = PriceFloor.id;''')
	cur.execute('''ALTER TABLE MasterTableTax ADD COLUMN Tax;''')
	cur.execute('''UPDATE MasterTableTax SET Tax = round((PriceFloor * 0.01) - 0.5);''')
	cur.execute('''CREATE TABLE NoBuyLimit AS SELECT *, ((PriceFloor - low - Tax) * 24 * MIN(GranularDailyMeanVolumeLow, GranularDailyMeanVolumeHigh)) AS NoBuyLimitProfit FROM MasterTableTax;''')
	cur.execute('''CREATE TABLE WithBuyLimit AS SELECT id, ((PriceFloor - low - Tax) * mappinglimit) AS WithBuyLimitProfit FROM MasterTableTax;''')
	cur.execute('''CREATE TABLE DailyCSVwithProfit AS SELECT *, MIN(NoBuyLimit.NoBuyLimitProfit, COALESCE(WithBuyLimit.WithBuyLimitProfit, 'NONE')) AS AdjustedPotentialDailyProfit FROM NoBuyLimit, WithBuyLimit WHERE NoBuyLimit.id = WithBuyLimit.id;''')
	cur.execute('''CREATE TABLE FinalOutput AS SELECT mappingname AS ItemName, low AS LowPrice, PriceFloor, mappinglimit AS BuyLimit, (PriceFloor - low - Tax) AS ProfitPerUnit, ((PriceFloor - low - Tax) / low) * 100 AS pctROI FROM DailyCSVwithProfit, JagexExchangeRate, BlackMarketRate WHERE mappinglimit > (4800 * (BlackMarketRate) / (JagexExchangeRate.JagexExchangeRate)) AND (GranularDailyMeanVolumeHigh + GranularDailyMeanVolumeLow) / 2 > 4800 * (BlackMarketRate) / (JagexExchangeRate.JagexExchangeRate) AND pctROI > 1 ORDER BY AdjustedPotentialDailyProfit DESC;''')

	# export AlchReport to outputdb sqlite DB and cleanup
	dfrows = pd.read_sql('''SELECT ItemName, LowPrice, round(PriceFloor) AS PriceFloor, BuyLimit, printf("%.2f", pctROI) AS pctROI FROM FinalOutput;''', tempdb)
	if dfrows.empty:
		dfrows = nullresult()
	dfrows.to_sql('AlchReport', outputdb, if_exists='replace', index=False)
	cur.executescript('''DROP TABLE IF EXISTS JagexExchangeRate; DROP TABLE IF EXISTS NatureRunePrice; DROP TABLE IF EXISTS PriceFloor; DROP TABLE IF EXISTS MasterTableTax; DROP TABLE IF EXISTS MaxTax; DROP TABLE IF EXISTS MinTax; DROP TABLE IF EXISTS DailyCSVwithTax; DROP TABLE IF EXISTS NoBuyLimit; DROP TABLE IF EXISTS WithBuyLimit; DROP TABLE IF EXISTS DailyCSVwithProfit; DROP TABLE IF EXISTS FinalOutput;''')

	## Report #3 - Crafting Reporting
	cur.execute('''CREATE TABLE LowEffortRecipeTable AS SELECT * FROM MasterTable WHERE ProductName IS NOT NULL;''')
	cur.executescript('''ALTER TABLE LowEffortRecipeTable ADD ingredient1lowprice; ALTER TABLE LowEffortRecipeTable ADD ingredient2lowprice; ALTER TABLE LowEffortRecipeTable ADD ingredient3lowprice;''')
	cur.executescript('''UPDATE LowEffortRecipeTable SET ingredient1lowprice = coalesce((SELECT MasterTable.low FROM MasterTable WHERE LowEffortRecipeTable.ingredient1id = MasterTable.id), 0); UPDATE LowEffortRecipeTable SET ingredient2lowprice = coalesce((SELECT MasterTable.low FROM MasterTable WHERE LowEffortRecipeTable.ingredient2id = MasterTable.id), 0); UPDATE LowEffortRecipeTable SET ingredient3lowprice = coalesce((SELECT MasterTable.low FROM MasterTable WHERE LowEffortRecipeTable.ingredient3id = MasterTable.id), 0);''')
	cur.executescript('''ALTER TABLE LowEffortRecipeTable ADD ingredient1buylimit; ALTER TABLE LowEffortRecipeTable ADD ingredient2buylimit; ALTER TABLE LowEffortRecipeTable ADD ingredient3buylimit;''')
	cur.executescript('''UPDATE LowEffortRecipeTable SET ingredient1buylimit = (SELECT MasterTable.mappinglimit FROM MasterTable WHERE LowEffortRecipeTable.ingredient1id = MasterTable.id); UPDATE LowEffortRecipeTable SET ingredient2buylimit = (SELECT MasterTable.mappinglimit FROM MasterTable WHERE LowEffortRecipeTable.ingredient2id = MasterTable.id); UPDATE LowEffortRecipeTable SET ingredient3buylimit = (SELECT MasterTable.mappinglimit FROM MasterTable WHERE LowEffortRecipeTable.ingredient3id = MasterTable.id);''')
	cur.executescript('''ALTER TABLE LowEffortRecipeTable ADD ingredient1hourlylowvolume; ALTER TABLE LowEffortRecipeTable ADD ingredient2hourlylowvolume; ALTER TABLE LowEffortRecipeTable ADD ingredient3hourlylowvolume;''')
	cur.executescript('''UPDATE LowEffortRecipeTable SET ingredient1hourlylowvolume = (SELECT MasterTable.GranularDailyMeanVolumeLow FROM MasterTable WHERE LowEffortRecipeTable.ingredient1id = MasterTable.id); UPDATE LowEffortRecipeTable SET ingredient2hourlylowvolume = (SELECT MasterTable.GranularDailyMeanVolumeLow FROM MasterTable WHERE LowEffortRecipeTable.ingredient2id = MasterTable.id); UPDATE LowEffortRecipeTable SET ingredient3hourlylowvolume = (SELECT MasterTable.GranularDailyMeanVolumeLow FROM MasterTable WHERE LowEffortRecipeTable.ingredient3id = MasterTable.id);''')
	cur.execute('''CREATE TABLE LowEffortRecipeTableTax AS SELECT * FROM LowEffortRecipeTable;''')
	cur.execute('''ALTER TABLE LowEffortRecipeTableTax ADD COLUMN Tax;''')
	cur.execute('''CREATE TABLE MaxTax AS SELECT * FROM LowEffortRecipeTableTax WHERE round(GranularDailyMeanHigh) > 500000000;''')
	cur.execute('''CREATE TABLE MinTax AS SELECT * FROM LowEffortRecipeTableTax WHERE round(GranularDailyMeanHigh) <= 500000000;''')
	cur.executescript('''DROP TABLE LowEffortRecipeTableTax; DROP TABLE LowEffortRecipeTable;''')
	cur.execute('''UPDATE MaxTax SET Tax = 5000000 + ProcessingCost;''')
	cur.execute('''UPDATE MinTax SET Tax = round((GranularDailyMeanHigh * 0.01) - 0.5) + ProcessingCost;''')
	cur.execute('''CREATE TABLE LowEffortRecipeTable AS SELECT * FROM MaxTax UNION SELECT * FROM MinTax;''')
	cur.execute('''CREATE TABLE LowEffortRecipeTable1 AS SELECT *, (ingredient1lowprice * ingredient1Qty) + (ingredient2lowprice * ingredient2Qty) + (ingredient3lowprice * ingredient3Qty) AS TotalLowCost, min(coalesce(ingredient1buylimit / ingredient1Qty, 'none'), coalesce(ingredient2buylimit / ingredient2Qty, 'none'), coalesce(ingredient3buylimit / ingredient3Qty, 'none'), coalesce((ingredient1hourlylowvolume * 4) / ingredient1Qty, 'none'), coalesce((ingredient2hourlylowvolume * 4) / ingredient2Qty, 'none'), coalesce((ingredient3hourlylowvolume * 4) / ingredient3Qty, 'none')) AS EffectiveBuyLimit FROM LowEffortRecipeTable;''')
	cur.execute('''CREATE TABLE LowEffortRecipeTable2 AS SELECT *, round(QtyProduced * (high - TotalLowCost - Tax)) AS HighMargin, round(QtyProduced * (low - TotalLowCost - Tax)) AS LowMargin FROM LowEffortRecipeTable1;''')
	cur.execute('''CREATE TABLE FinalOutput AS SELECT ProductName AS ItemName, RecipeType, HighMargin, LowMargin, round(HighMargin * EffectiveBuyLimit) AS HighMaxProfit, round(LowMargin * EffectiveBuyLimit) AS LowMaxProfit FROM LowEffortRecipeTable2 WHERE HighMargin + LowMargin > 0  ORDER BY HighMaxProfit DESC;''')

	# export CraftingReport to outputdb sqlite DB and cleanup
	dfrows = pd.read_sql('SELECT * FROM FinalOutput', tempdb)
	if dfrows.empty:
		dfrows = nullresult()
	dfrows.to_sql('CraftingReport', outputdb, if_exists='replace', index=False)
	cur.executescript('''DROP TABLE IF EXISTS LowEffortRecipeTable; DROP TABLE IF EXISTS LowEffortRecipeTableTax; DROP TABLE IF EXISTS MaxTax; DROP TABLE IF EXISTS MinTax; DROP TABLE IF EXISTS LowEffortRecipeTable; DROP TABLE IF EXISTS LowEffortRecipeTable1; DROP TABLE IF EXISTS LowEffortRecipeTable2; DROP TABLE IF EXISTS FinalOutput;''')

	## Report #4 - High-Low Margin (hourly)
	cur.execute('''CREATE TABLE MasterTableTax AS SELECT * FROM MasterTable;''')
	cur.execute('''ALTER TABLE MasterTableTax ADD COLUMN Tax;''')
	cur.execute('''CREATE TABLE MaxTax AS SELECT * FROM MasterTableTax WHERE round(GranularDailyMeanHigh) > 500000000;''')
	cur.execute('''CREATE TABLE MinTax AS SELECT * FROM MasterTableTax WHERE round(GranularDailyMeanHigh) <= 500000000;''')
	cur.execute('''UPDATE MaxTax SET Tax = 5000000;''')
	cur.execute('''UPDATE MinTax SET Tax = round((GranularDailyMeanHigh * 0.01) - 0.5);''')
	cur.execute('''CREATE TABLE DailyCSVwithTax AS SELECT * FROM MaxTax UNION SELECT * FROM MinTax;''')
	cur.execute('''CREATE TABLE NoBuyLimit AS SELECT *, ((GranularDailyMeanHigh - GranularDailyMeanLow - Tax) * 24 * MIN(GranularDailyMeanVolumeLow, GranularDailyMeanVolumeHigh)) AS NoBuyLimitProfit FROM DailyCSVwithTax;''')
	cur.execute('''CREATE TABLE WithBuyLimit AS SELECT id, ((GranularDailyMeanHigh - GranularDailyMeanLow - Tax) * mappinglimit) AS WithBuyLimitProfit FROM DailyCSVwithTax''')
	cur.execute('''CREATE TABLE DailyCSVwithProfit AS SELECT *, MIN(NoBuyLimit.NoBuyLimitProfit, COALESCE(WithBuyLimit.WithBuyLimitProfit, 'NONE')) AS AdjustedPotentialDailyProfit FROM NoBuyLimit, WithBuyLimit WHERE NoBuyLimit.id = WithBuyLimit.id;''')
	cur.execute('''CREATE TABLE FinalOutput AS SELECT *, ((GranularDailyMeanHigh - GranularDailyMeanLow - Tax) / GranularDailyMeanLow) * 100 AS ROI FROM DailyCSVwithProfit ORDER BY AdjustedPotentialDailyProfit DESC;''')
	
	# export High-Low Margin Report to outputdb sqlite DB and cleanup                                                                                                                                                                                                                                                                                               #filter items with poor ROI, #filter items with poor abs. profit, #filter items with low buy or sell volume                       #filter items where trade volume today is disproportionately higher than historical avg                              #filter items with extreme volatility                                                                   #filter items that don't trade on buy or sell at least once every 2 hours, eliminating items with low quantity of trades even if volume itself is high
	dfrows = pd.read_sql('''SELECT mappingname AS ItemName, round(GranularDailyMeanVolumeLow * 24) AS LowVol, round(GranularDailyMeanVolumeHigh * 24) AS HighVol, round(GranularDailyMeanLow) AS LowPrice, round(GranularDailyMeanHigh) AS HighPrice, round(AdjustedPotentialDailyProfit) AS DailyProfit, mappinglimit AS BuyLimit, round(ROI) AS pctROI FROM FinalOutput WHERE ROI > 4 AND AdjustedPotentialDailyProfit > 500000 AND MIN(GranularDailyMeanVolumeHigh, GranularDailyMeanVolumeLow) > 4 AND (MonthlyMeanVolumeLow + MonthlyMeanVolumeHigh) > 12 * (GranularDailyMeanVolumeLow + GranularDailyMeanVolumeHigh) AND MonthlyMaxHigh > MonthlyMaxLow AND (high - low - tax) > 0 AND MonthlyMedianVolumeHigh > 0 AND MonthlyMedianVolumeLow > 0 AND GranularDailyMedianVolumeHigh > 0 AND GranularDailyMedianVolumeLow > 0;''', tempdb)
	if dfrows.empty:
		dfrows = nullresult()
	dfrows.to_sql('HighLowSpread', outputdb, if_exists='replace', index=False)
	cur.executescript('''DROP TABLE IF EXISTS MasterTableTax; DROP TABLE IF EXISTS MaxTax; DROP TABLE IF EXISTS MinTax; DROP TABLE IF EXISTS DailyCSVwithTax; DROP TABLE IF EXISTS NoBuyLimit; DROP TABLE IF EXISTS WithBuyLimit; DROP TABLE IF EXISTS DailyCSVwithProfit; DROP TABLE IF EXISTS FinalOutput;''')
	
	#close connections
	tempdb.close()
	outputdb.commit()
	outputdb.close() 
