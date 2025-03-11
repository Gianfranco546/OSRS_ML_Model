## Originally published at https://poignanttech.com/projects/
## the primary function of this script is to fetch historical market data from the OSRS Wiki API (https://oldschool.runescape.wiki/w/RuneScape:Real-time_Prices)
## the secondary function of this script is to generate statistical reports of this data
## data is stored in a SQLite database 'osrsmarketdata.sqlite' in the same directory where the script is executed
## This script has been tested on Debian GNU/Linux 12 with Python 3.11.2 and is scheduled using run-one/cron to execute once every 5 minutes, which corresponds with the /5m endpoint refresh interval of the OSRS Wiki API

## Before using this script, you must uncomment the line below and enter contact information within the string. See the above URL for more information.
headers = {'User-Agent': 'Data collection for ML training - gianfrancoameri2002@gmail.com',}

## sleepdur sets our sleep interval (in seconds) between API calls (the OSRS Wiki API permits up-to 1 call per second)
sleepdur = 4

## By default, 'LowEffortRecipes.csv' must be located in the same directory as the script before execution. If you aren't interested in importing this table, set the following value to 0:
UseRecipeTable = 1

## define desired market data endpoints, their corresponding interval length (in seconds), and retention threshold (in days). 
# The retention threshold for each endpoint can be increased/decreased as you wish. The default settings will generate ~350MB of market data.
endpoints = [['5m/', 300, 2], ['1h/', 3600, 30], ['24h/', 86400, 365]]

## this script supports multiprocessing for report generation. cpucount specifies the number of processes to spawn for executing the ReportGenWorker function against all typeids
cpucount = 1

## define parameters for calculating market statistics (mean, median, min, max volume/price, etc)
# Smaller interval sizes correspond with greater granularity of data. (86400 = 1 day, 300 = 5 minutes)
# StartRange and StopRange together define the reporting window. StartRange indicates the point in the past where we want reporting to start. The default StopRange of 0 represents the present.
# the default settings reflect how data is processed for use on poignanttech.com and ensure compatibility with the companion script, OSRSReportGen.py.
ReportParameters = [
	{ "Type": "Weekly", "IntervalSize": 86400, "StartRange": 7, "StopRange": 0},
	{ "Type": "Monthly", "IntervalSize": 86400, "StartRange": 30, "StopRange": 0},
	{ "Type": "Yearly", "IntervalSize": 86400, "StartRange": 360, "StopRange": 0},
	{ "Type": "GranularDaily", "IntervalSize": 3600, "StartRange": 24, "StopRange": 0},
	{ "Type": "GranularBiweekly", "IntervalSize": 3600, "StartRange": 360, "StopRange": 0},
	{ "Type": "GranularMonthly", "IntervalSize": 3600, "StartRange": 720, "StopRange": 0},
	{ "Type": "VeryGranularFiveMinute", "IntervalSize": 300, "StartRange": 1, "StopRange": 0},
	{ "Type": "VeryGranularHourly", "IntervalSize": 300, "StartRange": 12, "StopRange": 0},
	{ "Type": "VeryGranularDaily", "IntervalSize": 300, "StartRange": 288, "StopRange": 0}
]

## define endpoint url prefix
urlprefix = 'https://prices.runescape.wiki/api/v1/osrs/'

## import modules
import sqlite3
import glob
from glob import glob; from os.path import expanduser
import pandas as pd
import requests
import json
import time
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from bs4 import BeautifulSoup
import re
import statistics
import multiprocessing as mp

## change working directory to script location
pathtorunfile = os.path.dirname(__file__)
os.chdir(pathtorunfile)

## configure logging
log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s')
logfile = 'logfile.txt'
my_handler = RotatingFileHandler(logfile, mode='a', maxBytes=1024*1024, backupCount=2, encoding=None, delay=0)
my_handler.setFormatter(log_formatter)
my_handler.setLevel(logging.INFO)
app_log = logging.getLogger('root')
app_log.setLevel(logging.INFO)
app_log.addHandler(my_handler)

### functions
## function for calling OSRS Wiki API
def FetchData(url):
	try:
		r = requests.get(url, headers=headers)
	except:
		# halt execution if the connection closes or if some other non-https error occurs
		app_log.info(f'fetching historical data failed due to connection/network issue; terminating script')
		sys.exit()
	else:
		if r.status_code == 200:
			# if connection succeeds, wait for sleep duration to pass and return response to parent thread 
			time.sleep(sleepdur)
			return(r)
		else:
			# if we get anything other than a 200 status, we will record the error and halt execution.
			app_log.info(f'fetching data for {url} failed due to {r.status_code}')
			sys.exit()

## function for saving json to sqlite database
def dbwrite(endpointinterval, r):
	jsondata = r.json()
	currenttimestamp = int(jsondata["timestamp"])
	# check database for a matching timestamp
	cur.execute("SELECT DISTINCT timestamp FROM marketdata WHERE interval = ? AND timestamp = ?;", (endpointinterval, currenttimestamp, ))
	dbtimestamp = cur.fetchone()
	# if the database is empty, or if there is no matching timestamp, proceed with conversion and write operations.
	if dbtimestamp is None:
		dbtimestamp = [0]
	if dbtimestamp[0] != currenttimestamp:
		responsedata = jsondata["data"]
		if bool(responsedata) is False:
			# if the response itself is empty, write a single row to the table including the interval and timestamp while leaving other fields null. This can occur when no data for the indicated timestamp is available (ie. server maintenance)
			cur.execute("INSERT INTO marketdata (interval, timestamp) VALUES (?, ?);", (endpointinterval, currenttimestamp, ))
			marketdatadb.commit()
		else:
			# if the response is not empty, extract keys (item typeids) and values (market statistics) from the json and assemble/write corresponding rows to the database.
			keylist = list(responsedata.keys())
			valuelist = list(responsedata.values())
			totalvalues = []
			totalkeysandvalues = []
			for value in valuelist:
				subvalues = list(value.values())
				totalvalues.append(subvalues)
			for zipkey, zipvalue in zip(keylist, totalvalues):
				totalkeysandvalues.append([endpointinterval, currenttimestamp, zipkey] + zipvalue)
			cur.executemany("INSERT INTO marketdata (interval, timestamp, typeid, avgHighPrice, highPriceVolume, avgLowPrice, lowPriceVolume) VALUES (?, ?, ?, ?, ?, ?, ?);", totalkeysandvalues)
			marketdatadb.commit()
	# return json timestamp to parent thread
	return(currenttimestamp)

## function for generating a list of missing timestamps
def gentimestamplist(endpointinterval, endpointretention, lasttimestamp):
	# check for existing endpoint/timestamp in marketdata table.
	try:
		cur.execute("SELECT timestamp FROM marketdata WHERE interval = ? AND timestamp = ?;", (endpointinterval, lasttimestamp))
		firsttimestamp = int(cur.fetchone()[0])
	# if there is no value in the marketdata table for the indicated endpoint, we will include all timestamps spanning the endpoint retention threshold through the present
	except:
		firsttimestamp = lasttimestamp - (86400 * endpointretention)
	# if the timestamp in marketdata is older than our endpoint retention threshold, the firsttimestamp will be limited to avoid fetching data outside the bounds of our desired timeframe
	finally:
		if firsttimestamp < lasttimestamp - (86400 * endpointretention):
			firsttimestamp = lasttimestamp - (86400 * endpointretention)
	# iterate over all timestamps between the first and last intervals, checking the database and adding any missing values to a list
	timestamplist = []
	while firsttimestamp != lasttimestamp:
		cur.execute("SELECT DISTINCT timestamp FROM marketdata WHERE interval = ? AND timestamp = ?;", (endpointinterval, firsttimestamp, ))
		dbtimestamp = cur.fetchone()
		if dbtimestamp is None:
			dbtimestamp = [0]
		if dbtimestamp[0] != firsttimestamp:
			timestamplist.append(firsttimestamp)
		firsttimestamp = firsttimestamp + endpointinterval
	return(timestamplist)
	
## function for deleting values for timestamps older than our retention threshold
def purgeold(endpointinterval, endpointretention, lasttimestamp):
	firsttimestamp = lasttimestamp - (86400 * endpointretention)
	cur.execute("DELETE FROM marketdata WHERE interval = ? AND ? > timestamp;", (endpointinterval, firsttimestamp, ))
	marketdatadb.commit()
	
## function for refreshing our Mapping endpoint data
def mappingrefresh(lasttimestamp):
	cur.execute("CREATE TABLE IF NOT EXISTS Mapping (typeid INT, members INT, lowalch INT, buylimit INT, value INT, highalch INT, icon TEXT, name TEXT);")
	cur.execute("CREATE TABLE IF NOT EXISTS MappingMax (timestamp INT);")
	marketdatadb.commit()
	# check MappingMax table to verify age of existing mapping endpoint data
	try:
		cur.execute("SELECT timestamp FROM MappingMax;")
		firsttimestamp = int(cur.fetchone()[0])
		marketdatadb.commit()
	except:
		firsttimestamp = 0
	finally:
		# obtain mapping data if existing data is more than a day old (ie. 86400 seconds)
		if lasttimestamp > firsttimestamp + 86400:
			# set mapping URL
			url = 'https://prices.runescape.wiki/api/v1/osrs/mapping'
			# call the API and extract data
			jsonresponse = FetchData(url)
			jsondata = jsonresponse.json()
			df = pd.DataFrame(jsondata)
			# drop "examine" column
			df.pop(df.columns[0])
			# insert new values into Mapping table and update MappingMax timestamp
			cur.execute("DELETE FROM Mapping;")
			for row in df.itertuples(index=False, name=None):
				cur.execute('INSERT INTO Mapping VALUES (?, ?, ?, ?, ?, ?, ?, ?)', row)
			cur.execute("DELETE FROM MappingMax;")
			cur.execute("INSERT INTO MappingMax VALUES (?);", (lasttimestamp, ))
			marketdatadb.commit()
			app_log.info('mapping update completed')
			time.sleep(sleepdur)
		
## function for obtaining black market GP exchange rates.
## By-default, this will return 0.20 (USD per 1M GP) as a rough approximation of the exchange rate.
## Since these websites generally don't offer APIs, getting this information requires web scraping. 
## I've provided a template below if you want to try to scrape this information yourself. You can alternatively comment-out this function in the runtime if you don't need this data.
## See the following article for more context on why this information is valuable: https://poignanttech.com/2024/03/09/virtual-markets-part-three-price-floors/
def blackmarketscrape(lasttimestamp):
	cur.execute("CREATE TABLE IF NOT EXISTS BlackMarket (timestamp INT, exchangerate REAL);")
	marketdatadb.commit()
	try:
		cur.execute("SELECT timestamp FROM BlackMarket;")
		firsttimestamp = int(cur.fetchone()[0])
	except:
		firsttimestamp = 0
	finally:
		if lasttimestamp > firsttimestamp + 86400:
			## remove these two lines if you don't want to use the default value of 0.20 (USD to 1M GP)
			cur.execute("REPLACE INTO BlackMarket(timestamp, exchangerate) VALUES(?, ?);", (lasttimestamp, 0.20))	
			marketdatadb.commit()
			## web scraping template. Reporting on poignanttech.com uses the average from three different websites.
			#spoofedheaders = {'User-Agent': 'enter headers here to pretend that your script is a web browser'}
			#goldpricevalues = []
			#goldurl = "some gold selling website"
			#try:
			#	currentprice = None
			#	page = requests.get(goldurl, headers=spoofedheaders)
			#	soup = BeautifulSoup(page.content, "html.parser")
			#	currentprice = 'this is where you should hack and slash the html to get at the data you want'
			#	goldpricevalues.append(currentprice)
			#except:
			#	pass
			#try:
			#	cur.execute("DELETE FROM BlackMarket;")
			#	cur.execute("REPLACE INTO BlackMarket(timestamp, exchangerate) VALUES(?, ?);", (lasttimestamp, statistics.mean(goldpricevalues)))			
			#	marketdatadb.commit()
			#	app_log.info(f'black market exchange rate from {len(goldpricevalues)} sites collected and averaged successfully')
			#except:
			#	app_log.info('ATTENTION: Collecting and/or saving black market exchange rates failed')
			#	pass

## function for running flagged statistical reports against a specified typeid. This function is invoked by a multiprocessing pool.
def ReportGenWorker(typeid):
	tempdb = sqlite3.connect(":memory:")
	tempcur = tempdb.cursor()
	tempcur.execute("ATTACH DATABASE 'osrsmarketdata.sqlite' AS marketdatadb;")
	ReportList = []
	for Report in ReportParametersRun:
		TimestampList = Report.get("TimestampList")
		TimestampCount = len(TimestampList)
		## generate dictionary to store results
		resultdict = {}
		resultdict["typeid"] = typeid
		resultdict["Type"] = Report.get("Type")
		
		## populate temporary table with data for current typeid and report type. this is substantially faster than running all of the following queries
		tempcur.execute("CREATE TABLE temptable AS SELECT timestamp, avgHighPrice, highPriceVolume, avgLowPrice, lowPriceVolume FROM marketdata WHERE typeid = ? AND interval = ? AND timestamp >= ? AND timestamp <= ?;", (typeid, Report.get("IntervalSize"), Report.get("FirstTimestamp"), Report.get("LastTimestamp")))
		
		## count number of entries for the current typeid
		tempcur.execute("SELECT COUNT(timestamp) FROM temptable;")
		timestampcount = tempcur.fetchone()
		MissingTSqty = int(TimestampCount) - int(timestampcount[0])

		## normalize dataset where entries are missing 
		if MissingTSqty > 0:
			NullList = []
			while MissingTSqty > 0:
				NullList.append((0, 0, 0))
				MissingTSqty = MissingTSqty - 1
			tempcur.executemany("INSERT OR IGNORE INTO temptable (timestamp, lowPriceVolume, highPriceVolume) VALUES (?, ?, ?);", NullList)

		# calculate weighted mean price, mean volume, min price, max price
		tempcur.execute("SELECT (sum(LowPriceVolume * avgLowPrice)) / sum(lowPriceVolume) AS MeanLow, (sum(HighPriceVolume * avgHighPrice)) / sum(highPriceVolume) AS MeanHigh, AVG(lowPriceVolume) AS MeanVolumeLow, AVG(highPriceVolume) AS MeanVolumeHigh, min(avgLowPrice) AS MinLow, min(avgHighPrice) AS MinHigh, max(avgLowPrice) AS MaxLow, max(avgHighPrice) AS MaxHigh FROM temptable;")
		rows = tempcur.fetchall()
		for row in rows:
			resultdict["MeanLow"] = row[0]
			resultdict["MeanHigh"] = row[1]
			resultdict["MeanVolumeLow"] = row[2]
			resultdict["MeanVolumeHigh"] = row[3]
			resultdict["MinLow"] = row[4]
			resultdict["MinHigh"] = row[5]
			resultdict["MaxLow"] = row[6]
			resultdict["MaxHigh"] = row[7]
			
		# calculate median volume. SQLite does not have a native median function so this is probably the slowest and most computationally expensive part of the script
		# we can use a simple heuristic to quickly infer when median volume will be 0 in-order to marginally reduce CPU load
		if int(TimestampCount) > int(timestampcount[0]) * 2:
			resultdict["MedianVolumeLow"] = 0
			resultdict["MedianVolumeHigh"] = 0
			tempcur.execute("SELECT AVG(avgLowPrice) AS MedianLow, AVG(avgHighPrice) AS MedianHigh FROM (SELECT avgLowPrice FROM temptable ORDER BY avgLowPrice LIMIT 2 - (SELECT COUNT(*) FROM temptable) % 2 OFFSET (SELECT (COUNT(*) - 1) / 2 FROM temptable)), (SELECT avgHighPrice FROM temptable ORDER BY avgHighPrice LIMIT 2 - (SELECT COUNT(*) FROM temptable) % 2 OFFSET (SELECT (COUNT(*) - 1) / 2 FROM temptable));")
			rows = tempcur.fetchall()
			for row in rows:
				resultdict["MedianLow"] = row[0]
				resultdict["MedianHigh"] = row[1]
		else:
			tempcur.execute("SELECT AVG(avgLowPrice) AS MedianLow, AVG(avgHighPrice) AS MedianHigh, AVG(lowPriceVolume) AS MedianVolumeLow, AVG(highPriceVolume) AS MedianVolumeHigh FROM (SELECT avgLowPrice FROM temptable ORDER BY avgLowPrice LIMIT 2 - (SELECT COUNT(*) FROM temptable) % 2 OFFSET (SELECT (COUNT(*) - 1) / 2 FROM temptable)), (SELECT avgHighPrice FROM temptable ORDER BY avgHighPrice LIMIT 2 - (SELECT COUNT(*) FROM temptable) % 2 OFFSET (SELECT (COUNT(*) - 1) / 2 FROM temptable)), (SELECT lowPriceVolume FROM temptable ORDER BY lowPriceVolume LIMIT 2 - (SELECT COUNT(*) FROM temptable) % 2 OFFSET (SELECT (COUNT(*) - 1) / 2 FROM temptable)), (SELECT highPriceVolume FROM temptable ORDER BY highPriceVolume LIMIT 2 - (SELECT COUNT(*) FROM temptable) % 2 OFFSET (SELECT (COUNT(*) - 1) / 2 FROM temptable));")
			rows = tempcur.fetchall()
			for row in rows:
				resultdict["MedianLow"] = row[0]
				resultdict["MedianHigh"] = row[1]
				resultdict["MedianVolumeLow"] = row[2]
				resultdict["MedianVolumeHigh"] = row[3]

		# cleanup and append result to list
		tempcur.execute("DROP TABLE IF EXISTS temptable;")
		ReportList.append(resultdict)
	# close db and return reports to main thread
	tempdb.close()	
	return(ReportList)

### runtime
if __name__ == '__main__':

	## data fetching section
	# create database file if it does not exist, enable write-ahead logging, create cursor/tables/columns/indices
	marketdatadb = sqlite3.connect("osrsmarketdata.sqlite")
	marketdatadb.execute('pragma journal_mode=wal')
	cur = marketdatadb.cursor()
	cur.execute("CREATE TABLE IF NOT EXISTS marketdata (interval INT, timestamp INT, typeid INT, avgHighPrice INT, highPriceVolume INT, avgLowPrice INT, lowPriceVolume INT);")
	cur.execute("CREATE INDEX IF NOT EXISTS marketdataindex ON marketdata (typeid, interval, timestamp);")
	cur.execute("CREATE INDEX IF NOT EXISTS marketdataintervalindex ON marketdata (interval, timestamp);")
	cur.execute("CREATE TABLE IF NOT EXISTS marketdatamax (interval INT, timestamp INT);")
	marketdatadb.commit()
	
	# collect data for each indicated endpoint
	lasttimestamplist = []
	for endpoint in endpoints:
		endpointname = endpoint[0]
		endpointinterval = endpoint[1]
		endpointretention = endpoint[2]
		# perform initial API call and append response to endpoint data list.
		r = FetchData(f'{urlprefix}{endpointname}')
		# write to database if nothing for this endpoint/timestamp already exists
		lasttimestamp = dbwrite(endpointname, r)
		# generate list of timestamps missing from the current endpoint table
		timestamplist = gentimestamplist(endpointinterval, endpointretention, lasttimestamp)
		# iterate over missing timestamps and fetch/write data to database. Due to the API rate limit, this can potentially take a very long time when running for the first time
		if timestamplist is not None:
			for timestamp in timestamplist:
				url = f'{urlprefix}{endpointname}?timestamp={timestamp}'
				r = FetchData(url)
				dbwrite(endpointinterval, r)
		# update Max table with current timestamp
		cur.execute("DELETE FROM marketdatamax WHERE interval = ?;", (endpointinterval, ))
		cur.execute("REPLACE INTO marketdatamax(interval, timestamp) VALUES(?, ?);", (endpointinterval, lasttimestamp))
		marketdatadb.commit()
		# purge data older than indicated retentionthreshold
		#purgeold(endpointinterval, endpointretention, lasttimestamp)
		lasttimestamplist.append(lasttimestamp)
	lasttimestamp = max(lasttimestamplist)	
	
	# refresh mapping if data is older than 1 day
	mappingrefresh(lasttimestamp)
	
	# refresh black market GP exchange rate if data is older than 1 day
	blackmarketscrape(lasttimestamp)
	
	# open and save csv recipe table to database
	if UseRecipeTable == 1:
		try:
			LowEffortRecipes = pd.read_csv("LowEffortRecipes.csv", sep=",")
			LowEffortRecipes.to_sql('Recipes', marketdatadb, if_exists='replace')
		except:
			app_log.info('Note: LowEffortRecipes.csv is missing from the directory. The Crafting report contained in OSRSReportGen.py will fail to run if this data has not been previously loaded.')	
	
	## statistics report generation section (if you're only interested in using the script to collect and save raw data from the API, the rest of the runtime can be omitted.)
	# generate list of unique typeids
	cur.execute("SELECT DISTINCT typeid FROM mapping;")
	typeidlist = [typeid[0] for typeid in cur.fetchall()]
	
	# identify stale reports and generate a list of dictionaries to process
	ReportParametersRun = []
	for Report in ReportParameters:
		IntervalSize = Report.get("IntervalSize")
		try:
			cur.execute("SELECT marketdatamax.timestamp AS timestamp FROM marketdatamax, marketstatstimestamp WHERE marketdatamax.interval = ? AND marketstatstimestamp.interval = ? AND marketdatamax.timestamp = marketstatstimestamp.timestamp;", (IntervalSize, IntervalSize))
			rows = cur.fetchone()
			currenttimestamp = int(rows[0])
		except:
			currenttimestamp = None
		if currenttimestamp is None:
			cur.execute("SELECT timestamp FROM marketdatamax WHERE interval = ?;", (IntervalSize, ))
			LatestTimestamp = cur.fetchone()[0]
			Report["FirstTimestamp"] = LatestTimestamp - (Report.get("StartRange") * IntervalSize)
			Report["LastTimestamp"] = LatestTimestamp - (Report.get("StopRange") * IntervalSize)
			cur.execute("SELECT DISTINCT timestamp FROM marketdata WHERE interval = ? AND timestamp >= ? AND timestamp <= ?;", (IntervalSize, Report.get("FirstTimestamp"), Report.get("LastTimestamp")))
			TimestampList = [timestamp[0] for timestamp in cur.fetchall()]
			Report["TimestampList"] = TimestampList
			Report["TimestampCount"] = len(TimestampList)
			ReportParametersRun.append(Report)
	
	# process any stale reports
	if len(ReportParametersRun) > 0: 
		with mp.Pool(cpucount) as p:
			ResultList = p.map(ReportGenWorker, typeidlist)
		ResultList = [result for results in ResultList for result in results]

		# save results to main db
		cur.execute("CREATE TABLE IF NOT EXISTS marketstats(typeid INT, Type TEXT, MeanLow INT, MeanHigh INT, MeanVolumeLow TEXT, MeanVolumeHigh TEXT, MedianLow INT, MedianHigh INT, MedianVolumeLow TEXT, MedianVolumeHigh TEXT, MinLow INT, MinHigh INT, MaxLow INT, MaxHigh INT, UNIQUE(typeid, Type) ON CONFLICT REPLACE);")
		cur.execute("CREATE INDEX IF NOT EXISTS marketstatsindex ON marketstats (Type, typeid);")
		cur.executemany("INSERT OR REPLACE INTO marketstats VALUES (:typeid, :Type, :MeanLow, :MeanHigh, :MeanVolumeLow, :MeanVolumeHigh, :MedianLow, :MedianHigh, :MedianVolumeLow, :MedianVolumeHigh, :MinLow, :MinHigh, :MaxLow, :MaxHigh);", ResultList)
		cur.execute("DROP TABLE IF EXISTS marketstatstimestamp;")
		cur.execute("CREATE TABLE IF NOT EXISTS marketstatstimestamp AS SELECT * FROM marketdatamax;")
		marketdatadb.commit()
		marketdatadb.close()