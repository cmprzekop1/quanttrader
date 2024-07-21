
#MACD Strategy
#Gather the opening and closing prices of the stock for each day in time range
import pandas as pd
from pandas_datareader.yahoo.headers import DEFAULT_HEADERS
import datetime
import pickle
import requests_cache
import requests
import yfinance as yf #(API) can use other packages like Naver Finance data or Nasdaq Trader Symbols
yf.pdr_override()

expire_after = datetime.timedelta(days=5)
session = requests_cache.CachedSession(cache_name='cache', backend='sqlite', expire_after = expire_after)
session.headers = DEFAULT_HEADERS

from bs4 import BeautifulSoup


#date class for date updates
class Dates:
    def __init__(self, current):
        #sets start date to the first of the year
        self.start = datetime.datetime(datetime.date.today().year, 1, 1)
        self.curr = current
        
    def setDate(self):
        if (datetime.date.today() - self.current).days > 1:
            hold = datetime.days.today().strftime("%Y-%m-%d")
            self.curr = hold.strftime("%Y-%m-%d")

startD = Dates(2024).start
startD = startD.strftime("%Y-%m-%d")
currentdate = datetime.datetime.now()
currentdate = currentdate.strftime("%Y-%m-%d")


#prints dictionary
def pDict(data):
    for key, value in data.items():
        if isinstance(value, tuple):
            value = f'{value[0]}(statistic), {value[1]}(pvalue)'
        print(f"{key}: {value}")

#Converts stock numbers that use letters to denote scale into only numerical values
def convert_to_num(string):
    suffixes = {
        'K': 10**3,
        'M': 10**6,
        'B': 10**9,
        'T': 10**12
    }

    #checks if the last character is a letter denoting suffix and converts the string into a float then multiplies by denoted factor
    if string[-1] in suffixes:
        return float(string[:-1]) * suffixes[string[-1]]
    else: 
        return float(string)

#save data to a pkl file so we don't have to redownload at start
class Stonk:
    def __init__(self, ticker, start=startD, end = currentdate, df=None, ini_capital = 1000, path = r"C:/Users/16096/Desktop/Projects/stockbacktester/data/stocks.pkl", save_new=True, debug = False):
        self.ticker = ticker
        self.ini_cap = ini_capital
        self.path = path
        self.savedData = None
        self.df = df
        self.start = start
        self.end = end
        self.positionOutCash = False
        self.strats = []
        self.printdebug = debug
        self.save_new = save_new
        self.sum_stats = None
    
        #context manager to open file
        with open(self.path, 'rb') as handle:
            self.savedBars = pickle.load(handle)
            if self.printdebug:
                print("saved bars")
                print(self.savedBars.keys())
            if ticker in self.savedBars:
                if self.printdebug:
                    print("ticker exists in savedBars")
                self.df = self.savedBars[ticker]

        if (self.df is None or save_new):
            print(1)
            print(self.start)
            print(self.end)
            self.df = self.getData(ticker, start = self.start, end = self.end, save_new=True)
        self.save = self.df
        self.positions = pd.DataFrame(self.df.index)
        self.sum_stats = self.sum_stats_getter(self.ticker)

    #gets stock data for day
    def getData(self, ticker, start = startD, end = currentdate, save_new = False, saveToCSV = True):
        #check if data exists
        if (self.save_new or self.ticker not in self.savedData):
            try: 
                stock_data = yf.download(self.ticker, start=start, end=end)
                stock_data = stock_data.reset_index(inplace=True)
                if saveToCSV:
                    path = r"C:/Users/16096/Desktop/Projects/stockbacktester/data/stock_csv"
                    stock_data.to_csv(f'{path + "/" +self.ticker}.csv', index =False)
                self.savedData[self.ticker] = stock_data
                with open(self.path, 'wb') as handle:
                    pickle.dump(self.savedData, handle, protocol=pickle.HIGHEST_PROTOCOL)
                return stock_data
            except:
                print("Skipping stock {}, bad data".format(self.ticker))
        else:
            return self.savedData[ticker]
        
    def sum_stats_getter(self, name):
        try:
            #parse the html document, findspeific elements within that document, and extract the text
            stat_dict = {}
            url = f'https://www.cnbc.com/quotes/{name}?tab=news'
            html = requests.get(url).content
            soup = BeautifulSoup(html, 'html.parser')
            sum_stats = soup.find_all('li', {'class': 'Summary-stat'})
            stat_dict = {}
            #finds all <li> items in the html that have the class summary stat and smmary value
            for stat in sum_stats:
                label = stat.find('span', {'class': 'Summary-label'}).text
                value = stat.find('span', {'class': 'Summary-value'}).text 

                if label == "Market Cap":
                    value = convert_to_num(value)
                stat_dict[label] = value
                stat_dict['date'] = datetime.datetime.now()
            return stat_dict
        except Exception as e:
            print(f"Error processing {name: {str(e)}}")
            return{}
    
    def addStrat(self, func):
        self.strats.append(func)
    
    def viewStrats(self):
        for x in self.strats:
            print(x)
    def backtest(self):
        strategies = self.strats

        self.positions[self.ticker + '_position'] = pd.concat(strategies).groupby(level=0).mean()
        self.positions[self.ticker + '_position'].at[0] = 0
        self.positions[self.ticker + '_open'] = self.save.Open
        self.positions[self.ticker + '_close'] = self.save.Close
        self.positions['date'] = self.save.Date
        self.positions[self.ticker + '_pos_diff'] = self.positions[self.ticker + "_position"].diff()
        self.positions[self.ticker + "_holdings"] = 0 
        self.positions[self.ticker + "_pos_diff"].at[0] = 0
        self.positions.at[0, f"{self.ticker}_cash"] = self.ini_cap

        for i, r in self.positions.iterrows():
            if i == 0:
                continue

            price = r[self.ticker + "_open"]
            cash = self.positions.loc[i - 1][self.ticker + "_cash"]
            holdings  = self.positions.loc[i-1][self.ticker + "_holdings"]
            position = self.positions.loc[i][self.ticker + '_posiiton']
            pos_diff = self.positions.loc[i][self.ticker+ '_position'] - self.positions.loc[i-1][self.ticker + '_position']
            cash_needed = pos_diff * price
            
            if (cash > cash_needed) and (not self.positionOutCash):
                cash = cash - cash_needed
            #holdings = holdings + cash_needed

            #Tests if there is enough cash to make a positional buy
            elif (cash_needed < 0):
                cash = cash-cash_needed
                if(cash > (position*price)):
                    self.positionOutCash = False
            else:
                if (i == 1): #ran out of money
                    self.positions.at[i, f"{self.ticker}_position"] = cash / price #set first index position
                    cash = 0
                else:
                    self.positions.at[i,f"{self.ticker}_position"] = self.positions.at[i-1, f"{self.ticker}_position"] #set position to prev position

            self.positions.at[i, f"{self.ticker}_cash"] = cash
           
        self.positions[f'{self.ticker}_holdings'] = self.positions[f"{self.ticker}__position"] * self.positions[f"{self.ticker}_open"]
        self.positions['total'] = self.positions[f"{self.ticker}_cash"] + self.positions[f"{self.ticker}_holdings"]
        self.positions = self.positions.round(2)
        return self.positions
    
    def plot(self):
        if 'total' not in self.positions.columns:
            self.backtest()
        self.positions.plot(x=f'date', y='total')
    
    def saveCSV(self):
        self.positions.set_index('date', inplace=True)
        self.positions.rename(columns={f"{self.ticker}_position": 'Position', 'total': 'Total'}, inplace=True, errors=False)
        self.df.set_index('Date', inplace=True)
        self.df['Date'] = self.df.index
        df = pd.merge(self.df[['Date', 'Open', 'Hihg', 'Low', 'Close', 'Adj Close', 'Volume']], self.positions[["Position", "Total"]], left_index=True, right_index=True, how='inner')
        #df['Date'] = df.index
        path = r"C:/Users/16096/Desktop/Projects/stockbacktester/data/stock_csv"
        df.to_csv(f'{path + "/" + self.ticker}.csv', index=False)

    #strats

    def macd(self, short=12, long=26):
        df = self.df
        exp1 = df['Adj Close'].ewm(span=short, adjust=False).mean()
        exp2 = df['Adj Close'].ewm(span=long, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_norm']  = (df['macd'] - df['macd'].min()) / (df['macd'].max() - df['macd'].min()) #normalize df and add to a column
        return df['macd_norm']
    
    def buyStock(self, amount, inDollars=False):
        if inDollars: #if true then buy amount of stock
            first_price = self.df['Open'].iloc[0]
            shares = amount / first_price
            return pd.Series(shares, index = self.save.index)
        return pd.Series(amount, index = self.save.index)
