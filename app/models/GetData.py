
#MACD Strategy
#Gather the opening and closing prices of the stock for each day in time range
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas_datareader as pdr
import datetime
import pickle
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

#date class for date updates
class Dates:
    def __init__(self, current):
        #sets start date to the first of the year
        self.start = datetime.datetime(datetime.date.today().year, 1, 1)
        self.curr = current
        
    def setDate(self):
        if (datetime.date.today() - self.curr) > 1:
            self.curr = datetime.date.today().strftime("%Y-%m-%d")
            

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
    def __init__(self, ticker, start=startD, end = currentdate, df=None, ini_cap = 1000, path = r"C:/Users/16096/Desktop/Projects/stockbacktester/app/models/data/stocks.pkl", save_new=False, debug = False):
        self.ticker = ticker
        self.ini_cap = ini_cap
        self.path = path
        self.savedData = {}
        self.df = df
        self.start = start
        self.end = end
        self.positionOutCash = False
        self.strats = []
        self.printdebug = debug
        self.save_new = save_new
        self.sum_stats = None
    
        # Load existing data if available
        try:
            with open(self.path, 'rb') as handle:
                self.savedBars = pickle.load(handle)
                print(self.savedBars.keys())
                if ticker in self.savedBars:
                    print("Ticker exists in savedBars")
                    self.df = self.savedBars[ticker]
        except FileNotFoundError:
            print(f"No existing data found at {self.path}, starting fresh.")
        except EOFError:
            print("Pickle file is empty, starting fresh.")

        # Fetch new data if necessary
        if self.df is None or save_new:
            self.df = self.getData(self.ticker, start=self.start, end=self.end, save_new=False)
            if self.df is None:
                try:
                    path = r"C:/Users/16096/Desktop/Projects/stockbacktester/app/models/data/stock_csv"
                    self.df =  pd.read_csv(f'{path}/{self.ticker}.csv', index_col=None)
                    if self.df is None:
                            raise ValueError("forcing exception")
                except:
                    raise ValueError(f"Failed to fetch data for {ticker}")
        self.save = self.df
        self.positions = pd.DataFrame(self.df.index)
        self.sum_stats = self.sum_stats_getter(self.ticker)

    def getData(self, ticker, start, end, save_new, saveToCSV=False):
        # Fetch and save new data if necessary
        if save_new or ticker not in self.savedBars:
            try:
                print(1)
                df = pdr.get_data_tiingo(ticker, start=start, end=end, api_key='b11072847c2bc8387ad9f5bfec15ec202bbbe9da')
                print(2)
                df.reset_index(inplace=True)

                if saveToCSV:
                    path = r"C:/Users/16096/Desktop/Projects/stockbacktester/app/models/data/stock_csv"
                    df.to_csv(f'{path}/{self.ticker}.csv', index=False)

                # Update the savedBars dictionary with new data
                self.savedBars[self.ticker] = df

                # Save the updated dictionary back to the pickle file
                with open(self.path, 'wb') as handle:
                    pickle.dump(self.savedBars, handle, protocol=pickle.HIGHEST_PROTOCOL)

                return df
            except Exception as e:
                print(f"Skipping stock {self.ticker}, bad data: {e}")
        else:
            return self.savedBars[ticker]
        
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
        #extracts list of strategies stored in self.starts
        strategies = self.strats
        print("strat")
        print(self.strats)
        #concats all the DFs in 'strategies', groups them together by the first level of their index, and calcs mean of each group
        self.positions[self.ticker + '_position'] = pd.concat(strategies).groupby(level=0).mean()

        #ensures there is no initial stock position
        self.positions[self.ticker + '_position'].at[0] = 0

        #add columns of opening and closing price to the dataframe
        self.positions[self.ticker + '_open'] = self.save.open
        self.positions[self.ticker + '_close'] = self.save.close

        #adds dates column to dataframe
        self.positions['date'] = self.save.date

        #calculates the difference in positions from one time step to the next and stores in new '_pos_diff' column
        self.positions[self.ticker + '_pos_diff'] = self.positions[self.ticker + "_position"].diff()

        #initiializes hodlings and the position difference for the first row to 0
        self.positions[self.ticker + "_holdings"] = 0 
        self.positions[self.ticker + "_pos_diff"].at[0] = 0

        #set initial cash captial to ini_cap
        self.positions.at[0, f"{self.ticker}_cash"] = self.ini_cap

        #iterates over each row in self.positions. skips first row
        for i, r in self.positions.iterrows():
            if i == 0:
                continue
            

            #for each row, extracts the opening price, cash from prev row, holdings from prev row, curr position, pos difference
            price = r[self.ticker + "_open"]
            cash = self.positions.loc[i - 1][self.ticker + "_cash"]
            holdings  = self.positions.loc[i-1][self.ticker + "_holdings"]
            position = self.positions.loc[i][self.ticker + '_position']
            pos_diff = self.positions.loc[i][self.ticker+ '_position'] - self.positions.loc[i-1][self.ticker + '_position']
            cash_needed = pos_diff * price
            #^calcs the cash needed to adjust position


            #Tests if there is enough cash to make a positional buy by testing if there is enough cash and flag 'positionOutCash' flag is not set
            if (cash > cash_needed) and (not self.positionOutCash):
                #if above is so, cash is reduced by cash needed
                cash = cash - cash_needed
            #holdings = holdings + cash_needed


            #if cash needed is negative (indicating a sell), it adjusts cash and checks if cash is greater than value of current position. Resets 'poisitionOutCash' if true
            elif (cash_needed < 0):
                cash = cash-cash_needed #minusing a negative is addition
                if(cash > (position*price)):
                    #if above is so, reset flag
                    self.positionOutCash = False

            #if there is not enough cash: if its the first iteration, it sets the position based on available cash. Otherwise it retains previous position
            else:
                if (i == 1): #ran out of money
                    self.positions.at[i, f"{self.ticker}_position"] = cash / price #set first index position
                    cash = 0
                else:
                    self.positions.at[i,f"{self.ticker}_position"] = self.positions.at[i-1, f"{self.ticker}_position"] #set position to prev position

            #updates cash value for current row
            self.positions.at[i, f"{self.ticker}_cash"] = cash
           
        #out of loop, calculates holdings by multiplying position and opening price.   
        self.positions[f'{self.ticker}_holdings'] = self.positions[f"{self.ticker}_position"] * self.positions[f"{self.ticker}_open"]
        #calcs total value by adding cash an holdings | rounds to two decimal places
        self.positions['total'] = self.positions[f"{self.ticker}_cash"] + self.positions[f"{self.ticker}_holdings"]
        self.positions = self.positions.round(2)
        #returns self.positions dataframe
        return self.positions
    
    def plot(self):
        if 'total' not in self.positions.columns:
            self.backtest()
        self.positions.plot(x=f'date', y='total')
    
    def saveCSV(self):
        self.positions.set_index('date', inplace=True)
        self.positions.rename(columns={f"{self.ticker}_position": 'Position', 'total': 'Total'}, inplace=True, errors=False)
        self.df.set_index('date', inplace=True)
        self.df['date'] = self.df.index
        df = pd.merge(self.df[['date', 'open', 'high', 'low', 'close', 'adjClose', 'volume']], self.positions[["Position", "Total"]], left_index=True, right_index=True, how='inner')
        #df['date'] = df.index
        path = r"C:/Users/16096/Desktop/Projects/stockbacktester/data/stock_csv"
        df.to_csv(f'{path + "/" + self.ticker}.csv', index=False)

    #strats

    def macd(self, short=12, long=26):
        df = self.df
        exp1 = df['adjClose'].ewm(span=short, adjust=False).mean()
        exp2 = df['adjClose'].ewm(span=long, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_norm']  = (df['macd'] - df['macd'].min()) / (df['macd'].max() - df['macd'].min()) #normalize df and add to a column
        return df['macd_norm']
    
    def buyStock(self, amount, inDollars=False):
        if inDollars: #if true then buy amount of stock
            first_price = self.df['open'].iloc[0]
            print(first_price)
            print(amount)
            shares = amount / first_price
            print(self.save.index)
            return pd.Series(shares, index = self.save.index)
        return pd.Series(amount, index = self.save.index)
