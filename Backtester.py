import GetData
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import statsmodels.formula.api as smf
from scipy.stats import skew, kurtosis, shapiro

class Portfolio:
    def __init__(self, tickers=[], start=GetData.startD, end = GetData.currentdate, ini_cap=1000, save_new=True):
        self.stocks = []
        self.cur_cap = ini_cap
        self.backtest_df = None
        self.markov_data = None
        self.startD = start
        self.endD = end
        self.save_new = save_new
        self.stats = None
        self.ff_stats= None
        self.ini_cap = ini_cap

        for ticker in tickers:
            self.addStock(ticker)

        #takes a ticker or a stock
    def addStock(self, stock_inst):
        #if input is a ticker, fetch and create stock object
        if isinstance(stock_inst, str):
            stock_inst = GetData.Stonk(stock_inst, df = None, start=self.startD, end = self.endD, save_new=self.save_new, debug=False)
            self.stocks.append(stock_inst)
        elif isinstance(stock_inst, GetData.Stonk):
            stock_inst.save_new = self.save_new
            self.stocks.append(stock_inst)
        self.updateIniCap(self.ini_cap)
        return stock_inst
    
    def updateIniCap(self, ini_cap):
        self.cur_cap = ini_cap
        for stock in self.stocks:
            self.cur_cap -= stock.ini_cap
    
    def equalWeightPortfolio(self, tickers, capital=1000):
        num_stocks = len(tickers)
        ini_cap = capital / num_stocks
        #self.ini_cap = self.ini_cap - capital
        for ticker in tickers:
            stock_inst = GetData.Stonk(ticker, start=self.startD, end=self.endD, ini_cap = ini_cap, save_new=self.save_new)
            stock_inst.addStrat(stock_inst.buyStock(ini_cap, inDollars=True))
            self.addStock(stock_inst)
        #self.updateIniCap(self.ini_cap)
    
    def marketCapWeightedPortfolio(self, tickers, capital=1000):
        total = 0
        for ticker in tickers:
            stock_inst = self.addStock(ticker)
            total += stock_inst.sum_stats['Market Cap']
        
        for stock in self.stocks:
            weight = stock.sum_stats['Market Cap'] / total
            print(f"For {stock.ticker}, Market Cap = {stock.sum_stats['Market Cap']}, Weight = {weight}")

            weighted_captial = capital * weight
            stock.ini_cap = weighted_captial
            stock.addStrat(stock.buyStock(weighted_captial, inDollars=True))
        self.updateIniCap(self.ini_cap)
    
    def get_by_ticker(self, ticker):
        for stock in self.stocks:
            if (stock.ticker==ticker):
                return stock
                break
        return None
    
    def markowitz(self, weight=1, msr=True, plot=False):
        self.addReturns()
        port_returns = pd.DataFrame()
        for stock in self.stocks:
            port_returns[stock.ticker] = stock.df['simple_returns']
        risk_free = 0
        markov_runs = 10000

        markov_data = pd.DataFrame(columns=["id", "return", "volatility", "weights"])
        for x in range(0, int(markov_runs)):
            weights = getRandomWeights(len(self.stocks))
            volatility = getPortWeightedVol(port_returns, weights)
            ann_ret = getPortWeightedAnnualReturn(port_returns, weights)
            row = {
                "id": x,
                "return": ann_ret,
                "volatility": volatility,
                "weights": weights
            }
            markov_data = pd.concat([markov_data, pd.DataFrame.from_records([row])])
            markov_data["sharpe"] = (markov_data["return"] - risk_free) / markov_data["volatility"]
            markov_data.reset_index()
            self.markov_data = markov_data

            best_port = None
            if msr:
                best_port = markov_data.sort_values(by=["sharpe"], ascending=False).head(1)
            else:
                best_port = markov_data.sort_values(by=["volatility"], ascending=True).head(1)
            MSR = markov_data.sort_values(by=["sharpe"], ascending=False).head(1)
            GMV = markov_data.sort_values(by=["volatility"], ascending=True).head(1)

            weights = {}
            for i, x in enumerate(list(best_port['weights'])[0]):
                weights[self.stocks[i].ticker] = x * weight
                stock_instance = self.get_by_ticker(self.stocks[i].ticker)
                stock_instance.ini_cap = x * weight
                stock_instance.addStrats(stock_instance.buyStock(x * weight,inDollars=True))
            self.updateIniCap(weight)

            if (plot):
                # Plotting the scatter plot
                plt.scatter(markov_data['volatility'],
                            markov_data['return'], label='Data')

                # Highlight the MSR index row with a different color
                plt.scatter(best_port['volatility'], best_port['return'], color='red',
                            label=f"Optimized Portfolio {'MSR' if msr else 'GMV'}")

                # Adding labels and title
                plt.xlabel('Volatility')
                plt.ylabel('Return')
                plt.title('Scatter Plot of Return vs Volatility')
                plt.legend()

            return markov_data, weights
    def addReturns(self):
        for stock in self.stocks:
            stock.df['simple_returns']  = stock.df['Adj Close'].pct_change()
            stock.df['log_returns'] = np.log(stock.df['Adj Close'] + 1)
            stock.df['cum_daily_return'] = ((1 + stock.df['simple_returns']).cumprod() - 1)
            stock.df = stock.df.dropna()
    
    def backtest(self, plot=False, stats=False, ff_stats=False, ff_2=False, saveToCsv=False):
        ticker_list = []
        self.backtest_df = pd.DataFrame(self.stocks[0].df.Date)
        #return backtest_df
        for stock in self.stocks:
            stock.backtest()
            ticker_list.append(f"{stock.ticker}_total")
            self.backtest_df[f"{stock.ticker}_total"] = stock.positions.total
            self.backtest_df[f"{stock.ticker}_position"] = stock.positions[f"{stock.ticker}_position"]

        self.backtest_df['total_strats'] = self.backtest_df[ticker_list].sum(axis=1)
        self.backtest_df['cash'] = self.cur_cap
        self.backtest_df['total'] = self.backtest_df['total_strats'] + self.cur_cap
        self.start = self.backtest_df['Date'].iloc[0]
        self.end = self.backtest_df['Date'].iloc[-1]

        if (plot):
            if 'total_returns'not in self.backtest_df.columns:
                self.backtest_df['total_returns'] = self.backtest_df['total'].pct_change()
                self.backtest_df['cum_daily_return'] = ((1 + self.backtest_df['total_returns']).cumprod() -1)
                self.plotBacktest()

        if(stats or ff_stats):
                self.backtest_df['total_returns'] = self.backtest_df['total'].pct_change()
                self.backtest_df['cum_daily_return'] = ((1 + self.backtest_df['total_returns']).cumprod() -1)

                self.stats = {}  
                self.stats['pct_return'] = ((self.backtest_df['total'].iloc[-1] - self.backtest_df['total'].iloc[0]) / self.backtest_df['total'].iloc[0])
                self.stats['vol'] = np.std(self.backtest_df['total_returns'])
                self.stats['vol_annual'] = self.stats['vol'] * np.sqrt(365-self.startD)
                self.stats['mean_daily_return'] = np.mean(self.backtest_df['total_returns'])
                self.stats['mean_annual_return'] = ((1+ self.stats['mean_daily_return'])**252) - 1

                #right skew mean negative lean, we want positive so predictable neg returns and long tail of positive
                self.stats['skewness'] = skew(self.backtest_df['total_returns'].dropna())
                self.stats['excess_kurtosis'] = kurtosis(self.backtest_df['total_returns'].dropna())

                #p-value is < 0.05 reject null and data is non-uniform
                self.stats['shapiro'] = shapiro(self.backtest_df['total_returns'].dropna())
                if (stats):
                    print("Returns stats")
                    GetData.pDict(self.stats)

                self.backtest_df['Date'] = pd.to_datetime(self.backtest_df['Date'])
                self.backtest_df.set_index('Date', inplace = True)

        if (ff_stats):
        #resample to monthly freq and sum returns for months

            monthly_returns = self.backtest_df.resample("M").agg(lambda x: (x+1).prod() - 1).to_period("M")

        if 'returns' not in monthly_returns.columns:
            monthly_returns = monthly_returns.rename(columns={'total_returns': 'returns'})
        
        ff_factor_data = None
        path = r"/Users/16096/Desktop/Projects/stockbacktester/data/factor_data.pkl"
        with open(path, 'rb') as handle:
            ff_factor_data = pickle.load(handle)
            #print('factor data' and ff_factor_data

        merged_df = pd.merge(ff_factor_data, monthly_returns['returns'], left_index=True, right_index=True, how='inner')
        merged_df['returns'] = merged_df['returns']*100

        if (merged_df.empty):
            return "No available FF data"
        merged_df['port_excess'] = (merged_df['returns']).sub(merged_df['RF'])
        merged_df.rename(column={'Mkt-RF': 'Mkt_RF'}, inplace=True, errors=False)
        
        model = smf.ols(formula = 'port_excess ~ Mkt_RF + SMB + HML + RMW + CMA', data = merged_df)
        fit = model.fit()
        adj_r_sq = fit.rsquared_adj
        ann_alpha = np.power(1 + (fit.params["Intercept"]/100), 12) - 1
        self.ff_stats = {
                "adjusted_r_squared": adj_r_sq,
                "HML p value": fit.pvalues["HML"], # high book to market ratio (value stocks) - low (growth stocks)
                "SMB p value": fit.pvalues["SMB"], # small cap - big cap
                "RMW p value": fit.pvalues["RMW"], # High operating profit - low
                "CMA p value": fit.pvalues["CMA"], # Conservative minus agressive
                "HML": fit.params["HML"],
                "SMB": fit.params["SMB"],
                "RMW": fit.params["RMW"],
                "CMA": fit.params["CMA"],
                "Alpha": fit.params["Intercept"]/100,
                "Annual Alpha": '{:f}'.format(ann_alpha),
                "ann_alpha": ann_alpha,
                "Beta": fit.params["Mkt_RF"]
        }
        

    #plot backtest
    def plotBacktest(self):
        #histogram
        plt.hist(self.backtest_df['total_returns'].dropna(), bins = 100, density=False)
        plt.show()

        #returns chart
        self.backtest_df.plot(x='Date', y= 'total_returns')

        #cumulative
        self.backtest_df.plot(x='Date', y='cum_daily_return')
        ax = self.backtest_df.plot(x=f'Date', y='total')
        ax.set_xlim(pd.Timestamp(self.start), pd.Timestamp(self.end))

def getRandomWeights(numstocks):
    weights = np.random.rand(numstocks)
    return (weights/np.sum(weights))
def getPortWeightedReturns(port_ret, weights):
    return port_ret.iloc[:, 0:len(weights)].mul(weights, axis = 1).sum(axis=1)

def getPortWeightedAnnualReturn(port_ret, weights):
    returns = getPortWeightedReturns(port_ret, weights)

    mean_return_daily = np.mean(returns)
    mean_return_annualized = ((1+mean_return_daily)**252) - 1
    return mean_return_annualized

def getPortWeightedVol(port_ret, weights):
    cov_mat = port_ret.cov()

    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
    return port_vol
def convert_to_tuple(date_str):
    date_object = GetData.datetime.datetime.strptime(date_str, '%Y-%m-%d')
    return (date_object.year, date_object.month, date_object.day)
    

stock = GetData.Stonk('MSFT')
stock.addStrat(stock.buyStock(1))

    





