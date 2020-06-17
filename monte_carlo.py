import pandas as pd
import pandas_datareader.data as pdr
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from scipy import stats

style.use('fivethirtyeight')

start = dt.datetime(2020,1,1)
end = dt.datetime.now()


prices = pdr.DataReader('SPY', 'yahoo', start, end)['Adj Close']

returns = prices.pct_change()

last_price = prices[-1]

num_trials = 1000
num_days = 252

sim_df = pd.DataFrame()

for i in range(num_trials):
    ct = 0
    daily_vol = returns.std()
    
    price_series = []
    
    price = last_price * (1 + np.random.normal(0, daily_vol))
    price_series.append(price)
    
    for p in range(num_days):
        if ct == 251:
            break
        price = price_series[ct] * (1 + np.random.normal(0, daily_vol))
        price_series.append(price)
        
        ct += 1
        
    sim_df[i] = price_series
    
sim_df

fig = plt.figure()
fig.suptitle('Monte Carlo for SPY')
plt.plot(sim_df)
plt.axhline(y = last_price, c='r', linestyle = '-')
plt.xlabel('Day')
plt.ylabel('Price')
plt.show()


avg_price = sim_df.mean(axis=1)

plt.plot(avg_price)


plt.hist(sim_df.std())

sim_stats = pd.DataFrame({'Mean':sim_df.mean(),
                          'Std':sim_df.std()})

plt.scatter(sim_stats['Mean'], sim_stats['Std'])

start2 = dt.datetime(1990, 1, 1)

spy_d = pdr.DataReader('SPY', 'yahoo-dividends', start2, end)

spy_d


plt.scatter(spy_d.index, spy_d['value'])

spy_d['year'] = spy_d.index.astype(str).str[:4]

divg = spy_d['value'].groupby(spy_d['year']).mean()

plt.scatter(divg.index, divg)

np.poly1d(divg.index, divg)

type(divg[0])

x = divg.index.astype(int)
y = np.array(divg)

slope, intercept, r, p, stderr = stats.linregress(x,y)

z = np.arange(2050)

y2 = slope*z + intercept

plt.plot(z,divg)
plt.plot(y2)
plt.ylim([0,2.0])
plt.xlim([1990,2020])
plt.show()
