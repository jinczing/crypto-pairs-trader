# Cryptocurrency Pairs Trading via Unsupervised Learning

## Methodology
In this project, we propose an unsupervised learning based approach for pairs selection in cryptocurrency perpetual futures market. <br/> <br/>
We first use dimension reduction and clustering algorithm to bundle assets in to each group.
Then, we use ADF test to select top cointegrated pairs from the same group. <br/> <br/>
The result shows that our strategy is superior to pure cointegration testing strategy in terms of PnL (Profit and Loss) and Sharpe ratio. <br/> <br/>
See more at this [report](https://www.notion.so/Cryptocurrency-Pairs-Trading-via-Unsupervised-Learning-140a1a27de774f19847eba8ee200ffde).
## Implementation
Our backtesting framework is [Jesse Trade](https://jesse.trade/). The implementation of our strategy can be found in `strategies/AutoPairsTrading`.
## Results
##### Naitve Cointegration-based Pairs Selection
![Untitled](https://raw.githubusercontent.com/jinczing/crypto-pairs-trader/master/naive.png)
Sharpe Ratio: 0.47 Annualized Return: 8%
##### Clustering-based Pairs Selection
![Untitled](https://raw.githubusercontent.com/jinczing/crypto-pairs-trader/master/clustering.png)
Sharpe Ratio: 1.89 Annualized Return: 50.44%
