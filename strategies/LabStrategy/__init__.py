from jesse.strategies import Strategy, cached
import jesse.indicators as ta
from jesse import utils
from jesse import research
from pprint import pprint
from pykalman import KalmanFilter
import numpy as np
from datetime import datetime, timedelta
import statsmodels.api as sm

# timestamp = self.current_candle[0]
# open_price = self.current_candle[1]
# close_price = self.current_candle[2]
# high_price = self.current_candle[3]
# low_price = self.current_candle[4]
# volume = self.current_candle[5]

# now = datetime.utcfromtimestamp(self.time / 1e3)
# start = (now-timedelta(days=60)).strftime('%Y-%m-%d')
# end = now.strftime('%Y-%m-%d')
# print(now, start, end, self.symbol)
# print(self.id)
# if self.symbol == 'BTC-USDT':
#     print(self.get_candles('Binance Spot', 'ETH-USDT', '1h').shape, 'BTC')
# else:
#     print(self.get_candles('Binance Spot', 'BTC-USDT', '1h').shape, 'ETH')


class LabStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.exchange = 'Binance Perpetual Futures'
        self.kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, 
                  initial_state_mean=np.ones(2),
                  initial_state_covariance=np.ones((2, 2)),
                  transition_matrices=np.eye(2),
                  observation_covariance=0.5,
                  transition_covariance=0.000001 * np.eye(2),
                  random_state=np.random.RandomState(0))
        self.state_mean = np.ones(2)
        self.state_cov = np.ones((2, 2))
        self.bb_window = 24*7
        self.spread = np.zeros(self.bb_window-1)

    def before(self):

        # get data
        self.comp = self.get_candles(self.exchange, 'COMP-USDT', '1h')
        self.aave = self.get_candles(self.exchange, 'AAVE-USDT', '1h')

        # update Kalman filtering states
        self.state_mean, self.state_cov = self.kf.filter_update(
            filtered_state_mean=self.state_mean,
            filtered_state_covariance=self.state_cov,
            observation=self.comp[-1, 2], # COMP
            observation_matrix=sm.add_constant(self.aave[-1:, 2], prepend=False, has_constant='add')
        )
        
        spread = self.comp[-1, 2] - self.aave[-1, 2] * self.state_mean[0] - self.state_mean[1]
        self.spread = np.append(self.spread, spread)

    def spread_bb_gt(self) -> bool:
        bb_mean = self.spread[-self.bb_window:].mean()
        bb_std = self.spread[-self.bb_window:].std()
        bb_margin = 2
        # if '2021' in datetime.utcfromtimestamp(self.time / 1e3).strftime('%Y-%m-%d'):
        #     bb_mean = 0
        if self.spread[-1] > bb_mean+bb_std*bb_margin:
            return True
        else:
            return False
    
    def spread_bb_lt(self) -> bool:
        bb_mean = self.spread[-self.bb_window:].mean()
        bb_std = self.spread[-self.bb_window:].std()
        bb_margin = 2
        # if '2021' in datetime.utcfromtimestamp(self.time / 1e3).strftime('%Y-%m-%d'):
        #     bb_margin = 3
        if self.spread[-1] < bb_mean-bb_std*bb_margin:
            return True
        else:
            return False

    def should_long(self) -> bool:
       
        if self.spread_bb_lt() and 'COMP' in self.symbol:
            return True
        if self.spread_bb_gt() and 'AAVE' in self.symbol:
            return True

        return False

    def go_long(self):
        
        hedge_ratio = 1
        # if 'AAVE' in self.symbol:
        #     hedge_ratio = self.state_mean[0]
        qty = hedge_ratio * self.balance/2
        if qty > self.balance:
            qty = self.balance
        qty /= self.price
        self.buy = qty, self.price

    def update_position(self):
        bb_mean = 0 # self.spread[-self.bb_window:].mean()
        if (self.spread[-1]-bb_mean) * (self.spread[-2]-bb_mean) < 0:
            self.liquidate()

    def should_short(self) -> bool:
        
        if self.spread_bb_gt() and 'COMP' in self.symbol:
            return True
        if self.spread_bb_lt() and 'AAVE' in self.symbol:
            return True

        return False

    def go_short(self):
        
        hedge_ratio = 1
        # if 'AAVE' in self.symbol:
        #     hedge_ratio = self.state_mean[0]
        qty = hedge_ratio * self.balance/2
        if qty > self.balance:
            qty = self.balance
        qty /= self.price
        self.sell = qty, self.price

    def should_cancel_entry(self) -> bool:
        True
