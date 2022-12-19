from jesse.strategies import Strategy, cached
import jesse.indicators as ta
from jesse import utils
from jesse import research
from pprint import pprint
from pykalman import KalmanFilter
import numpy as np
from datetime import datetime, timedelta
import statsmodels.api as sm
import pandas as pd
import pickle

def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val][0]

class PairsTradingLab(Strategy):
    def __init__(self):
        super().__init__()

        # open "pairs" as a dictionary with key-value as a pair
        # assume the symbol in a given pair is only appear once
        with open('./pairs', 'rb') as f:
            self.pairs = pickle.load(f)
            print(self.pairs)
            self.pairs_set = set(list(self.pairs.keys()) + list(self.pairs.values()))
        self.initialized = False

        # hyperparameters    
        self.exchange = 'Binance Perpetual Futures'
        self.freq = '1h'


    def initialize(self):

        if self.symbol in self.pairs.keys():
            self.pair = self.symbol+'_'+self.pairs[self.symbol]
        elif self.symbol in self.pairs.values():
            self.pair = get_keys_from_value(self.pairs, self.symbol)+'_'+self.symbol
        else:
            self.pair = None
            return

        # if in pairs keys and not initialize, then initialize
        if self.pair not in self.shared_vars:

            # spread initialization
            kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, 
                    initial_state_mean=np.ones(2),
                    initial_state_covariance=np.ones((2, 2)),
                    transition_matrices=np.eye(2),
                    observation_covariance=0.5,
                    transition_covariance=0.000001 * np.eye(2),
                    random_state=np.random.RandomState(0))
            state_mean = np.ones(2)
            state_cov = np.ones((2, 2))
            bb_window = 24*7
            spread = np.zeros(bb_window-1)

            self.shared_vars[self.pair] = {
                'kf': kf,
                'mean': state_mean,
                'cov': state_cov,
                'bb_window': bb_window,
                'spread': spread,
                'updated': False,
            }

        self.initialized = True

    def before(self):
        self.symbol = self.symbol.split('-')[0]

        if self.initialized == False:
            self.initialize()

        if self.pair is None:
            return

        ### lock for first of the pair to enter
        if 'time' not in self.shared_vars[self.pair]:
            self.shared_vars[self.pair]['time'] = self.time
        elif self.time == self.shared_vars[self.pair]['time']:
            return
        else:
            self.shared_vars[self.pair]['time'] = self.time

        self.shared_vars[self.pair]['balance'] = self.balance
        # get data
        a = self.get_candles(self.exchange, self.pair.split('_')[0]+'-USDT', self.freq)
        b = self.get_candles(self.exchange, self.pair.split('_')[1]+'-USDT', self.freq)

        kf = self.shared_vars[self.pair]['kf']
        state_mean = self.shared_vars[self.pair]['mean']
        state_cov = self.shared_vars[self.pair]['cov']
        state_mean, state_cov = kf.filter_update(
            filtered_state_mean=state_mean,
            filtered_state_covariance=state_cov,
            observation=a[-1, 2],
            observation_matrix=sm.add_constant(b[-1:, 2], prepend=False, has_constant='add')
        )
        self.shared_vars[self.pair]['kf'] = kf
        self.shared_vars[self.pair]['mean'] = state_mean
        self.shared_vars[self.pair]['cov'] = state_cov

        spread = a[-1, 2] - b[-1, 2] * state_mean[0] - state_mean[1]
        # print(self.symbol, self.pair, spread)
        self.shared_vars[self.pair]['spread'] = np.append(self.shared_vars[self.pair]['spread'], spread)

        # self.shared_vars[self.pair]['updated'] = True

    def spread_bb_gt(self) -> bool:

        spread = self.shared_vars[self.pair]['spread']
        bb_window = self.shared_vars[self.pair]['bb_window']

        bb_mean = spread[-bb_window:].mean()
        bb_std = spread[-bb_window:].std()
        bb_margin = 2
        if spread[-1] > bb_mean+bb_std*bb_margin:
            return True
        else:
            return False
    
    def spread_bb_lt(self) -> bool:

        spread = self.shared_vars[self.pair]['spread']
        bb_window = self.shared_vars[self.pair]['bb_window']

        bb_mean = spread[-bb_window:].mean()
        bb_std = spread[-bb_window:].std()
        bb_margin = 2

        if spread[-1] < bb_mean-bb_std*bb_margin:
            return True
        else:
            return False

    def should_long(self) -> bool:
        if self.pair is None:
            return False
        if self.spread_bb_lt() and self.symbol in self.pairs.keys():
            return True
        if self.spread_bb_gt() and self.symbol in self.pairs.values():
            return True

        return False

    def go_long(self):

        hedge_ratio = 1
        # if self.symbol in self.pairs.values():
        #     hedge_ratio = self.shared_vars[self.pair]['mean'][0]
        balance = self.shared_vars[self.pair]['balance']
        qty = hedge_ratio * balance/10
        if qty > self.balance:
            qty = self.balance
        qty /= self.price
        self.buy = qty, self.price

    def update_position(self):
        if self.pair is None:
            return
        bb_mean = 0 # self.spread[-self.bb_window:].mean()
        spread = self.shared_vars[self.pair]['spread']
        if (spread[-1]-bb_mean) * (spread[-2]-bb_mean) < 0:
            self.liquidate()

    def should_short(self) -> bool:
        if self.pair is None:
            return False
        if self.spread_bb_gt() and self.symbol in self.pairs.keys():
            return True
        if self.spread_bb_lt() and self.symbol in self.pairs.values():
            return True

        return False

    def go_short(self):

        hedge_ratio = 1
        # if self.symbol in self.pairs.values():
        #     hedge_ratio = self.shared_vars[self.pair]['mean'][0]
        balance = self.shared_vars[self.pair]['balance']
        qty = hedge_ratio * balance/10
        if qty > self.balance:
            qty = self.balance
        qty /= self.price
        self.sell = qty, self.price

    def should_cancel_entry(self) -> bool:
        True

    # def after(self):
    #     self.shared_vars[self.pair]['updated'] = False
