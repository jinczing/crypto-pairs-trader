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
from sklearn import linear_model
from sklearn.manifold import TSNE
from sklearn import linear_model
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA
import statsmodels.api as sm
import itertools
from itertools import product
from collections import Counter

def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val][0]

def reg(x,y):
    regr = linear_model.LinearRegression()
    x_constant = pd.concat([x], axis=1)
    regr.fit(x_constant, y)    
    beta = regr.coef_[0]
    alpha = regr.intercept_
    spread = y - x*beta - alpha
    return spread

class AutoPairsTrading(Strategy):
    def __init__(self):
        super().__init__()

        # hyperparameters    
        self.exchange = 'Binance Perpetual Futures'
        self.freq = '1h'
        self.start_time = '2021-06-01'
        self.windows = []
        for i in range(1, 181, 10):
            self.windows.append(24*i)
        # self.windows = [24, 24*2, 24*7, 24*14, 24*30, 24*90, 24*180, 24*360]
        self.max_pairs = 3
        self.period = 30

        # self.initialized = False
        if 'initialized' not in self.shared_vars:
            self.shared_vars['initialized'] = {}

        # initialize if pairs does not exist (the first instantiated strategy)
        if 'pairs' not in self.shared_vars:
            self.reevaluate_pairs()
        else:
            self.pairs = self.shared_vars['pairs']

    def reevaluate_pairs(self):
        with open('universe_symbols', 'rb') as f:
            universe = pickle.load(f)
        d = {}
        for symbol in universe:
            self.shared_vars['initialized'][symbol] = False
            cs = []
            _d = pd.Series(self.get_candles(self.exchange, symbol+'-USDT', self.freq)[:, 2])
            _d = _d.diff(24)/_d.shift(24)
            # print(_d)
            for window in self.windows:
                cs.append(_d.rolling(window).mean().ffill().fillna(0))
            d[symbol] = pd.concat(cs, axis=1).iloc[-1:, :]
            # print(d[symbol].shape)
        sample = []
        for k, v in d.items():
            v.index = [k]
            sample.append(v)
        sample = pd.concat(sample)
        sample = (sample-sample.mean())/sample.std()
        # print(sample)
        sample.fillna(0, inplace=True)
        pca = PCA(svd_solver='full', n_components=min(len(self.windows), len(universe)))
        sample_pca = pca.fit_transform(sample)

        clusters = 6
        model = AgglomerativeClustering(n_clusters=clusters, affinity='euclidean', linkage='complete')
        model.fit(sample_pca)
        labels = model.labels_
        labels = np.zeros_like(labels) # all in the same class
        # print(labels)
        label_counter = Counter(labels)

        adfs = []
        pairs = []
        for k, v in label_counter.items():
            if v<2:
                continue
            symbols = sample.index[labels==k]
            for symbol1, symbol2, in itertools.combinations(symbols, 2):
                a = pd.Series(self.get_candles(self.exchange, symbol1+'-USDT', self.freq)[-self.windows[-1]:, 2])
                b = pd.Series(self.get_candles(self.exchange, symbol2+'-USDT', self.freq)[-self.windows[-1]:, 2])
                # print(a.shape)
                _spread = reg(a, b)
                adf = sm.tsa.stattools.adfuller(_spread, maxlag=1)
                adfs.append(adf[1])
                pairs.append((symbol1, symbol2))
        # print(adfs)
        counter = 0
        unique = set()
        selected_pairs = {}
        for i in np.argsort(adfs):
            if pairs[i][0] in unique or pairs[i][1] in unique:
                continue
            counter += 1
            unique.add(pairs[i][0])
            unique.add(pairs[i][1])
            selected_pairs[pairs[i][0]] = pairs[i][1]
            if counter == self.max_pairs:
                break

        self.shared_vars['pairs'] = selected_pairs
        self.pairs = selected_pairs
        
        print(self.pairs)
        

    def initialize(self):

        self.pairs = self.shared_vars['pairs']

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

        # self.initialized = True
        self.shared_vars['initialized'][self.symbol] = True

    def check_time(self):
        if 'last' not in self.shared_vars:
            self.shared_vars['last'] = datetime.utcfromtimestamp(self.time / 1e3)

        now = datetime.utcfromtimestamp(self.time / 1e3)
        last = self.shared_vars['last']
        if now-timedelta(days=self.period)>last:
            print(now, last, 'reevaluate')
            ### lock
            self.shared_vars['last'] = now

            ### reevaluate
            self.reevaluate_pairs()

            ### clear previous computed spread if old pair not exists
            old_pairs = []
            for key in self.shared_vars:
                if '_' in key:
                    old_pairs.append(key.split('_'))
            for key, value in old_pairs:
                if key in self.pairs and value == self.pairs[key]:
                    continue
                del self.shared_vars[key+'_'+value] # force pair to initialize

            


    def before(self):
        self.symbol = self.symbol.split('-')[0]

        self.check_time()

        if self.shared_vars['initialized'][self.symbol] == False:
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
