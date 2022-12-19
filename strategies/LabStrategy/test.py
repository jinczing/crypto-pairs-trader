from jesse.strategies import Strategy, cached
import jesse.indicators as ta
from jesse import utils
from pprint import pprint


class LabStrategy(Strategy):
    def __init__(self):
        super().__init__()
        pprint(self.exchange, self.symbol)
        # pprint(self.get_candles('Binance Spot', 'BTC-USDT', '1h'))

    def should_long(self) -> bool:
        return False

    def go_long(self):
        pass

    def should_short(self) -> bool:
        # For futures trading only
        return False

    def go_short(self):
        # For futures trading only
        pass

    def should_cancel_entry(self) -> bool:
        return True

if __name__ == '__main__':
    s = LabStrategy()

