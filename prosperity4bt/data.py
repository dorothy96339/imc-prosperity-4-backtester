from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
import pandas as pd

from prosperity4bt.datamodel import Symbol, Trade
from prosperity4bt.file_reader import FileReader

DEFAULT_POSITION_LIMIT = 10

LIMITS: dict[str, int] = {
    "EMERALDS": 80,
    "TOMATOES": 80,
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    "VEV_4000": 300,
    "VEV_4500": 300,
    "VEV_5000": 300,
    "VEV_5100": 300,
    "VEV_5200": 300,
    "VEV_5300": 300,
    "VEV_5400": 300,
    "VEV_5500": 300,
    "VEV_6000": 300,
    "VEV_6500": 300,
}


def get_position_limit(symbol: str, overrides: Optional[dict[str, int]] = None) -> int:
    if overrides is not None and symbol in overrides:
        return overrides[symbol]
    return LIMITS.get(symbol, DEFAULT_POSITION_LIMIT)


@dataclass
class PriceRow:
    day: int
    timestamp: int
    product: Symbol
    bid_prices: list[int]
    bid_volumes: list[int]
    ask_prices: list[int]
    ask_volumes: list[int]
    mid_price: float
    profit_loss: float


def get_column_values(columns: list[str], indices: list[int]) -> list[int]:
    values = []
    for index in indices:
        value = columns[index]
        if value == "":
            break
        values.append(int(value))
    return values


@dataclass
class ObservationRow:
    timestamp: int
    bidPrice: float
    askPrice: float
    transportFees: float
    exportTariff: float
    importTariff: float
    sugarPrice: float
    sunlightIndex: float


@dataclass
class BacktestData:
    round_num: int
    day_num: int
    prices: dict[int, dict[Symbol, PriceRow]]
    trades: dict[int, dict[Symbol, list[Trade]]]
    observations: dict[int, ObservationRow]
    products: list[Symbol]
    profit_loss: dict[Symbol, float]


def create_backtest_data(
    round_num: int, day_num: int, prices: list[PriceRow], trades: list[Trade], observations: list[ObservationRow]
) -> BacktestData:
    prices_by_timestamp: dict[int, dict[Symbol, PriceRow]] = defaultdict(dict)
    for row in prices:
        prices_by_timestamp[row.timestamp][row.product] = row

    trades_by_timestamp: dict[int, dict[Symbol, list[Trade]]] = defaultdict(lambda: defaultdict(list))
    for trade in trades:
        trades_by_timestamp[trade.timestamp][trade.symbol].append(trade)

    products = sorted(set(row.product for row in prices))
    profit_loss = {product: 0.0 for product in products}
    observations_by_timestamp = {row.timestamp: row for row in observations}

    return BacktestData(
        round_num=round_num,
        day_num=day_num,
        prices=prices_by_timestamp,
        trades=trades_by_timestamp,
        observations=observations_by_timestamp,
        products=products,
        profit_loss=profit_loss,
    )


def has_day_data(file_reader: FileReader, round_num: int, day_num: int) -> bool:
    with file_reader.file([f"round{round_num}", f"prices_round_{round_num}_day_{day_num}.parquet"]) as file:
        return file is not None


def read_day_data(file_reader: FileReader, round_num: int, day_num: int, no_names: bool) -> BacktestData:

    # ---------------------------------------------------------------- #
    # Prices                                                            #
    # ---------------------------------------------------------------- #
    prices = []
    with file_reader.file([f"round{round_num}", f"prices_round_{round_num}_day_{day_num}.parquet"]) as file:
        if file is None:
            raise ValueError(f"Prices data is not available for round {round_num} day {day_num}")

        df = pd.read_parquet(file)
        for _, r in df.iterrows():
            prices.append(PriceRow(
                day=int(r["day"]),
                timestamp=int(r["timestamp"]),
                product=r["product"],
                bid_prices=[int(r[c]) for c in ["bid_price_1", "bid_price_2", "bid_price_3"] if pd.notna(r.get(c, float("nan"))) and r.get(c) != ""],
                bid_volumes=[int(r[c]) for c in ["bid_volume_1", "bid_volume_2", "bid_volume_3"] if pd.notna(r.get(c, float("nan"))) and r.get(c) != ""],
                ask_prices=[int(r[c]) for c in ["ask_price_1", "ask_price_2", "ask_price_3"] if pd.notna(r.get(c, float("nan"))) and r.get(c) != ""],
                ask_volumes=[int(r[c]) for c in ["ask_volume_1", "ask_volume_2", "ask_volume_3"] if pd.notna(r.get(c, float("nan"))) and r.get(c) != ""],
                mid_price=float(r["mid_price"]),
                profit_loss=float(r["profit_loss"]),
            ))

    # ---------------------------------------------------------------- #
    # Trades                                                            #
    # ---------------------------------------------------------------- #
    trades = []
    with file_reader.file([f"round{round_num}", f"trades_round_{round_num}_day_{day_num}.parquet"]) as file:
        if file is not None:
            df = pd.read_parquet(file)
            for _, r in df.iterrows():
                trades.append(Trade(
                    symbol=r["symbol"],
                    price=int(float(r["price"])),
                    quantity=int(r["quantity"]),
                    buyer=r["buyer"],
                    seller=r["seller"],
                    timestamp=int(r["timestamp"]),
                ))

    # ---------------------------------------------------------------- #
    # Observations                                                      #
    # ---------------------------------------------------------------- #
    observations = []
    with file_reader.file([f"round{round_num}", f"observations_round_{round_num}_day_{day_num}.parquet"]) as file:
        if file is not None:
            df = pd.read_parquet(file)
            for _, r in df.iterrows():
                observations.append(ObservationRow(
                    timestamp=int(r["timestamp"]),
                    bidPrice=float(r["bidPrice"]),
                    askPrice=float(r["askPrice"]),
                    transportFees=float(r["transportFees"]),
                    exportTariff=float(r["exportTariff"]),
                    importTariff=float(r["importTariff"]),
                    sugarPrice=float(r["sugarPrice"]),
                    sunlightIndex=float(r["sunlightIndex"]),
                ))

    return create_backtest_data(round_num, day_num, prices, trades, observations)
