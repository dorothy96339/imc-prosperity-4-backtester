def has_day_data(file_reader: FileReader, round_num: int, day_num: int) -> bool:
    for ext in ["csv", "parquet"]:
        with file_reader.file([f"prices_round_{round_num}_day_{day_num}.{ext}"]) as file:
            if file is not None:
                return True
    return False


def read_day_data(file_reader: FileReader, round_num: int, day_num: int, no_names: bool) -> BacktestData:

    # ---------------------------------------------------------------- #
    # Prices                                                            #
    # ---------------------------------------------------------------- #
    prices = []
    prices_file = None

    for ext in ["csv", "parquet"]:
        with file_reader.file([f"prices_round_{round_num}_day_{day_num}.{ext}"]) as file:
            if file is not None:
                prices_file = (file, ext)
                break

    if prices_file is None:
        raise ValueError(f"Prices data is not available for round {round_num} day {day_num}")

    file, ext = prices_file
    if ext == "csv":
        for line in file.read_text(encoding="utf-8").splitlines()[1:]:
            columns = line.split(";")
            prices.append(
                PriceRow(
                    day=int(columns[0]),
                    timestamp=int(columns[1]),
                    product=columns[2],
                    bid_prices=get_column_values(columns, [3, 5, 7]),
                    bid_volumes=get_column_values(columns, [4, 6, 8]),
                    ask_prices=get_column_values(columns, [9, 11, 13]),
                    ask_volumes=get_column_values(columns, [10, 12, 14]),
                    mid_price=float(columns[15]),
                    profit_loss=float(columns[16]),
                )
            )
    else:
        import pandas as pd
        df = pd.read_parquet(file)
        for _, r in df.iterrows():
            prices.append(PriceRow(
                day=int(r["day"]),
                timestamp=int(r["timestamp"]),
                product=r["product"],
                bid_prices=[int(r[c]) for c in ["bid_price_1","bid_price_2","bid_price_3"] if pd.notna(r.get(c, float("nan")))],
                bid_volumes=[int(r[c]) for c in ["bid_volume_1","bid_volume_2","bid_volume_3"] if pd.notna(r.get(c, float("nan")))],
                ask_prices=[int(r[c]) for c in ["ask_price_1","ask_price_2","ask_price_3"] if pd.notna(r.get(c, float("nan")))],
                ask_volumes=[int(r[c]) for c in ["ask_volume_1","ask_volume_2","ask_volume_3"] if pd.notna(r.get(c, float("nan")))],
                mid_price=float(r["mid_price"]),
                profit_loss=float(r["profit_loss"]),
            ))

    # ---------------------------------------------------------------- #
    # Trades                                                            #
    # ---------------------------------------------------------------- #
    trades = []
    for ext in ["csv", "parquet"]:
        with file_reader.file([f"trades_round_{round_num}_day_{day_num}.{ext}"]) as file:
            if file is None:
                continue
            if ext == "csv":
                for line in file.read_text(encoding="utf-8").splitlines()[1:]:
                    columns = line.split(";")
                    trades.append(Trade(
                        symbol=columns[3],
                        price=int(float(columns[5])),
                        quantity=int(columns[6]),
                        buyer=columns[1],
                        seller=columns[2],
                        timestamp=int(columns[0]),
                    ))
            else:
                import pandas as pd
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
            break

    # ---------------------------------------------------------------- #
    # Observations                                                      #
    # ---------------------------------------------------------------- #
    observations = []
    for ext in ["csv", "parquet"]:
        with file_reader.file([f"observations_round_{round_num}_day_{day_num}.{ext}"]) as file:
            if file is None:
                continue
            if ext == "csv":
                for line in file.read_text(encoding="utf-8").splitlines()[1:]:
                    columns = line.split(",")
                    observations.append(ObservationRow(
                        timestamp=int(columns[0]),
                        bidPrice=float(columns[1]),
                        askPrice=float(columns[2]),
                        transportFees=float(columns[3]),
                        exportTariff=float(columns[4]),
                        importTariff=float(columns[5]),
                        sugarPrice=float(columns[6]),
                        sunlightIndex=float(columns[7]),
                    ))
            else:
                import pandas as pd
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
            break

    return create_backtest_data(round_num, day_num, prices, trades, observations)
