import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np

PATH = f"{Path(__file__).absolute().parent}"
DATE_FORMAT = "%Y-%m-%d"
RNG = np.random.default_rng(seed=42)
PRICE_MULTIPLE = 2.5
SALES_DATE_COL = "sales_date"
QUANTITY_COL = "quantity"
SIMPLE_DF_COL_NAMES = [SALES_DATE_COL, QUANTITY_COL]
FREQ = "M"


def plot_data():
    with open(f"{PATH}/bnxt_datasets.json") as fp:
        datasets = json.load(fp)["datasets"]
    count = 0
    for dataset in datasets:
        count += 1

        df = get_simple_df_from_dataset(dataset)
        df = fill_missing_dates(df, is_simple=True)
        df = group_dates(df, FREQ)
        df = convert_quantities_to_float(df)

        # plt.plot(df[SALES_DATE_COL], df[QUANTITY_COL])
        # plt.show()

        # from pprint import pprint

        # pprint(df)

    print(count)


def get_simple_df_from_dataset(dataset) -> pd.DataFrame:
    df = pd.DataFrame(columns=SIMPLE_DF_COL_NAMES)
    df = add_txn_rows_to_df(df, dataset)
    df = convert_str_to_dates_in_column(df, SALES_DATE_COL)
    return df


def add_txn_rows_to_df(df, dataset):
    for txn in dataset["transactions"]:
        other = pd.DataFrame(
            [
                [
                    txn["salesDate"],
                    txn["quantity"],
                ]
            ],
            columns=SIMPLE_DF_COL_NAMES,
        )
        df = pd.concat([other, df]).reset_index(drop=True)
    return df


def fill_missing_dates(df: pd.DataFrame, is_simple: bool) -> pd.DataFrame:
    missing_dates = get_missing_dates(df[SALES_DATE_COL])
    for dt in missing_dates:
        if is_simple:
            data = [[dt, 0.0]]
        else:
            data = [[f"fill-{str(uuid.uuid4())}", dt, dt, 0.0, 0.0, 0.0]]
        row = pd.DataFrame(data=data, columns=df.columns)
        df = pd.concat([row, df]).reset_index(drop=True)
    sort_df_by_dates(df)
    return df


def sort_df_by_dates(df: pd.DataFrame):
    df.sort_values(by=SALES_DATE_COL, inplace=True, ignore_index=True)


def get_missing_dates(dates_col: pd.Series) -> List[datetime]:
    start_dt, end_dt = dates_col.min(), dates_col.max()
    date_range = set(get_date_range(start=start_dt, end=end_dt, freq="D"))
    return [dt for dt in sorted(date_range - set(dates_col))]


def group_dates(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = df.copy()
    date_col = df[SALES_DATE_COL]
    df[SALES_DATE_COL] = date_col.apply(lambda dt: get_last_dt(dt, freq))
    df = df.groupby(by=SALES_DATE_COL, as_index=False).sum()
    sort_df_by_dates(df)
    return df


def convert_quantities_to_float(df: pd.DataFrame):
    df[QUANTITY_COL] = pd.to_numeric(df[QUANTITY_COL], downcast="float")
    return df


def generate_data():
    with open(f"{PATH}/dataset_ids.json") as fp:
        dataset_ids = list(json.load(fp)["dataset_ids"])
    datasets = []
    for dataset_id in dataset_ids:
        sample_rate = RNG.integers(low=6, high=10) / 10.0
        cost_level = float(RNG.integers(low=10, high=1000))
        price_level = cost_level * PRICE_MULTIPLE
        init_quantity = float(RNG.integers(low=10, high=40))
        trend = RNG.choice(["negative", "none", "positive"])
        if trend == "none":
            final_quantity = init_quantity
        else:
            trend_multiple = 3.0 * RNG.random() + 1.0
            if trend == "positive":
                final_quantity = init_quantity * trend_multiple
            elif trend == "negative":
                final_quantity = init_quantity / trend_multiple
        add_season = RNG.choice([True, False])
        if add_season:
            season = [RNG.integers(low=1, high=13)]
        else:
            season = []
        datasets.append(
            generate_dataset(
                dataset_id=dataset_id,
                start="2018-01-01",
                end="2023-02-08",
                dt_frmt=DATE_FORMAT,
                sample_rate=sample_rate,
                cost_level=cost_level,
                price_level=price_level,
                init_quantity=init_quantity,
                final_quantity=final_quantity,
                volatility_type="low",
                season=season,
                season_mult=2.0,
            )
        )
    with open(f"{PATH}/bnxt_datasets.json", "w") as fp:
        json.dump(
            obj={
                "tenantId": "bnxt-demo",
                "datasets": datasets,
            },
            fp=fp,
        )


def generate_dataset(
    dataset_id: str,
    start: str,
    end: str,
    dt_frmt: str,
    sample_rate: float,
    cost_level: float,
    price_level: float,
    init_quantity: float,
    final_quantity: float,
    volatility_type: str = "high",
    season: List[int] = [],
    season_mult: float = 2.0,
) -> List:
    """Mocks a dataset DTO for an imaginary product.

    The dataset DTO consists of a dataset ID field and a transactions field.

    The transactions field consists of transaction DTOs, with the following fields:
        - Transaction ID
        - Sales date
        - Requested delivery date
        - Unit cost
        - Unit price
        - Quantity

    The days between the sales dates and requested delivery dates are sampled
        between 0 and 5, and is not subject to change for simplicity.

    Args:
        dataset_id: Unique dataset ID. Make it descriptive of the test.
        start: The date of the first transaction in the dateframe.
        end: The date of the last transaction in the dataframe.
        dt_frmt: The date format.
        sample_rate: The probability that a given date between the start and end
            dates will have a transaction in it.
        cost_level: The cost level used in the sampling.
        price_level: The price level used in the sampling.
        init_quantity: The quantity the generator uses for sampling in the beginning
            of the data generation.
        final_quantity: The quantity the generator uses for sampling in the end of
            the data generation.
        volatility_type: How much volatility the sampled data should contain. One of
            'high', 'medium', or 'low'. If none is provided, the function will default
            to 'high'.
        season: Which months that are in season, i.e. have higher quantities than other
            months. The functionality only covers the list containing integers between
            1 and 12.
        season_mult: The factor to multiply the quantities of the dates that are
            in season with.

    Returns:
        Return value. The dataframe following the refinery bucket format.
    """

    # Set the date columns
    start, end = datetime.strptime(start, dt_frmt), datetime.strptime(end, dt_frmt)
    date_range = get_date_range(start, end, "D")
    sales_dates = [d for d in date_range if RNG.random() < sample_rate]
    if sales_dates[0] != date_range[0]:
        sales_dates.insert(0, date_range[0])
    if sales_dates[-1] != date_range[-1]:
        sales_dates.append(date_range[-1])
    delivery_dates = [d + timedelta(int(RNG.random() * 5)) for d in sales_dates]

    # Set transaction ID, costs, and price columns
    nbr_dates = len(sales_dates)
    tx_ids = [str(uuid.uuid4()) for _ in range(nbr_dates)]
    cs = [cost_level for _ in range(nbr_dates)]
    ps = [price_level for _ in range(nbr_dates)]

    # Set the quantity column
    growth_rate = final_quantity ** (1 / nbr_dates) / init_quantity ** (1 / nbr_dates)
    a, b = get_ab(volatility_type)
    qs = [
        ((b - a) * RNG.random() + a)
        * (init_quantity * season_mult if is_season(dt, season) else init_quantity)
        * growth_rate**i
        for i, dt in enumerate(sales_dates)
    ]
    qs = [float(int(q)) for q in qs]

    # Set dates to strings as this is the format defined in the DTOs
    sales_dates = [get_str_from_date(d) for d in sales_dates]
    delivery_dates = [get_str_from_date(d) for d in delivery_dates]

    # Define the DTOs
    transactions = []
    for tx_id, sd, dd, c, p, q in zip(tx_ids, sales_dates, delivery_dates, cs, ps, qs):
        transactions.append(
            {
                "transactionId": tx_id,
                "requestedDeliveryDate": dd,
                "salesDate": sd,
                "quantity": q,
                "unitPrice": p,
                "unitCost": c,
            }
        )

    return {"datasetId": dataset_id, "transactions": transactions}


def get_ab(volatility_type: str) -> Tuple[float, float]:
    if volatility_type == "high":
        return 0.0, 1.0
    elif volatility_type == "medium":
        return 0.25, 0.75
    elif volatility_type == "low":
        return 0.4, 0.6
    else:
        return 0.0, 1.0


def is_season(dt: datetime, season: List[int]) -> bool:
    return dt.month in season


def get_date_range(start: datetime, end: datetime, freq: str) -> List[datetime]:
    """Generates a list of consecutive datetimes between the start and end datetime.

    For frequencies 'W', 'M', and 'Y', the date of the last day in the period
        is used.

    Args:
        start: The start datetime object.
        end: The end datetime object.
        freq: The frequency of the time series data the datetime object belongs to.
            One of 'D', 'W', 'M', 'Y'.

    Returns:
        Return value. A sorted list of consecutive datetime objects.
    """
    delta = end - start
    s = set(get_last_dt(start + timedelta(days=i), freq) for i in range(delta.days + 1))
    dates = list(s)
    dates.sort()
    return dates


def get_last_dt(dt: datetime, freq: str) -> datetime:
    """Finds the last date of the given datetime object within the defined frequency.

    Example: Given the frequency 'M' and the date 2022-05-11, the function will
        return the date 2022-05-31 as a datetime object.

    Args:
        dt: The datetime object.
        freq: The frequency of the time series data the datetime object belongs to.
            One of 'D', 'W', 'M', 'Y'.

    Returns:
        Return value. The datetime object of the last date within the defined frequency.
    """
    if freq == "D":
        return dt
    elif freq == "W":
        return get_last_day_of_week(dt)
    elif freq == "M":
        return get_last_day_of_month(dt)
    elif freq == "Y":
        return get_last_day_of_year(dt)


def get_last_day_of_week(dt: datetime) -> datetime:
    """Finds the last date of the week the provided datetime object is within.

    Args:
        dt: The datetime object.

    Returns:
        Return value. The last date of the week the datetime is within.
    """
    return get_first_day_of_week(dt) + timedelta(days=6)


def get_first_day_of_week(dt: datetime) -> datetime:
    """Finds the first date of the week the provided datetime object is within.

    Args:
        dt: The datetime object.

    Returns:
        Return value. The first date of the week the datetime is within.
    """
    return dt - timedelta(days=dt.weekday())


def get_last_day_of_month(dt: datetime) -> datetime:
    """Finds the last date of the month the provided datetime object is within.

    Args:
        dt: The datetime object.

    Returns:
        Return value. The last date of the month the datetime is within.
    """
    next_month = dt.replace(day=28) + timedelta(days=4)
    return next_month - timedelta(days=next_month.day)


def get_last_day_of_year(dt: datetime) -> datetime:
    """Finds the last date of the year the provided datetime object is within.

    Args:
        dt: The datetime object.

    Returns:
        Return value. The last date of the year the datetime is within.
    """
    return datetime(dt.year, 12, 31)


def get_str_from_date(dt: datetime) -> str:
    """Convert a datetime object to a string.

    Args:
        dt: The datetime object.

    Returns:
        Return value. The date string.
    """
    return dt.strftime(DATE_FORMAT)


def convert_str_to_dates_in_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Converts a column of string dates to datetime dates.

    Note that the string dates must follow the DATE_FORMAT_DAY format specified in
        mlf_sales_trainer.general_utils.constants.

    Args:
        df: The dataframe containing the column of string dates.
        col_name: The name of the column with the string dates.

    Returns:
        Return value. The dataframe with converted dates.
    """
    df[col_name] = df[col_name].apply(lambda x: get_date_from_str(x))
    return df


def get_date_from_str(date_str: str) -> datetime:
    """Convert a date string to a datetime object.

    Args:
        date_str: The date string following the DATE_FORMAT_DAY format specified in
            mlf_sales_trainer.general_utils.constants.

    Returns:
        Return value. The datetime object.
    """
    return datetime.strptime(date_str, DATE_FORMAT)


if __name__ == "__main__":
    generate_data()
    plot_data()
