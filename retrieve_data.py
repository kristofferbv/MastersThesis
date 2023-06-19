import os
import random
from datetime import timedelta

import pandas as pd
import matplotlib


# from mgarch_dcc_r import arch_test


def categorize_products(file_name, time_interval, should_write_to_file):
    """
    This read data from a .csv file and categorize the products into smooth, lumpy, erratic and intermittent, based on the granularity (time_interval)
    :param file_name: The name of the .csv file containing the data
    :param time_interval: Could be either 'm' for months or 'w' for weeks
    :param should_write_to_file: Determine weather or not the new categories should be written to file
    :return: returns the four product categories
    """
    if time_interval != "w" and time_interval != "m":
        raise ValueError("Time_interval parameter must be 'm' or 'w'")
    data_frame = pd.read_csv(file_name, index_col=0)

    # Defining the date range
    date_range = pd.date_range(start='2016-01-01', end='2020-12-30')

    # Creating data_frames for the different categories
    intermittent_demand = pd.DataFrame(columns=data_frame.columns)
    smooth_demand = pd.DataFrame(columns=data_frame.columns)
    erratic_demand = pd.DataFrame(columns=data_frame.columns)
    lumpy_demand = pd.DataFrame(columns=data_frame.columns)

    # Sorting products by product hash
    sort_by_product = data_frame.groupby('product_hash')['sales_quantity'].sum().sort_values(ascending=False).reset_index()
    count = 0
    count_intermittent = 0
    count_smooth = 0
    count_erratic = 0
    count_lumpy = 0
    number_of_products = len(sort_by_product)

    print(number_of_products)
    for _, product in sort_by_product.iterrows():
        if count % 1000 == 0:
            print("progress: " + str(count) + "/" + str(number_of_products))
        count += 1
        # Getting all products in dataFrame with same product hash
        products = data_frame.loc[data_frame['product_hash'] == product['product_hash']]
        products = products.copy()
        products['requested_delivery_date'] = pd.to_datetime(products['requested_delivery_date'])

        # Only consider products within the defined date range
        products = products[products['requested_delivery_date'].isin(date_range)]

        products = products.sort_values(by=['requested_delivery_date'])
        if products.empty:  # Check if the DataFrame is empty
            continue  # Skip the current iteration
        first_date = products['requested_delivery_date'].iloc[0]
        last_date = products['requested_delivery_date'].iloc[len(products) - 1]
        difference = (last_date - first_date).days / 7

        if difference < 156:  # want only data with at least 3 years of data
            continue
        if (len(products.loc[products['sales_quantity'] > 0, 'requested_delivery_date'].unique())) > 0:
            average_demand_interval = difference / len(products.loc[products['sales_quantity'] > 0, 'requested_delivery_date'].unique())
        else:
            average_demand_interval = 0
        cv_squared = (products["sales_quantity"].std() / products["sales_quantity"].mean()) ** 2
        if average_demand_interval == 0:  # means we only have one occurrence of the product
            continue
        if average_demand_interval > 1.32:
            if cv_squared >= 0.49:
                count_lumpy += 1
                lumpy_demand = pd.concat([lumpy_demand, products])
            else:
                count_intermittent += 1
                intermittent_demand = pd.concat([intermittent_demand, products])
        else:
            if cv_squared >= 0.49:
                count_erratic += 1
                erratic_demand = pd.concat([erratic_demand, products])
            else:
                count_smooth += 1
                smooth_demand = pd.concat([smooth_demand, products])

    print("number of intermittent products: ")
    print(count_intermittent)
    print("number of smooth products: ")
    print(count_smooth)
    print("number of erratic products: ")
    print(count_erratic)
    if should_write_to_file:
        if time_interval == "m":
            # 'months' in the filename indicates that the demand interval is in months
            intermittent_demand.to_csv("data/intermittent_months.csv")
            lumpy_demand.to_csv("data/lumpy_months.csv")
            smooth_demand.to_csv("data/smooth_months.csv")
            erratic_demand.to_csv("data/erratic_months.csv")
        else:
            intermittent_demand.to_csv("data/intermittent_weeks.csv")
            lumpy_demand.to_csv("data/lumpy_weeks.csv")
            smooth_demand.to_csv("data/smooth_weeks.csv")
            erratic_demand.to_csv("data/erratic_weeks.csv")
    return intermittent_demand, lumpy_demand, smooth_demand, erratic_demand

def read_products_with_hashes_2(start_date, end_date, product_hashes, category = "erratic", frequency = "weeks"):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # construct the path to the file
    file_path = os.path.join(current_dir, f"data/{category}_{frequency}.csv")
    df = pd.read_csv(file_path)
    df['requested_delivery_date'] = pd.to_datetime(df['requested_delivery_date'])
    # Filter out the weeks outside the specified date range
    df = df[(df['requested_delivery_date'] >= start_date) & (df['requested_delivery_date'] <= end_date)]
    # Filter the DataFrame using the provided product_hashes
    df = df[df["product_hash"].isin(product_hashes)]

    # Group by product_hash and requested_delivery_date (week) and sum the sales_quantity
    df = df.groupby(["product_hash", pd.Grouper(key="requested_delivery_date", freq="w")]).agg({"sales_quantity": "sum", "unit_price": "mean"})
    # Calculate the average unit_price for each product_hash
    df['average_unit_price'] = df.groupby('product_hash')['unit_price'].transform('mean')

    # If you want to keep only the new average_unit_price column and drop the original unit_price column
    df = df.drop(columns=['unit_price'])

    df = df.reset_index()


    products = []
    for product_hash in product_hashes:
        product_df = df.loc[df['product_hash'] == product_hash]
        product_df = product_df.set_index("requested_delivery_date")
        # Resample with weekly frequency and fill missing weeks with 0
        product_df = product_df.resample("W").asfreq(fill_value=0)

        products.append(product_df)

    return products



def read_products_with_hashes(start_date, end_date, product_hashes):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # construct the path to the file
    file_path = os.path.join(current_dir, f"data/erratic_weeks.csv")
    df = pd.read_csv(file_path, index_col=0)
    df['requested_delivery_date'] = pd.to_datetime(df['requested_delivery_date'])

    # Filter the DataFrame using the provided product_hashes
    df = df[df["product_hash"].isin(product_hashes)]

    df = df.groupby(["product_hash", pd.Grouper(key="requested_delivery_date", freq="w")]).agg({"sales_quantity": "sum", "unit_price": "mean"})
    # Calculate the average unit_price for each product_hash
    df['average_unit_price'] = df.groupby('product_hash')['unit_price'].transform('mean')

    # If you want to keep only the new average_unit_price column and drop the original unit_price column
    df = df.drop(columns=['unit_price'])

    df = df.reset_index()

    # Filter out the weeks outside the specified date range
    mask = (df["requested_delivery_date"] >= start_date) & (df["requested_delivery_date"] <= end_date)
    df = df[mask]

    products = []
    for product_hash in product_hashes:
        product_df = df[df['product_hash'] == product_hash].set_index('requested_delivery_date')
        # Resample with weekly frequency and fill missing weeks with 0
        product_df = product_df.resample("W").asfreq(fill_value=0)
        products.append(product_df)

        products.append(product_df)

    return products


def read_products(start_date, end_date, category="erratic", frequency="weeks"):
    """
    This function is an updated version of read_products, where also products that has some weeks without any data is included.
    Args:
        start_date:
        end_date:

    Returns: list of products (as dataframes) that has data from start_date to end_date

    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # construct the path to the file
    try:
        file_path = os.path.join(current_dir, f"data/{category}_{frequency}.csv")
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
    except:
        print(f"not able to read file with file-directory: data/{category}_{frequency}.csv")
        return

    # Convert the 'requested_delivery_date' column to a datetime object
    df['requested_delivery_date'] = pd.to_datetime(df['requested_delivery_date'])

    # Filter out the rows that fall outside the date range
    df = df[(df['requested_delivery_date'] >= start_date) & (df['requested_delivery_date'] <= end_date)]

    # Group by product_hash and requested_delivery_date (week) and sum the sales_quantity
    df = df.groupby(["product_hash", pd.Grouper(key="requested_delivery_date", freq="w")]).agg({"sales_quantity": "sum", "unit_price": "mean"})
    # Calculate the average unit_price for each product_hash
    df['average_unit_price'] = df.groupby('product_hash')['unit_price'].transform('mean')

    # If you want to keep only the new average_unit_price column and drop the original unit_price column
    df = df.drop(columns=['unit_price'])

    df = df.reset_index()

    # Define the time periods
    january_2016 = pd.to_datetime(start_date)
    february_2016 = pd.to_datetime(start_date + timedelta(days=30))
    december_2020 = pd.to_datetime(end_date)
    january_2021 = pd.to_datetime(end_date + timedelta(days = 30))


    # Filter the DataFrame based on the time periods
    january_2016_data = df[(df['requested_delivery_date'] >= january_2016) &
                           (df['requested_delivery_date'] < february_2016)]

    december_2020_data = df[(df['requested_delivery_date'] >= december_2020) &
                            (df['requested_delivery_date'] < january_2021)]

    # Identify the product hashes that have data in both periods
    product_hashes_in_both_periods = set(january_2016_data['product_hash'].unique()) & set(december_2020_data['product_hash'].unique())

    # Filter the original DataFrame to include only these product hashes
    df = df[df['product_hash'].isin(product_hashes_in_both_periods)]

    unique_product_hashes = df['product_hash'].unique()

    products = []
    for product_hash in unique_product_hashes:
        product_df = df[df['product_hash'] == product_hash].set_index('requested_delivery_date')
        # Resample with weekly frequency and fill missing weeks with 0
        product_df = product_df.resample("W").asfreq(fill_value=0)
        products.append(product_df)
    return products

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # construct the path to the file
    file_path = os.path.join(current_dir, "data/sales_orders.csv")
    categorize_products(file_path, "w", True)
# main()
