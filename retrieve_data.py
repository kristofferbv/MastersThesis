import random
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
        # products.loc[:, 'requested_delivery_date'] = pd.to_datetime(products['requested_delivery_date'])
        products = products.copy()
        products['requested_delivery_date'] = pd.to_datetime(products['requested_delivery_date'])
        # products.loc[:, 'requested_delivery_date'] = pd.to_datetime(products['requested_delivery_date'])
        products = products.sort_values(by=['requested_delivery_date'])
        first_date = products['requested_delivery_date'].iloc[0]
        last_date = products['requested_delivery_date'].iloc[len(products) - 1]
        difference = (last_date - first_date).days / 7

        if difference < 156:  # want only data with at lest 3 years of data
            continue
        average_demand_interval = difference / len(products)
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
            intermittent_demand.to_csv("intermittent_months.csv")
            lumpy_demand.to_csv("data/lumpy_months.csv")
            smooth_demand.to_csv("data/smooth_months.csv")
            erratic_demand.to_csv("data/erratic_months.csv")
        else:
            intermittent_demand.to_csv("intermittent_weeks.csv")
            lumpy_demand.to_csv("data/lumpy_weeks.csv")
            smooth_demand.to_csv("data/smooth_weeks.csv")
            erratic_demand.to_csv("data/erratic_weeks.csv")
    return intermittent_demand, lumpy_demand, smooth_demand, erratic_demand


def read_products(start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Get number of weeks between start and end dates
    num_weeks = len(pd.date_range(start=start_date, end=end_date, freq="W"))
    df = pd.read_csv("data/erratic_weeks.csv", index_col=0)
    df['requested_delivery_date'] = pd.to_datetime(df['requested_delivery_date'])
    df = df.groupby(["product_hash", pd.Grouper(key="requested_delivery_date", freq="w")])["sales_quantity"].sum()
    df = df.reset_index()
    # filter out the weeks outside the specified date range
    mask = (df["requested_delivery_date"] >= start_date) & (df["requested_delivery_date"] <= end_date)

    # mask = df['requested_delivery_date'].dt.isocalendar().week.between(1, 52, inclusive='both') & df['requested_delivery_date'].dt.year.between(2015, 2018, inclusive='both')
    df = df[mask]
    df.to_csv("testing.csv")
    df = df.groupby("product_hash").filter(lambda x: len(x) >= num_weeks-5)
    value_counts = df['product_hash'].value_counts()
    smallest_counts = value_counts.nsmallest(n=1).iloc[-1]
    # Getting product hash of the products with fewest weeks:
    smallest_indexes = value_counts[value_counts == smallest_counts].index
    # Filter the DataFrame to only include rows for 'product_hash' = product_hash_fewest_weeks
    filtered_df = df[df["product_hash"].isin(smallest_indexes)]
    # Find the first and last date for the first product in the filtered DataFrame
    first_date = filtered_df.loc[filtered_df['product_hash'] == smallest_indexes[0]]["requested_delivery_date"].min()
    last_date = filtered_df.loc[filtered_df['product_hash'] == smallest_indexes[0]]["requested_delivery_date"].max()
    if (len(smallest_indexes)>1):
        # Need to find the product that has the last first date, and the first last date
        for index in smallest_indexes:
            first_date = max(first_date, filtered_df[filtered_df['product_hash'] == index]["requested_delivery_date"].min())
            last_date =  min(last_date, filtered_df[filtered_df['product_hash'] == index]["requested_delivery_date"].max())
    # Filter so that all product hashes have the same start and end date
    mask = (df["requested_delivery_date"] >= first_date) & (df["requested_delivery_date"] <= last_date)
    df = df[mask]
    # # Choosing 6 random product_hashes and filter by them
    random_product_hashes = random.sample(df.product_hash.unique().tolist(), 6)
    df = df[df["product_hash"].isin(random_product_hashes)]
    products = []
    # Creating a dataframe for each product and Grouping by week and aggregating by sum, and then adding to list.
    for product_hash in random_product_hashes:
        product_df = df.loc[df['product_hash'] == product_hash]
        product_df = product_df.set_index("requested_delivery_date")
        print(product_df)
        products.append(product_df)
    df = pd.concat(products)
    # df.to_csv("jada.csv")
    return products


def main():
    categorize_products("data/sales_orders.csv", "w", True)
# main()
