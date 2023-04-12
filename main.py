import deterministic_model as det_mod
import pandas as pd
import random


def simulate(time_steps, products):
    dict_demands = {}
    # initialize model
    deterministic_model = det_mod.DeterministicModel()
    for time in range(time_steps):
        for product_index in range(len(products)):
            dict_demands[product_index] = products[product_index].head(time_steps).tolist()
            # dropping the first row
            products[product_index] = products[product_index].iloc[1:]
        deterministic_model.set_demand_forecast(dict_demands)
        deterministic_model.set_up_model()
        deterministic_model.model.setParam("OutputFlag", 0)
        deterministic_model.optimize()
        for var in deterministic_model.model.getVars():
            print(f"{var.varName}: {var.x:.2f}")


if __name__ == '__main__':
    # deterministic_model = det_mod.DeterministicModel()
    # deterministic_model.set_up_model()
    # deterministic_model.model.optimize()
    # Define start and end dates

    start_date = pd.to_datetime("2015-01-01")
    end_date = pd.to_datetime("2018-01-01")

    # Get number of weeks between start and end dates
    num_weeks = len(pd.date_range(start=start_date, end=end_date, freq="W"))
    df = pd.read_csv("data/erratic_weeks.csv", index_col=0)
    df['requested_delivery_date'] = pd.to_datetime(df['requested_delivery_date'])

    # filter out the weeks outside the specified date range
    mask = df['requested_delivery_date'].dt.isocalendar().week.between(1, 52, inclusive='both') & df['requested_delivery_date'].dt.year.between(2015, 2018, inclusive='both')
    df = df[mask]

    df['week'] = df['requested_delivery_date'].dt.isocalendar().week

    # group the DataFrame by product_hash and count the number of unique weeks in each group
    week_counts = df.groupby('product_hash')['week'].nunique()
    # filter out any groups that don't have a count of weeks equal to the number of weeks between the specified dates
    mask = week_counts.eq(df['week'].nunique())
    df = df.groupby('product_hash').filter(lambda x: mask.loc[x.name])
    # drop the week column
    df = df.drop('week', axis=1)
    # Choosing 6 random product_hashes and filter by them
    random_product_hashes = random.sample(df.product_hash.unique().tolist(), 6)
    df = df[df["product_hash"].isin(random_product_hashes)]
    products = []
    # Creating a dataframe for each product and Grouping by week and aggregating by sum, and then adding to list.
    for product_hash in random_product_hashes:
        product_df = df.loc[df['product_hash'] == product_hash]
        product_df = product_df.set_index("requested_delivery_date")
        product_df = product_df.groupby(pd.Grouper(freq="w"))["sales_quantity"].sum()
        products.append(product_df)


    simulate(13, products)





    """
    Algorithm for simulation optimization: 
        1) Find n products and use historical data to forecast the next k periods at time t = t0
        2) Find current inventory level for each product
        2) Once the forecast and start inventory level is decided, run the optimization algorithm to find the best option
            at time t = t0. 
        3) Store this option! 
        
        4) Now at time t = t0 + 1, use the new available data to create a forecast of the next k periods at time t = t0 + 1
        5) repeat point 2-4 until t = t0 + k
    """
