import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
from sklearn.linear_model import LinearRegression
import numpy

# from mgarch_dcc_r import arch_test


matplotlib.use('TkAgg')



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

def main():
    categorize_products("data/sales_orders.csv", "w", True)
main()
