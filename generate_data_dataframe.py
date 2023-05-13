import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os


def generate_data(num_ids: int, weeks: int = 204):
    """Generate a DataFrame with num_ids unique IDs and random transactions over specified weeks."""
    dates = pd.date_range(end=datetime.datetime.today(), periods=weeks, freq='W').to_pydatetime().tolist()
    data = {'ID': [], 'date': [], 'transaction_amount': []}

    # Create a seasonal pattern with a sine wave
    t = np.linspace(0, 4 * np.pi, weeks)
    seasonal_pattern = 5 + np.sin(t)

    for i in range(num_ids):
        data['ID'].extend([f'ID_{i}'] * weeks)
        data['date'].extend(dates)
        # Add seasonal pattern to random transaction amounts
        data['transaction_amount'].extend(np.random.uniform(0, 5, weeks) + seasonal_pattern)

    df = pd.DataFrame(data)
    df['date'] = df['date'].dt.date  # Convert to just date (no time)
    df.to_csv('transactions.csv', index=False)  # Save DataFrame to file
    return df


def plot_data(df):
    """Plot the transaction amounts over time."""
    df['date'] = pd.to_datetime(df['date'])  # Ensure date column is datetime type
    df_grouped = df.groupby(['date', 'ID']).sum().reset_index()  # Group by date and ID, summing transaction amounts

    plt.figure(figsize=(12, 6))
    for id in df_grouped['ID'].unique():
        plt.plot(df_grouped[df_grouped['ID'] == id]['date'], df_grouped[df_grouped['ID'] == id]['transaction_amount'], label=id)
    plt.xlabel('Date')
    plt.ylabel('Transaction Amount')
    plt.legend()
    plt.show()


# Generate and plot data
df = generate_data(5, 30)
plot_data(df)
