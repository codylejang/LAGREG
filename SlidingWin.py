import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions
from pypfopt.exceptions import OptimizationError

def get_optimal_portfolio(df_window):

    mu = mean_historical_return(df_window)
    S = CovarianceShrinkage(df_window).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    # ef.add_objective(objective_functions.L2_reg, gamma=0.1)
    risk_free_rate = mu.min() # for now risk_free_rate is the minimum return of the all returns - must be changed not sure how
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    cleaned_weights = ef.clean_weights()

    ticker_list = df_window.columns.tolist()
    var_names = ["Time_Start", "Time_End"] + ticker_list
    w = pd.DataFrame(columns=var_names, index=range(1))
    w["Time_Start"] = df_window.index[0]
    w["Time_End"] = df_window.index[-1]
    w[ticker_list] = list(cleaned_weights.values())

    return w

def get_raw_prices(ticker_list, start_date, end_date):
    url = 'https://api.orats.io/datav2/hist/dailies'
    payload = {
        'token': 'mytoken',
        'ticker': ','.join(ticker_list),
        'tradeDate': start_date + ',' + end_date,
        'fields': 'tradeDate,ticker,clsPx'
    }

    response = requests.get(url, params=payload)
    response_dict = response.json()

    # Extracting 'data' from JSON
    data_list = response_dict['data']

    # Creating DataFrame
    ticker_df = pd.DataFrame(data_list)

    # Reformat
    ticker_df = ticker_df[['tradeDate', 'ticker', 'clsPx']]
    # Convert 'date' column to datetime
    ticker_df['tradeDate'] = pd.to_datetime(ticker_df['tradeDate'])
    ticker_df = ticker_df.reset_index().rename(columns={'tradeDate': 'Time'})
    ticker_df.set_index('Time', inplace=True)
    ticker_df = ticker_df.pivot(columns='ticker', values='clsPx')
    ticker_df.columns.name = None

    return ticker_df


def sliding(ticker_list, start_date, end_date, K):   
    '''
    
    Parameters:
    ticker_list (list of tickers): A list of tickers representing the input data.
    start_date: start date
    end_date: end date
    K (int): The size of the sliding window.
    
    Returns:
    list of int: A list containing the maximum values from each sliding window of size K as it moves through the list arr
    '''
    
    url = 'https://api.orats.io/datav2/hist/dailies'
    payload = {
        'token': 'e0167530-b9b3-4c8b-b854-588c90a69afb',
        'ticker': ','.join(ticker_list),
        'tradeDate': start_date + ',' + end_date,
        'fields': 'tradeDate,ticker,clsPx'
    }

    response = requests.get(url, params=payload)
    response_dict = response.json()

    # Extracting 'data' from JSON
    data_list = response_dict['data']

    # Creating DataFrame
    ticker_df = pd.DataFrame(data_list)

    # Reformat
    ticker_df = ticker_df[['tradeDate', 'ticker', 'clsPx']]
    # Convert 'date' column to datetime
    ticker_df['tradeDate'] = pd.to_datetime(ticker_df['tradeDate'])
    ticker_df.set_index('tradeDate', inplace=True)
    ticker_df = ticker_df.reset_index().rename(columns={'tradeDate': 'Time'}).set_index('Time')
    ticker_df = ticker_df.pivot(columns='ticker', values='clsPx')
    ticker_df.columns.name = None
    
    var_names = ["Time_Start", "Time_End"] + ticker_df.columns.tolist()

    result_df = pd.DataFrame(columns=var_names)
    result_df["Time_Start"] = ticker_df.index
    result_df["Time_End"] = result_df["Time_Start"].shift(-K) #personal note: shift up by K instead of adding K days bc you not every day has price
    result_df = result_df.iloc[:len(ticker_df) - K] # take out the last K results bc out of range

    #get the optimal portfolio for every window, 
    for i, row in result_df.iterrows():
        start_time = row["Time_Start"]
        end_time = row["Time_End"]

        window = ticker_df.loc[start_time:end_time]
        # Lookback function
        weights = get_optimal_portfolio(window)
        
        result_df.loc[i, ticker_df.columns] = weights[ticker_df.columns].values

    # Set Time_Start as the index
    result_df.set_index(['Time_End'], inplace=True) #index is time end because we are "looking back" in the window to determine weights
    
    return result_df

def train_model(ticker_list, start_date, end_date, K, futureSteps):
    sliding_data = sliding(ticker_list, start_date, end_date, K)
    # Prepare the training data
    Xtrain = np.array([sliding_data[ticker].apply(lambda x: x[0]) for ticker in ticker_list]).T
    
    # Ytrain: target values: Xtrain vals shifted by futureSteps upward
    Ytrain = np.roll(Xtrain, -futureSteps, axis=0)
    
    #slice, remove values that were rolled around
    Xtrain = Xtrain[:-futureSteps, :]
    Ytrain = Ytrain[:-futureSteps, :]

    #train model
    models = []
    for i in range(Ytrain.shape[1]):
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3)
        model.fit(Xtrain, Ytrain[:, i])
        models.append(model)

    return models

def predict_weights(models, ticker_list, start_date, end_date, K):
    #new sliding data
    new_sliding_data = sliding(ticker_list, start_date, end_date, K)
    #turn list into array for each ticker
    Xtest = np.array([new_sliding_data[ticker].apply(lambda x: x[0]) for ticker in ticker_list]).T

    # Generate predictions using the models
    predictions = []
    for model in models:
        pred = model.predict(Xtest)
        predictions.append(pred)

    weights_df = pd.DataFrame(np.column_stack(predictions), columns=[f'{ticker}_weight' for ticker in ticker_list])
    weights_df.index = new_sliding_data.index #ensure there are dates

    #normalize, divides each weight by the total sum of weights for that row, summing to 1
    weights_df = weights_df.div(weights_df.sum(axis=1), axis=0)
    return weights_df

def backtest(ticker_list, weights_df, starting_cash, start_date, end_date):
    # Gather raw price data
    price_data = get_raw_prices(ticker_list, start_date, end_date)
    
    net_cash = starting_cash
    previous_prices = None
    portfolio_values = []
    
    for date in weights_df.index:  # Use weights_df index to ensure alignment (first several may not have weights bc of lookback)
        weights = weights_df.loc[date] 
        if previous_prices is None:
            previous_prices = {ticker: price_data.loc[date, ticker] for ticker in ticker_list}
            
            # Skip the first iteration since there's no previous price to compare
            portfolio_values.append(net_cash)
            continue
        
        # Allocate net_cash based on weights
        allocations = {ticker: net_cash * weights[f'{ticker}_weight'] for ticker in ticker_list}
        
        # As the prices of the tickers change, the value of each allocation will change.
        # The total capital is then recalculated based on the new values of these allocations
        for ticker in ticker_list:
            current_price = price_data.loc[date, ticker]
            price_change = (current_price - previous_prices[ticker]) / previous_prices[ticker]
            allocations[ticker] += allocations[ticker] * price_change
            previous_prices[ticker] = current_price
        
        # Sum the allocations to get the updated capital
        net_cash = sum(allocations.values())
        portfolio_values.append(net_cash)

    portfolio_df = pd.DataFrame(portfolio_values, index=weights_df.index, columns=['Portfolio Value'])
    
    return portfolio_df

def main():
    ticker_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'AMD', 'NVDA', 'INTC', 'QCOM', 'IBM']
    # ticker_list = ['AAPL', 'TSLA']

    # # Print sliding window data for the given tickers and date range
    # print(sliding(ticker_list, '2020-01-04', '2024-08-19', 5))
    # print(get_raw_prices(ticker_list, '2020-01-04', '2024-08-19'))
    today = datetime.today()

    # Set training and testing date ranges
    train_start_date = '2011-01-01'
    train_end_date = (today - timedelta(days=10 + 365)).strftime('%Y-%m-%d')  # End training 10 days before 1 year from now
    
    test_start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')      # Start testing (predictions) 1 year from now
    test_end_date = today.strftime('%Y-%m-%d')                                 # End testing (predictions) today

    K = 8  # Size of sliding window
    futureSteps = 7  # How many days in the future you want to predict

    # Train the model with the specified parameters
    models = train_model(ticker_list, train_start_date, train_end_date, K, futureSteps)

    # Predict weights using the trained models and test data
    weights_df = predict_weights(models, ticker_list, test_start_date, test_end_date, K)
    print(weights_df)

    # Plotting the predicted weights for each ticker over time
    plt.figure(figsize=(14, 7))
    for ticker in ticker_list:
        plt.plot(weights_df.index, weights_df[f'{ticker}_weight'], label=f'{ticker} Weight')
    plt.title('Predicted Weights Over Time')
    plt.legend()
    plt.show()

    # Backtest simulation
    start_date = test_start_date
    end_date = test_end_date
    starting_cash = 100000

    # Backtest with our dynamic allocation strategy
    portfolio_df = backtest(ticker_list, weights_df, starting_cash, start_date, end_date)

    # Backtest static evenly distributed allocation strategy
    num_tickers = len(ticker_list) 
    even_weight = 1.0 / num_tickers #even weight calculation
    static_weights_df = pd.DataFrame(even_weight, index=weights_df.index, columns=[f'{ticker}_weight' for ticker in ticker_list])
    static_portfolio_df = backtest(ticker_list, static_weights_df, starting_cash, test_start_date, test_end_date)

    # Plotting the portfolio values over time for both strategies
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_df.index, portfolio_df['Portfolio Value'], label='Dynamic Allocation')
    plt.plot(static_portfolio_df.index, static_portfolio_df['Portfolio Value'], label='Dummy Even Allocation', linestyle='--')
    plt.title('Backtest Portfolio Value Over Time')
    plt.legend()
    plt.show()

    # Grab the final net cash value for each strategy
    final_net_cash_dynamic = portfolio_df['Portfolio Value'].iloc[-1]
    final_net_cash_static = static_portfolio_df['Portfolio Value'].iloc[-1]

    # Print the final portfolio values for both strategies
    print(f"Final net cash for dynamic allocation strategy: ${final_net_cash_dynamic:.2f}")
    print(f"Final net cash for static allocation strategy: ${final_net_cash_static:.2f}")

if __name__ == '__main__':
    main()




