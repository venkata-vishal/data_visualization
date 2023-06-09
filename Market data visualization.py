import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from pandas_datareader import data as pdr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class StockAnalyzer:
    def __init__(self, stock_list):
        self.stock_list = stock_list
        self.start_date = datetime.now() - pd.DateOffset(years=1)
        self.end_date = datetime.now()
        self.stock_data = []
        self.company_names = []
        self.predicted_prices = []

    def download_stock_data(self):
        yf.pdr_override()
        for stock in self.stock_list:
            data = pdr.get_data_yahoo(stock, self.start_date, self.end_date)
            data['Stock'] = stock
            self.stock_data.append(data)

    def combine_stock_data(self):
        self.combined_data = pd.concat(self.stock_data, axis=0)

    def summarize_data(self):
        print(self.combined_data.describe())

    def display_info(self):
        print(self.combined_data.info())

    def plot_closing_price(self):
        plt.figure(figsize=(15, 10))
        plt.subplots_adjust(top=1.25, bottom=1.2)

        for i, stock in enumerate(self.stock_data, 1):
            plt.subplot(2, 2, i)
            stock['Adj Close'].plot()
            plt.ylabel('Adj Close')
            plt.xlabel(None)
            plt.title(f"Closing Price of {stock['Stock'].iloc[0]}")

        plt.tight_layout()
        plt.show()

    def plot_volume(self):
        plt.figure(figsize=(15, 10))
        plt.subplots_adjust(top=1.25, bottom=1.2)

        for i, stock in enumerate(self.stock_data, 1):
            plt.subplot(2, 2, i)
            stock['Volume'].plot()
            plt.ylabel('Volume')
            plt.xlabel(None)
            plt.title(f"Sales Volume for {stock['Stock'].iloc[0]}")

        plt.tight_layout()
        plt.show()

    def calculate_moving_averages(self, ma_periods=[10, 20, 50]):
        for ma in ma_periods:
            for stock in self.stock_data:
                column_name = f"MA for {ma} days"
                stock[column_name] = stock['Adj Close'].rolling(ma).mean()

        fig, axes = plt.subplots(nrows=2, ncols=2)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        for i, stock in enumerate(self.stock_data, 1):
            axes[(i - 1) // 2, (i - 1) % 2].plot(
                stock[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']])
            axes[(i - 1) // 2, (i - 1) % 2].set_title(stock['Stock'].iloc[0])

        fig.tight_layout()
        plt.show()

    def calculate_daily_returns(self):
        for stock in self.stock_data:
            stock['Daily Return'] = stock['Adj Close'].pct_change()

        fig, axes = plt.subplots(nrows=2, ncols=2)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        for i, stock in enumerate(self.stock_data, 1):
            axes[(i - 1) // 2, (i - 1) % 2].plot(stock.index, stock['Daily Return'], linestyle='--', marker='o')
            axes[(i - 1) // 2, (i - 1) % 2].set_title(stock['Stock'].iloc[0])

        fig.tight_layout()
        plt.show()

    def calculate_expected_return_risk(self):
        returns = []
        for stock in self.stock_data:
            returns.append(stock['Daily Return'])

        expected_returns = []
        risks = []

        for return_series in returns:
            expected_return = return_series.mean()
            expected_returns.append(expected_return)

            risk = return_series.std()
            risks.append(risk)

        return expected_returns, risks

    def calculate_correlation(self):
        all_closes = pd.concat([stock['Adj Close'] for stock in self.stock_data], axis=1)
        all_closes.columns = self.stock_list

        correlation = all_closes.pct_change().corr()
        return correlation

    def predict_closing_price(self):
        for stock in self.stock_data:
            stock.dropna(inplace=True)
            X = stock[['Open', 'High', 'Low', 'Volume']]
            y = stock['Adj Close']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            predicted_price = model.predict(X_test)
            self.predicted_prices.append(predicted_price)

    def plot_actual_vs_predicted_prices(self):
        for i, stock in enumerate(self.stock_data):
            plt.figure(figsize=(10, 6))
            plt.plot(stock.index[-len(self.predicted_prices[i]):],
                     stock['Adj Close'].tail(len(self.predicted_prices[i])), label='Actual Price')
            plt.plot(stock.index[-len(self.predicted_prices[i]):], self.predicted_prices[i], label='Predicted Price')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title(f'Actual vs Predicted Closing Price of {stock["Stock"].iloc[0]}')
            plt.legend()
            plt.show()


# Testing
if __name__ == "__main__":
    stock_list = ['BMWYY', 'MBGYY', 'VWAGY', 'RACE']
    analyzer = StockAnalyzer(stock_list)

    analyzer.download_stock_data()
    analyzer.combine_stock_data()

    analyzer.summarize_data()
    analyzer.display_info()

    analyzer.plot_closing_price()
    analyzer.plot_volume()

    analyzer.calculate_moving_averages([10, 20, 50])

    analyzer.calculate_daily_returns()

    expected_returns, risks = analyzer.calculate_expected_return_risk()
    print("Expected Returns:")
    print(expected_returns)
    print("Risks:")
    print(risks)

    correlation = analyzer.calculate_correlation()
    print("Correlation Matrix:")
    print(correlation)

    analyzer.predict_closing_price()
    analyzer.plot_actual_vs_predicted_prices()