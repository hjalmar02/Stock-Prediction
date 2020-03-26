import math
import pandas as pd
import pandas_datareader as web
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import sys
plt.style.use('ggplot')
pd.set_option('display.max_columns', 7)

today = dt.date.today()
std_start = today - dt.timedelta(30)


class Stock:

    def __init__(self, name):
        """[...]

        Args:
                name (str): Stock to examine, e.g. "AAPL" for Apple Inc.
                """

        self.name = name

    def get_quote(self, start=std_start, end=today):
        """Get historical quote for the stock from Yahoo.

        Args:
                start (str): Starting date (YYYY-MM-DD).
                end (str): Ending date (YYYY-MM-DD).
        """

        self.end = end

        # Create a pandas DataFrame with the stock quote
        self.quote = web.DataReader(
            self.name, data_source='yahoo', start=start, end=end)

    def graph(self, data):
        """Graph stock data in pyplot.

        Args:
                data (str): Type of data to graph, e.g. 'Close'.
        """
        plt.figure(figsize=(16, 8))
        plt.title('Test')
        plt.plot(self.quote[data])
        plt.xlabel('Date', fontsize=18)
        plt.ylabel(data + ' price USD')
        plt.show()

    def predict(self, date):
        """Predict closing price on date.

        Args:
                date (date): Date on which to predict closing price (YYYY-MM-DD)
        """

        mdate = str(date)
        rdate = str(self.end)
        mdate1 = dt.datetime.strptime(mdate, "%Y-%m-%d").date()
        rdate1 = dt.datetime.strptime(rdate, "%Y-%m-%d").date()
        delta = (mdate1 - rdate1).days

        predict_index = self.quote.shape[0] + delta - 1

        # Converting index to numerical values instead of dates.
        index = []
        for i in range(self.quote.shape[0]):
            index.append(i)
        self.quote['Index'] = index
        self.quote['Date'] = self.quote.index.values
        self.quote.set_index('Index', inplace=True)

        x = self.quote.index.values.reshape(-1, 1).flatten()
        y = self.quote['Close'].values

        model = LinearRegression().fit(x.reshape(-1, 1), y)

        self.quote.set_index('Date', inplace=True)

        return model.predict(np.array([predict_index]).reshape(-1, 1))

    def test(self, period):
        """Test accuracy of prediction relative to real values.

        Args:
            period (int): How many days in the past to test accuracy
        """

        dates = []
        real_closes = []
        predicted_closes = []
        errors = []
        for i in range(1, period):
            try:
                date = today - dt.timedelta(i)
                real_close = self.quote['Close'][date]
                predict_close = self.predict(date)[0]
                error = abs(predict_close - real_close) / real_close * 100

                dates.append(date)
                real_closes.append(real_close)
                predicted_closes.append(predict_close)
                errors.append(error)
            except:
                continue

        test_df = pd.DataFrame(data={'Date': dates, 'Predicted Close': predicted_closes,
                                     'Real Close': real_closes,
                                     'Error': errors})
        test_df.set_index('Date', inplace=True)
        plt.figure(figsize=(16, 8))
        plt.title('Test')
        plt.plot(test_df['Real Close'])
        plt.plot(test_df['Predicted Close'])
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Closing price USD')
        plt.show()

        print(test_df)

        mean_error = sum(errors) / len(errors)
        print(f"Mean error: {round(mean_error, 2)} %")


if __name__ == "__main__":
    test = Stock(input('Stock name:'))
    test.get_quote(today - dt.timedelta(1000))

    while True:
        action = input('Choose action [predict/exit/test]: ')


        if action == 'predict':
            dates = []
            closes = []
            dates.append(today)
            closes.append(test.quote['Close'][today])
            for i in range(1, 1000):
                date = today + dt.timedelta(i)
                close = test.predict(date)[0]
                dates.append(date)
                closes.append(close)

            prediction = pd.DataFrame(data={'Date': dates, 'Close': closes})
            prediction.set_index('Date', inplace=True)

            plt.figure(figsize=(16, 8))
            plt.title('Test')
            plt.plot(test.quote['Close'])
            plt.plot(prediction['Close'])
            plt.xlabel('Date', fontsize=18)
            plt.ylabel('Closing price USD')
            plt.show()

        elif action == 'test':
            test.test(100)

        elif action == 'exit':
            sys.exit()
        else:
            print('invalid command')
