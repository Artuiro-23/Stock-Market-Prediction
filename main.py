import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('infolimpioavanzadoTarget.csv')

print(df.info())
print(df.head())
print(df.describe())


plt.figure(figsize=(12, 6))
sns.histplot(df['close'], kde=True)
plt.title('Distribution of Stock Prices')
plt.xlabel('Closing Price')
plt.show()


df['date'] = pd.to_datetime(df['date']) 
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['close'], label='Closing Price')
plt.title('Stock Price Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.xticks(rotation=45)
plt.legend()
plt.show()


correlation_matrix = df[['open', 'high', 'low', 'close']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', fmt=".2f") 
plt.title('Correlation Matrix')
plt.show()


sns.pairplot(df[['open', 'high', 'low', 'close']])
plt.suptitle('Pairplot of Stock Price Features', y=1.02)
plt.show()
