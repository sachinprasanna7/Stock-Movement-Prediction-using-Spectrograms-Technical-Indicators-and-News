## Stock-Trend-Prediction

### Introduction
We propose a novel method for Stock Movement Prediction that utilizes four different models. The best model is selected through a weighted average method, where the weights are treated as parameters and optimized using a Neural Network. The proposed model aims to predict whether a stock will rise or fall on a given day.

---

### Methodology

#### A. Methodologies
1) **Converting Close Data to Spectrograms**:  
   a) Transform historical close data into spectrograms.  
   b) Visualization of patterns and potential reduction of noise.  
   c) Analysis of spectrograms using Convolutional Neural Networks (CNNs).  

2) **Utilizing Technical Indicators**:  
   a) Integrate technical indicators into the analysis to gain additional insights into stock movements.

3) **Using Correlations with Related Stocks**:  
   a) Use rolling correlation to analyze the influence of related stocks in predicting the selected stock's trend.

4) **Utilizing General News**:  
   a) Incorporate general news to understand its impact on stock prices.

5) **Using Neural Networks for Combining Spectrograms, Indicators, and General News**:  
   a) Employ a neural network to combine features from spectrograms, technical indicators, and general news.  
   b) Leverage the strengths of each data type for more accurate predictions.


---

### Dataset

**Dataset 1**  
We focused on 12 stocks of BankNifty and created a CSV file for each stock along with their technical indicators. The daily stock data from 1 Jan 2017 to 31 December 2022 was used in our study and dataset construction. The dataset was created using the `yfinance` and `pandas_ta` libraries.

- `yfinance`: Easily downloads historical market data from Yahoo Finance.
- `pandas_ta`: A technical analysis library built on top of Pandas offering a wide range of indicators.

**Technical Indicators used in the dataset include**:  
Simple Moving Averages (SMA 9 and SMA 21), Exponential Moving Averages (EMA 9 and EMA 21), Double Exponential Moving Averages (DEMA 9 and DEMA 21), MACD (Line and Signal Line), RSI (14), Stochastic Oscillator (K and D), Bollinger Bands (middle, upper, and lower), ADX (ADX, ADX+DI, ADX-DI), CMF (Chaikin Money Flow), OBV (On Balance Volume), CCI (Commodity Channel Index), Williams %R, ATR (Average True Range).

The dataset was cleaned to remove rows with NaN values and normalized using min-max normalization. After cleaning and normalizing, the final size of the CSV file for each stock was (1322, 29).


**Dataset 2**  
**News & Stock Movement**  
News that impacts Bank Nifty stocks. We collected 50,000 financial news articles from Business Standard (2003-2020).

**Data Focus**  
We restricted the dataset to 2018-2020 to focus on recent trends and align it with the technical data.


---

### Work Done So Far

#### Approach 1: Using Spectrograms
Spectrograms are visual representations of the frequency spectrum of a signal over time. We generated spectrograms from windowed segments of stock closing prices (30-day window). The generated spectrograms were used as features for predictive modeling using CNN architectures: AlexNet, ResNet, DenseNet, and EfficientNet.

- **Test Accuracy of CNN Models**:  
  - AlexNet: 54.12%  
  - ResNet: 54.12%  
  - DenseNet: 45.88%  
  - EfficientNet: 45.88%

To improve accuracy, we plan to explore alternative strategies like adjusting the method of spectrogram generation and altering hyperparameters.

#### Approach 2: Using Technical Indicators
To predict stock price movements, machine learning techniques were applied using technical indicators as features. Additional steps taken include:

1) **Technical Indicators as Features**:  
   Mathematical calculations based on historical data were used to predict stock price direction.
   
   **Test Accuracy of Machine Learning Models**:  
   - Random Forest: 50.19%  
   - Support Vector Machine: 51.70%  
   - Decision Tree: 50.57%  
   - Logistic Regression: 51.70%  
   - Neural Network: 50.94%

2) **Lagging Technical Features**:  
   Lagging versions of indicators (e.g., 20-day moving average) were introduced to capture past market conditions.

3) **Feature Selection Approach**:  
   We performed feature selection using the Random Forest method to reduce dataset dimensionality while maintaining high predictive power. The top five indicators were selected for the final model.

   **Test Accuracy of Machine Learning Models**:  
   - Random Forest: 50.19%  
   - Support Vector Machine: 51.70%  
   - Decision Tree: 50.57%  
   - Logistic Regression: 51.70%  
   - Neural Network: 50.94%

#### Approach 3: Using News
News events are known to influence stock price movements. We integrated financial news data from 50,000 news articles sourced from Business Standard, covering the period from 2018 to 2021. Key features extracted from the news include named entities, topics, tone, sentiment, etc.  

We employed the FinBERT sentiment analysis model, which is trained on financial texts, to extract sentiment from the news data, improving the predictive accuracy of our stock movement model.

# Financial News Feature Extraction for Stock Movement Prediction

## Overview
This module focuses on extracting critical features from financial news articles to enhance stock movement prediction models. By combining sentiment analysis with feature extraction, it captures the impact of news events on stock prices.

## Data Collection
- **Source**: Business Standard
- **Time Period**: 2018 - 2021
- **Volume**: 50,000 financial news articles
- **Included Data**: Headlines, publication dates, and sentiment scores

## Data Preprocessing
The preprocessing pipeline extracts relevant features such as named entities, topics, tone, and sentiment. We use the FinBERT model, fine-tuned for financial contexts, to evaluate sentiment for each news article.

## Feature Extraction
The following key features are extracted from the news data:

1. **Company**: The primary company mentioned in the news article.
2. **Event**: The main event or action, categorized into predefined types (e.g., "Merger," "New Product").
3. **Reason**: The cause of the event, categorized into predefined types (e.g., "Market Demand," "Regulatory Requirement").
4. **Sentiment**: The overall sentiment of the news article.

## Integration with Stock Data
The extracted features are integrated with traditional technical indicators and spectrogram data to build a comprehensive dataset for stock movement prediction. This combined dataset enables machine learning models to leverage multiple data types for improved accuracy.

## Explanation of Feature Usefulness for Stock Predictions

- **Company**
  - **Why it's useful**: Identifying the primary company in a news article helps directly link the news to that company’s stock. This allows for targeted analysis of news impact on the stock price of the specific company.

- **Event**
  - **Why it's useful**: Categorizing the main event or action (e.g., "Merger," "New Product") helps to understand the nature of the news. Different event types impact stock prices differently; for example, a merger announcement may lead to a stock price increase, while a product recall might decrease it.

- **Reason**
  - **Why it's useful**: Understanding the reason behind an event (e.g., "Market Demand," "Regulatory Requirement") provides context that can influence the magnitude and direction of stock price movement. For instance, a new product launch driven by high market demand is generally seen more positively than one due to regulatory pressure.

- **Sentiment**
  - **Why it's useful**: The overall sentiment of the article (positive, negative, neutral) serves as a direct indicator of market sentiment. Positive sentiment often leads to stock price increases, while negative sentiment can decrease it. Sentiment analysis quantifies this impact.

### Module: Financial News Sentiment Integration for Stock Prediction

#### Overview
This module integrates financial news sentiment with stock price data to simulate a trading strategy and evaluate its profitability. The goal is to understand how news sentiment can influence stock prices and trading decisions.

#### Methodology

1. **Data Collection and Preprocessing**
   - Stock data is collected from CSV files, and financial news articles are filtered to include only those relevant to specific stocks.
   - The news data is preprocessed to extract sentiment scores, which are then merged with the stock price data based on the date.

2. **Trading Strategy Implementation**
   - A simple trading strategy is defined based on the sentiment of the news articles:
     - **Positive Sentiment**: Buy shares if the sentiment is positive and there is available cash.
     - **Negative Sentiment**: Sell all shares if the sentiment is negative.
   - The strategy is applied over the period covered by the data to simulate trading decisions.

3. **Profit Calculation**
   - The final amount and profit are calculated after applying the trading strategy, providing a measure of the strategy's effectiveness.

4. **Results Visualization**
   - The profits for each stock are plotted, with green bars indicating positive profits and red bars indicating negative profits. This visualization helps in assessing the impact of news sentiment on stock performance.

#### Significance
- **Sentiment Analysis Integration**: By incorporating sentiment analysis from financial news, this module captures the impact of news events on stock prices, providing a more comprehensive view of market conditions.
- **Explainable Trading Strategy**: The straightforward and interpretable trading strategy based on sentiment makes it easier to understand the decision-making process.
- **Profit Evaluation**: The final profit calculation and visualization help in assessing the effectiveness of the sentiment-based trading strategy, offering insights into which stocks are more influenced by news sentiment.
- **Data-Driven Decisions**: The integration of news sentiment with stock data allows for data-driven trading decisions, potentially leading to better investment outcomes.

This module demonstrates the value of combining financial news with traditional stock data to enhance predictive models and trading strategies.


