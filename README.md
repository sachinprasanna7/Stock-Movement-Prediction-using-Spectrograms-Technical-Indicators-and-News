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

Module: Financial News Feature Extraction for Stock Movement Prediction
Overview
This module focuses on extracting relevant features from financial news articles to enhance stock movement prediction models. By integrating sentiment analysis and feature extraction, we aim to capture the impact of news events on stock prices.

Data Collection
We collected 50,000 financial news articles from Business Standard, covering the period from 2018 to 2021. The dataset includes news headlines, publication dates, and sentiment scores.

Data Preprocessing
The news data was preprocessed to extract relevant features such as named entities, topics, tone, and sentiment. We employed the FinBERT sentiment analysis model, fine-tuned for financial texts, to assess the sentiment of each news article.

Feature Extraction
Key features extracted from the news data include:

Company: The primary company mentioned in the news article.
Event: The main event or action, categorized into predefined types such as "Merger," "New Product," etc.
Reason: The reason for the event, categorized into predefined types such as "Market Demand," "Regulatory Requirement," etc.
Sentiment: The overall sentiment of the news article.
Integration with Stock Data
The extracted features are integrated with traditional technical indicators and spectrogram data to create a comprehensive dataset for stock movement prediction. This combined dataset is used to train machine learning models, leveraging the strengths of each data type.

Explanation of Feature Usefulness for Stock Predictions
Company
Why it's useful: Identifying the primary company mentioned in a news article helps in directly linking the news to the stock of that company. This allows for targeted analysis of how specific news impacts the stock price of the mentioned company.
Event
Why it's useful: Categorizing the main event or action (e.g., "Merger," "New Product") helps in understanding the nature of the news. Different types of events have varying impacts on stock prices. For example, a merger announcement might lead to a stock price increase, while a product recall might lead to a decrease.
Reason
Why it's useful: Understanding the reason behind an event (e.g., "Market Demand," "Regulatory Requirement") provides context that can influence the magnitude and direction of the stock price movement. For instance, a new product launch due to high market demand might be more positively received than one due to regulatory pressure.
Sentiment
Why it's useful: The overall sentiment of the news article (positive, negative, neutral) is a direct indicator of market sentiment. Positive news sentiment generally leads to stock price increases, while negative sentiment can lead to decreases. Sentiment analysis helps quantify this impact.
Example Code Explanation
The provided code snippet is part of the data preprocessing step where the news data is loaded and filtered to retain only the relevant columns (Title and Date). This is a preliminary step before extracting the key features mentioned above.

Loading Data: The CSV file containing news articles is loaded into a DataFrame.
Filtering Columns: Only the 'Title' and 'Date' columns are retained, as these are essential for further feature extraction.
Displaying Data: A specific row's title is printed to verify the data loading process.
This preprocessing step ensures that the dataset is clean and focused, making it easier to extract the key features that will be used for stock prediction.

