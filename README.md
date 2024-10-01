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
