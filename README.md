# Stock Movement Prediction using Spectrograms, Technical Indicators, and News

## Introduction
This project introduces a novel approach to Stock Movement Prediction by leveraging a combination of techniques to improve predictive accuracy. The solution involves using four distinct models, with the best-performing model selected via a weighted average, where weights are optimized through a Neural Network. The objective is to predict whether a stock will rise or fall on a given day.

## Methodology

### 1. Converting Close Data to Spectrograms
- **Transformation**: Historical stock close prices are transformed into spectrograms.
- **Visualization**: This approach helps visualize hidden patterns and potentially reduces noise.
- **CNN Analysis**: These spectrograms are then analyzed using Convolutional Neural Networks (CNNs) to capture significant features.

### 2. Utilizing Technical Indicators
- **Integration of Indicators**: A variety of technical indicators are computed and integrated into the model for a deeper understanding of market behavior.
- **Indicators Used**: Includes SMAs, EMAs, MACD, RSI, Bollinger Bands, ADX, Chaikin Money Flow, and more (details below).

### 3. Incorporating Financial News
- **News Impact**: Financial news plays a crucial role in influencing stock prices. Sentiment analysis of news articles is incorporated to account for this factor.
- **Sentiment Analysis**: We employ FinBERT, a model fine-tuned for financial sentiment, to assess the sentiment of financial news and its effect on stock price movement.

### 4. Neural Networks for Feature Combination
- **Unified Model**: A neural network is used to combine features extracted from spectrograms, technical indicators, and news data. This allows the model to leverage the strengths of each data type for more accurate stock movement predictions.

---

## Dataset

### Dataset 1: Stock Prices & Technical Indicators
- **Scope**: 12 BankNifty stocks from 1 Jan 2017 to 31 Dec 2022.
- **Technical Indicators**: 
    - Simple Moving Averages (SMA 9, SMA 21)
    - Exponential Moving Averages (EMA 9, EMA 21)
    - MACD (Line, Signal Line)
    - RSI (14), Bollinger Bands (Upper, Lower, Middle)
    - ADX, Chaikin Money Flow (CMF), On Balance Volume (OBV)
    - Average True Range (ATR), and others.
- **Data Collection**: Data was gathered using `yfinance` and technical indicators were calculated via the `pandas_ta` library.
- **Data Size**: Each stock's data resulted in a CSV with (1322, 29) dimensions after cleaning and normalization.

### Dataset 2: News & Stock Movement
- **Scope**: 50,000 financial news articles from Business Standard (2003-2020), focused on the period from 2018 to 2020.
- **Content**: Extracted features such as named entities, topics, tone, and sentiment using FinBERT for sentiment analysis.
- **Purpose**: News data helps capture market sentiment that may affect stock trends.

---

## Work Done So Far

### Approach 1: Spectrograms
- **Description**: We generated spectrograms from 30-day windowed segments of stock closing prices. These were input into CNN models such as AlexNet, ResNet, DenseNet, and EfficientNet for prediction.
- **Results**:
    - **AlexNet**: 54.12% accuracy
    - **ResNet**: 54.12% accuracy
    - **DenseNet**: 45.88% accuracy
    - **EfficientNet**: 45.88% accuracy
- **Next Steps**: Explore alternative spectrogram generation methods and adjust hyperparameters to improve accuracy.

### Approach 2: Technical Indicators
- **Description**: Applied machine learning models using technical indicators as features to predict stock price movements.
- **Feature Selection**: Utilized Random Forest for feature selection, reducing dimensionality and selecting the top 5 indicators for the final model.
- **Results**:
    - **Random Forest**: 50.19% accuracy
    - **Support Vector Machine (SVM)**: 51.70% accuracy
    - **Decision Tree**: 50.57% accuracy
    - **Logistic Regression**: 51.70% accuracy
    - **Neural Network**: 50.94% accuracy
- **Next Steps**: Further feature engineering and model optimization.

### Approach 3: Financial News
- **Description**: We integrated sentiment analysis on financial news articles using FinBERT to capture the sentiment and impact of news on stock prices.
- **Results**: This approach enhanced the model’s understanding of market sentiment, improving its predictive capabilities.


---

## Tools & Libraries
- **yfinance**: For downloading historical stock data.
- **pandas_ta**: For calculating technical indicators.
- **CNN Architectures**: AlexNet, ResNet, DenseNet, EfficientNet.
- **FinBERT**: For financial sentiment analysis.

---
