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

News events are known to influence stock price movements. We integrated financial news data from 5000 news articles sourced from Business Standard, covering the period from 2018 to 2021. Key features extracted from the news include named entities, topics, tone, sentiment, etc.  

We employed the FinBERT sentiment analysis model, which is trained on financial texts, to extract sentiment from the news data, improving the predictive accuracy of our stock movement model.

# Financial News Feature Extraction for Stock Movement Prediction

## Overview

This module focuses on extracting critical features from financial news articles to enhance stock movement prediction models. By combining sentiment analysis with feature extraction, it captures the impact of news events on stock prices.

## Data Collection
- **Source**: Business Standard
- **Time Period**: 2018 - 2021
- **Volume**: 5000 financial news articles
- **Included Data**: Headlines, publication dates, and sentiment scores

## Data Preprocessing
The preprocessing pipeline extracts relevant features such as named entities, topics, tone, and sentiment. Extracting these features is tricky due to the unstructured nature of news articles. Due to this reason, we employed an LLM based extraction approach. We tested a sample of the data on the GPT-4o model which is the current state of the art in language models. The model was able to extract the features with great accuracy. But due to expensive token costs, we had to shift to Google's Gemini model which is also capable of extracting the features with good accuracy.

## Feature Extraction
The following key features are extracted from the news data:

1. **Company**: The primary company mentioned in the news article.
2. **Event**: The main event or action, categorized into predefined types (e.g., "Merger," "New Product").
3. **Reason**: The cause of the event, categorized into predefined types (e.g., "Market Demand," "Regulatory Requirement").
4. **Sentiment**: The overall sentiment of the news article.

The prompt used to extract these features is given below:

```
"Analyze the following financial news headline: {news}."

    Based on this headline, extract the following structured information as a tuple, choosing each feature from its predefined categories:

    1. **Company**: Identify the primary company mentioned"
    2. **Event**: Identify the main event or action, choosing from: "Merger," "New Product," "Profit/Loss Announcement," "Partnership/Collaboration," "Policy Change."
    3. **Reason**: Identify the reason for the event, choosing from: "Market Demand," "Regulatory Requirement," "Internal Strategy," "External Competition," "Economic Conditions."
    4. **Verdict**: Assess the likely impact on stock, choosing from: "UP," "DOWN," "NEUTRAL."

    NOTE: give categories from the categories provided in the prompt. DO NOT EXPLAIN THE CATEGORIES or output. Also, try to reason out the answer based on the headline see if the news might have any impact on the stock price of the company mentioned in the headline. DO NOT OUTPUT ANYTHING ELSE OTHER THEN THE TUPLE. DO NOT SAY - " Here's the structured information extracted from the headlines:"

    **Return the information in tuple format** only and nothing else using the example format below:

    Example format:
    `("Company", "Event", "Reason", "Verdict")`

        """
   ```



## Explanation of Feature Usefulness for Stock Predictions

The extracted features help us understand the context and impact of news events on stock prices. The reason behind choosing these features is as follows:

- **Company**
  - **Why it's useful**: Identifying the primary company in a news article helps directly link the news to that company’s stock. This allows for targeted analysis of news impact on the stock price of the specific company.

- **Event**
  - **Why it's useful**: Categorizing the main event or action (e.g., "Merger," "New Product") helps to understand the nature of the news. Different event types impact stock prices differently; for example, a merger announcement may lead to a stock price increase, while a product recall might decrease it.

- **Reason**
  - **Why it's useful**: Understanding the reason behind an event (e.g., "Market Demand," "Regulatory Requirement") provides context that can influence the magnitude and direction of stock price movement. For instance, a new product launch driven by high market demand is generally seen more positively than one due to regulatory pressure.

- **Sentiment**
  - **Why it's useful**: The overall sentiment of the article (positive, negative, neutral) serves as a direct indicator of market sentiment. Positive sentiment often leads to stock price increases, while negative sentiment can decrease it. Sentiment analysis quantifies this impact.

## Evaluation of the Impact of News Features on Stock Movement Prediction through trading strategy

In order to evaluation the effect of sentiment on the Bank Nifty stocks, we will use the sentiment extracted from the news articles to create a trading strategy. The strategy will be based on the sentiment of the news articles and the stock movement on the same day. We will backtest the strategy on the historical data to evaluate its performance.

```

def trading_strategy(row, amount, shares):
        if row['Sentiment'] == 'Positive' and amount > 0:
            # Buy shares
            shares_to_buy = amount / row['Close']
            amount -= shares_to_buy * row['Close']
            shares += shares_to_buy
        elif row['Sentiment'] == 'Negative' and shares > 0:
            # Sell shares
            amount += shares * row['Close']
            shares = 0
        return amount, shares

    # Apply the trading strategy over the period
    for index, row in merged_df.iterrows():
        amount, shares = trading_strategy(row, amount, shares)
```

On running this strategy, we found out that the strategy was profitable on KOtak, sbi, federal, axis, idfc banks, while it was not profitable on the rest





\subsection{Financial News Extraction}

\subsubsection{Overview}

This module focuses on extracting critical features from financial news articles to enhance stock movement prediction models. By combining sentiment analysis with feature extraction, it captures the impact of news events on stock prices.

\subsubsection{Data Collection}
\begin{itemize}
    \item \textbf{Source}: Business Standard
    \item \textbf{Time Period}: 2018 - 2021
    \item \textbf{Volume}: 5000 financial news articles
    \item \textbf{Included Data}: Headlines, publication dates, and sentiment scores
\end{itemize}

\subsubsection{Data Preprocessing}
The preprocessing pipeline extracts relevant features such as named entities, topics, tone, and sentiment. Extracting these features is challenging due to the unstructured nature of news articles. To address this, we employed a large language model (LLM)-based extraction approach. A sample of the data was tested on the GPT-4 model, known for its state-of-the-art accuracy. However, due to token cost constraints, we shifted to Google's Gemini model, which also provided accurate feature extraction.

\subsubsection{Feature Extraction}
The following key features are extracted from the news data:

\begin{enumerate}
    \item \textbf{Company}: The primary company mentioned in the news article.
    \item \textbf{Event}: The main event or action, categorized into predefined types (e.g., "Merger," "New Product").
    \item \textbf{Reason}: The cause of the event, categorized into predefined types (e.g., "Market Demand," "Regulatory Requirement").
    \item \textbf{Sentiment}: The overall sentiment of the news article.
\end{enumerate}

The prompt used to extract these features is as follows:

\begin{verbatim}
"Analyze the following financial news headline: {news}."

Based on this headline, extract the following structured information as a tuple, choosing each feature from its predefined categories:

1. **Company**: Identify the primary company mentioned.
2. **Event**: Identify the main event or action, choosing from: "Merger," "New Product," "Profit/Loss Announcement," "Partnership/Collaboration," "Policy Change."
3. **Reason**: Identify the reason for the event, choosing from: "Market Demand," "Regulatory Requirement," "Internal Strategy," "External Competition," "Economic Conditions."
4. **Verdict**: Assess the likely impact on stock, choosing from: "UP," "DOWN," "NEUTRAL."

NOTE: give categories from the categories provided in the prompt. DO NOT EXPLAIN THE CATEGORIES or output. Also, try to reason out the answer based on the headline to see if the news might have any impact on the stock price of the company mentioned in the headline. DO NOT OUTPUT ANYTHING ELSE OTHER THAN THE TUPLE.

**Return the information in tuple format** only and nothing else using the example format below:

Example format:
`("Company", "Event", "Reason", "Verdict")`
\end{verbatim}

\subsubsection{Explanation of Feature Usefulness for Stock Predictions}

\begin{itemize}
    \item \textbf{Company}
    \begin{itemize}
        \item \textit{Why it's useful}: Identifying the primary company in a news article helps directly link the news to that company’s stock. This allows for targeted analysis of news impact on the stock price of the specific company.
    \end{itemize}

    \item \textbf{Event}
    \begin{itemize}
        \item \textit{Why it's useful}: Categorizing the main event or action (e.g., "Merger," "New Product") helps to understand the nature of the news. Different event types impact stock prices differently; for example, a merger announcement may lead to a stock price increase, while a product recall might decrease it.
    \end{itemize}

    \item \textbf{Reason}
    \begin{itemize}
        \item \textit{Why it's useful}: Understanding the reason behind an event (e.g., "Market Demand," "Regulatory Requirement") provides context that can influence the magnitude and direction of stock price movement. For instance, a new product launch driven by high market demand is generally seen more positively than one due to regulatory pressure.
    \end{itemize}

    \item \textbf{Sentiment}
    \begin{itemize}
        \item \textit{Why it's useful}: The overall sentiment of the article (positive, negative, neutral) serves as a direct indicator of market sentiment. Positive sentiment often leads to stock price increases, while negative sentiment can decrease it. Sentiment analysis quantifies this impact.
    \end{itemize}
\end{itemize}



\subsection{Evaluation of the Impact of News Features on Stock Movement Prediction through Trading Strategy}

In order to evaluate the effect of sentiment on the Bank Nifty stocks, we use the sentiment extracted from news articles to create a trading strategy. The strategy is based on the sentiment of the news articles and the stock movement on the same day. We backtest the strategy on historical data to assess its performance.

\subsubsection{Trading Strategy Algorithm}

Let:
\begin{itemize}
    \item \( \text{Sentiment}_t \) be the sentiment score (Positive, Negative, Neutral) for day \( t \).
    \item \( \text{Close}_t \) be the closing stock price on day \( t \).
    \item \( \text{Amount}_t \) be the cash available on day \( t \).
    \item \( \text{Shares}_t \) be the number of shares held at the end of day \( t \).
\end{itemize}

The trading strategy is defined as follows:

1. **Initial Conditions:**
   \[
   \text{Amount}_0 = \text{initial investment amount}
   \]
   \[
   \text{Shares}_0 = 0
   \]

2. **Daily Trading Decision:**
   For each trading day \( t \):
   
   \begin{itemize}
       \item **If** \( \text{Sentiment}_t = \text{Positive} \) and \( \text{Amount}_{t-1} > 0 \):
           \begin{align*}
           \text{Shares Bought}_t &= \frac{\text{Amount}_{t-1}}{\text{Close}_t} \\
           \text{Amount}_t &= \text{Amount}_{t-1} - (\text{Shares Bought}_t \times \text{Close}_t) \\
           \text{Shares}_t &= \text{Shares}_{t-1} + \text{Shares Bought}_t
           \end{align*}

       \item **If** \( \text{Sentiment}_t = \text{Negative} \) and \( \text{Shares}_{t-1} > 0 \):
           \begin{align*}
           \text{Amount}_t &= \text{Amount}_{t-1} + (\text{Shares}_{t-1} \times \text{Close}_t) \\
           \text{Shares}_t &= 0
           \end{align*}
       
       \item **Otherwise**:
           \begin{align*}
           \text{Amount}_t &= \text{Amount}_{t-1} \\
           \text{Shares}_t &= \text{Shares}_{t-1}
           \end{align*}
   \end{itemize}

3. **Final Outcome**: 
   The total portfolio value at the end of the period \( T \) is:
   \[
   \text{Portfolio Value} = \text{Amount}_T + (\text{Shares}_T \times \text{Close}_T)
   \]

After running this strategy, we observed that it was profitable for Kotak, SBI, Federal, Axis, and IDFC banks, but not profitable for the other banks.
