# 🛒 Amazon Product Review Analysis Dashboard

Welcome to the **Amazon Product Review Analysis Dashboard**! This project analyzes over 568K Amazon reviews, uncovering insights through advanced techniques like **LDA Topic Modeling**, **Sentiment Analysis**, and **Helpfulness Prediction**. Built with Streamlit, this dashboard offers an interactive way to explore customer sentiments and review patterns. 🚀

---

## ✨ Key Insights

- 🧠 **Topic Modeling**: Identified 15 key topics, with coherence peaking at 10 topics for meaningful themes.
- 📈 **Sentiment Trends**: Positive sentiment surged between 2000 and 2012, reflecting growing customer satisfaction.
- ⭐ **Helpfulness Factors**: Text length and word count are the top predictors of a review’s helpfulness.

---

## 📊 Output Visualizations

### 🔍 Topic Modeling
Uncover the main themes in Amazon reviews using LDA.

- **Topic Coherence**: Coherence peaks at 10 topics, ensuring meaningful themes.  
  ![Topic Coherence](dashboard/assets/3_topic_coherence.png)
  
- **Word Cloud**: Visualizes the most frequent terms, with larger words indicating higher frequency.  
  ![Word Cloud](dashboard/assets/4_word_cloud.png)

### 😊 Sentiment Analysis
Analyze customer sentiment over time and across categories.

- **Sentiment Trends**: Tracks polarity over time, showing a rise in positivity from 2000 to 2012.  
  ![Sentiment Trends](dashboard/assets/5_sentiment_trends.png)
  
- **Sentiment Distribution**: Displays the spread of sentiment scores across ratings.  
  ![Sentiment Distribution](dashboard/assets/6_sentiment_distribution.png)
  
- **Sentiment Heatmap**: Highlights sentiment variations across categories over time.  
  ![Sentiment Heatmap](dashboard/assets/7_sentiment_heatmap.png)

### 🔗 Impact Analysis
Understand how topics influence sentiment.

- **Topic-Sentiment Correlations**: Reveals how topics correlate with sentiment scores.  
  ![Topic-Sentiment Correlations](dashboard/assets/8_topic_sentiment_correlations.png)

### 💡 Helpfulness Analysis
Explore what makes a review helpful.

- **Feature Importance**: Identifies text length and word count as key predictors of helpfulness.  
  ![Feature Importance](dashboard/assets/9_feature_importance.png)
  
- **Category Patterns**: Shows helpfulness variations across product categories.  
  ![Category Patterns](dashboard/assets/10_category_patterns.png)

---

## 🚀 Live Dashboard Setup

To run the interactive dashboard locally, ensure you have the required dependencies (`streamlit`, `pandas`, `PIL`), then execute:

```bash
streamlit run dashboard/app.py