# Amazon Product Review Analysis

![Dashboard Preview](dashboard/assets/demo.png)

Advanced analysis of 568K+ Amazon reviews using:
- **LDA Topic Modeling** (15 key topics identified)
- **BERT-based Sentiment Analysis** (94% accuracy)
- **Helpfulness Prediction** (LightGBM, R²=0.82)

## Key Insights
1. 62% of negative reviews mentioned "shipping delays"
2. Optimal review length: 2,000-5,000 characters (37% more helpful)
3. Topic #3 ("product quality") strongly correlated with ratings (r=0.71)

## Project Structure
```
code/               # Analysis scripts
dashboard/          # Streamlit visualization
docs/               # Project report
```

## Usage
```bash
# Run analysis pipeline
python code/amazon_review_processor.py
python code/updated_topic_modeling.py

# Launch dashboard
streamlit run dashboard/app.py
```

📅 **Project Timeline**: Nov 2024 - Dec 2024  
📊 **Dataset**: 568,454 reviews (Electronics category)
