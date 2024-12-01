import streamlit as st
import pandas as pd
from PIL import Image

# Title
st.title("🛒 **Amazon Product Review Analysis Dashboard**")

# Define file paths
file_paths = {
    "Preprocessing": {
        "Data": r"C:\Users\MEET\Desktop\DS5230 Streamlit Dashboard\1 data.png",
    },
    "Topic Modeling": {
        "Top Terms": r"C:\Users\MEET\Desktop\DS5230 Streamlit Dashboard\2 top_terms.txt",
        "Topic Coherence": r"C:\Users\MEET\Desktop\DS5230 Streamlit Dashboard\3 topic_coherence.png",
        "Word Cloud": r"C:\Users\MEET\Desktop\DS5230 Streamlit Dashboard\4 word_cloud.png",
    },
    "Sentiment Analysis": {
        "Trends": r"C:\Users\MEET\Desktop\DS5230 Streamlit Dashboard\5 sentiment_trends.png",
        "Distribution": r"C:\Users\MEET\Desktop\DS5230 Streamlit Dashboard\6 sentiment_distribution.png",
        "Heatmap": r"C:\Users\MEET\Desktop\DS5230 Streamlit Dashboard\7 sentiment_heatmap.png",
    },
    "Impact Analysis": {
        "Topic Sentiment Correlations": r"C:\Users\MEET\Desktop\DS5230 Streamlit Dashboard\8 topic_sentiment_correlations.png",
    },
    "Helpfulness Analysis": {
        "Feature Importance": r"C:\Users\MEET\Desktop\DS5230 Streamlit Dashboard\9 feature_importance.png",
        "Category Patterns": r"C:\Users\MEET\Desktop\DS5230 Streamlit Dashboard\10 category_patterns.png",
    }
}

# Flowchart Steps
scroll_steps = [
    {"title": "Overview", 
     "content": "_The dashboard presents an analysis of Amazon product reviews, employing techniques like LDA for topic modeling and sentiment analysis to understand customer behavior._"},
    
    {"title": "Step 1: Data Preprocessing", 
     "content": """
     **Objective**  
     Prepare raw Amazon reviews by cleaning, handling missing data, and generating features for analysis.
     
     **Process Overview:**  
     - **Data Loading**: Load dataset, convert timestamps, and calculate helpfulness ratios.
     - **Missing Data**: Replace NaN values in `Text` and `Summary` with empty strings.
     - **Text Normalization**: Remove special characters, convert to lowercase, and normalize.
     - **Sentiment Analysis**: Use TextBlob for polarity and subjectivity scores.
     - **Feature Engineering**: Create features like text length, summary length, and word count.
     - **TF-IDF Vectorization**: Extract top 1,000 features for topic modeling.
     """},
    
    {"title": "Step 2: Topic Modeling", 
     "content": "_Topic modeling extracts the main themes from Amazon product reviews using Latent Dirichlet Allocation (LDA), and the results are visualized with word clouds. Below, we explain the process and how it fits into the dashboard._"},
    
    {"title": "Step 3: Sentiment Analysis", 
     "content": "_Analyze polarity and trends in sentiment over time to gauge customer satisfaction._"},
    
    {"title": "Step 4: Impact Analysis", 
     "content": "_Correlate topics and sentiment to identify factors influencing product performance._"},
    
    {"title": "Step 5: Helpfulness Analysis", 
     "content": "_Explore patterns in helpfulness ratings to identify what makes reviews valuable._"},
]

# Display Steps with Relevant Content
for step in scroll_steps:
    # Update title style for overview and steps
    st.markdown(f"<h3 style='color:#FF6347; font-family:\"Arial Black\"; font-size:30px;'>{step['title']}</h3>", unsafe_allow_html=True)  # Stylish titles with orange-red color and bold font
    st.markdown(step["content"])  # Descriptive content
    
    # Display relevant files based on the step
    if step["title"] == "Step 1: Data Preprocessing":
        st.image(file_paths["Preprocessing"]["Data"], caption="Dataset Overview", use_container_width=True)
        st.markdown("The dataset overview provides a snapshot of the raw Amazon reviews dataset before any processing steps are applied.")

    elif step["title"] == "Step 2: Topic Modeling":
        st.subheader("📌 Topic Coherence")
        st.image(file_paths["Topic Modeling"]["Topic Coherence"], caption="Topic Coherence", use_container_width=True)
        st.markdown("Topic coherence is a measure of how semantically meaningful the topics are. Higher coherence indicates that the words in a topic are more related to each other, helping identify more meaningful themes in the reviews.")
        
        st.subheader("📄 Top Terms in Topics")
        
        # Create a scrollable container for the topics
        with st.container():
            st.markdown("""
                <div style="height: 300px; overflow-y: scroll;">
                    <b>Topic 1</b>: food, cat, cats, baby, loves, eat, son, old, like, love<br>
                    <b>Topic 2</b>: tea, teas, green, flavor, chai, like, drink, taste, good, love<br>
                    <b>Topic 3</b>: food, dog, dogs, cat, cats, br, dry, ingredients, wellness, vet<br>
                    <b>Topic 4</b>: kitchen, oven, use, clean, works, easy, recommend, cooking, great, love<br>
                    <b>Topic 5</b>: shampoo, hair, thick, soft, smells, wash, healthy, clean, conditioner, perfect<br>
                    <b>Topic 6</b>: drink, coffee, espresso, taste, strong, flavor, machine, brew, like, quality<br>
                    <b>Topic 7</b>: pet, dog, food, health, dry, puppy, wellness, nutrition, love, vitamins<br>
                    <b>Topic 8</b>: soap, body, skin, dry, sensitive, moisturizing, organic, soft, scent, smells<br>
                    <b>Topic 9</b>: watch, time, design, gift, stylish, value, recommend, looks, band, quality<br>
                    <b>Topic 10</b>: charger, fast, phone, battery, works, plug, easy, charge, portable, life<br>
                    <b>Topic 11</b>: shoes, fit, size, comfortable, running, workout, material, perfect, durable, support<br>
                    <b>Topic 12</b>: camera, picture, quality, photo, video, lens, shot, DSLR, clear, zoom<br>
                    <b>Topic 13</b>: food, snacks, healthy, tasty, diet, calories, protein, bars, meal, gluten-free<br>
                    <b>Topic 14</b>: jacket, warm, size, cold, winter, fit, comfortable, breathable, outdoor, coat<br>
                    <b>Topic 15</b>: laptop, screen, fast, battery, performance, works, good, quality, lightweight, keyboard<br>
                </div>
            """, unsafe_allow_html=True)

        st.subheader("☁️ Word Cloud")
        st.image(file_paths["Topic Modeling"]["Word Cloud"], caption="Word Cloud Visualization", use_container_width=True)
        st.markdown("The word cloud dynamically displays the most common terms within the selected topics. Larger words indicate more frequent mentions in the reviews.")

    elif step["title"] == "Step 3: Sentiment Analysis":
        st.subheader("📈 Sentiment Trends")
        st.image(file_paths["Sentiment Analysis"]["Trends"], caption="Trends in Sentiment Over Time", use_container_width=True)
        st.markdown("The sentiment trend chart shows the polarity of reviews over time, helping to gauge how customer sentiment changes.")
        
        st.subheader("📊 Sentiment Distribution")
        st.image(file_paths["Sentiment Analysis"]["Distribution"], caption="Distribution of Sentiments", use_container_width=True)
        st.markdown("This distribution chart visualizes the spread of sentiment scores, showing how positive or negative the reviews tend to be.")
        
        st.subheader("🌡️ Sentiment Heatmap")
        st.image(file_paths["Sentiment Analysis"]["Heatmap"], caption="Heatmap of Sentiment Across Categories", use_container_width=True)
        st.markdown("The heatmap highlights sentiment variations across different product categories, showing areas of customer satisfaction or dissatisfaction.")

    elif step["title"] == "Step 4: Impact Analysis":
        st.subheader("🔗 Topic Sentiment Correlations")
        st.image(file_paths["Impact Analysis"]["Topic Sentiment Correlations"], caption="Correlations Between Topics and Sentiment", use_container_width=True)
        st.markdown("This chart reveals the relationships between identified topics and sentiment scores, helping to understand how different themes impact customer satisfaction.")

    elif step["title"] == "Step 5: Helpfulness Analysis":
        st.subheader("💡 Feature Importance")
        st.image(file_paths["Helpfulness Analysis"]["Feature Importance"], caption="Importance of Features in Predicting Helpfulness", use_container_width=True)
        st.markdown("Feature importance analysis shows which factors are most predictive of a review's helpfulness rating.")
        
        st.subheader("📊 Category Patterns")
        st.image(file_paths["Helpfulness Analysis"]["Category Patterns"], caption="Patterns in Helpfulness Across Categories", use_container_width=True)
        st.markdown("The patterns display the variation in helpfulness scores across different product categories, providing insights into which types of products tend to have more helpful reviews.")

# Final message to close
st.markdown("✨ **This concludes the analysis of Amazon Product Reviews. Thank you for exploring!**")

# Footer
st.markdown("<p style='font-size: 10px; color: grey;'>Created by Meet Jain | AI/ML Enthusiast</p>", unsafe_allow_html=True)
