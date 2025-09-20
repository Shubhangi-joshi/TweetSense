import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Page configuration
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1DA1F2;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sentiment-positive {
        color: #00C851;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #ff4444;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #ffbb33;
        font-weight: bold;
    }
    .sentiment-irrelevant {
        color: #6c757d;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class SentimentAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.model = None
        self.vectorizer = None
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text) or text == "":
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def load_model(self, model_path):
        """Load the pre-trained model"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.vectorizer = model_data['vectorizer']
            self.model = model_data['model']
            return True
        except FileNotFoundError:
            return False
    
    def predict_sentiment(self, text):
        """Predict sentiment for a single text"""
        if self.model is None or self.vectorizer is None:
            return None, None
        
        cleaned_text = self.clean_text(text)
        if cleaned_text == "":
            return "Neutral", [0.25, 0.25, 0.25, 0.25]
        
        text_vectorized = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(text_vectorized)[0]
        probability = self.model.predict_proba(text_vectorized)[0]
        
        return prediction, probability

def load_sample_data():
    """Load sample data for visualization"""
    try:
        train_df = pd.read_csv('twitter_training.csv', names=['id', 'company', 'sentiment', 'text'])
        val_df = pd.read_csv('twitter_validation.csv', names=['id', 'company', 'sentiment', 'text'])
        return train_df, val_df
    except FileNotFoundError:
        return None, None

def create_sentiment_distribution_chart(df):
    """Create sentiment distribution pie chart"""
    sentiment_counts = df['sentiment'].value_counts()
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color_discrete_map={
            'Positive': '#00C851',
            'Negative': '#ff4444',
            'Neutral': '#ffbb33',
            'Irrelevant': '#6c757d'
        }
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_company_sentiment_chart(df):
    """Create company-wise sentiment analysis chart"""
    company_sentiment = df.groupby(['company', 'sentiment']).size().unstack(fill_value=0)
    
    # Get top 10 companies by total tweets
    top_companies = df['company'].value_counts().head(10).index
    company_sentiment_top = company_sentiment.loc[top_companies]
    
    fig = px.bar(
        company_sentiment_top,
        title="Sentiment Analysis by Company (Top 10)",
        labels={'value': 'Number of Tweets', 'company': 'Company'},
        color_discrete_map={
            'Positive': '#00C851',
            'Negative': '#ff4444',
            'Neutral': '#ffbb33',
            'Irrelevant': '#6c757d'
        }
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_text_length_analysis(df):
    """Create text length analysis"""
    df['text_length'] = df['text'].str.len()
    
    fig = px.box(
        df,
        x='sentiment',
        y='text_length',
        title="Text Length Distribution by Sentiment",
        color='sentiment',
        color_discrete_map={
            'Positive': '#00C851',
            'Negative': '#ff4444',
            'Neutral': '#ffbb33',
            'Irrelevant': '#6c757d'
        }
    )
    return fig

def main():
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Header
    st.markdown('<h1 class="main-header">🐦 Twitter Sentiment Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Predict Sentiment", "Data Analysis", "Model Training"])
    
    if page == "Predict Sentiment":
        st.header("🔮 Sentiment Prediction")
        
        # Load model
        model_loaded = analyzer.load_model('sentiment_model.pkl')
        
        if not model_loaded:
            st.error("Model not found! Please train the model first using the 'Model Training' page.")
            st.info("Run the following command in your terminal to train the model:")
            st.code("python sentiment_analysis.py", language="bash")
        else:
            st.success("✅ Model loaded successfully!")
            
            # Text input
            text_input = st.text_area(
                "Enter your text for sentiment analysis:",
                placeholder="Type your text here...",
                height=100
            )
            
            if st.button("Analyze Sentiment", type="primary"):
                if text_input.strip():
                    prediction, probability = analyzer.predict_sentiment(text_input)
                    
                    if prediction:
                        # Display prediction
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Predicted Sentiment", prediction)
                        
                        with col2:
                            confidence = max(probability) * 100
                            st.metric("Confidence", f"{confidence:.1f}%")
                        
                        with col3:
                            st.metric("Text Length", len(text_input))
                        
                        # Sentiment styling
                        sentiment_class = f"sentiment-{prediction.lower()}"
                        st.markdown(f'<p class="{sentiment_class}">Prediction: {prediction}</p>', unsafe_allow_html=True)
                        
                        # Probability distribution
                        st.subheader("Prediction Probabilities")
                        prob_df = pd.DataFrame({
                            'Sentiment': ['Positive', 'Negative', 'Neutral', 'Irrelevant'],
                            'Probability': probability
                        })
                        
                        fig = px.bar(
                            prob_df,
                            x='Sentiment',
                            y='Probability',
                            title="Sentiment Probability Distribution",
                            color='Sentiment',
                            color_discrete_map={
                                'Positive': '#00C851',
                                'Negative': '#ff4444',
                                'Neutral': '#ffbb33',
                                'Irrelevant': '#6c757d'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Error in prediction. Please try again.")
                else:
                    st.warning("Please enter some text to analyze.")
    
    elif page == "Data Analysis":
        st.header("📊 Data Analysis Dashboard")
        
        # Load data
        train_df, val_df = load_sample_data()
        
        if train_df is not None and val_df is not None:
            # Combine datasets
            combined_df = pd.concat([train_df, val_df], ignore_index=True)
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Tweets", f"{len(combined_df):,}")
            
            with col2:
                st.metric("Unique Companies", combined_df['company'].nunique())
            
            with col3:
                st.metric("Sentiment Classes", combined_df['sentiment'].nunique())
            
            with col4:
                avg_length = combined_df['text'].str.len().mean()
                st.metric("Avg Text Length", f"{avg_length:.0f} chars")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = create_sentiment_distribution_chart(combined_df)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = create_company_sentiment_chart(combined_df)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Text length analysis
            fig3 = create_text_length_analysis(combined_df)
            st.plotly_chart(fig3, use_container_width=True)
            
            # Sample data
            st.subheader("Sample Data")
            st.dataframe(combined_df.head(10))
            
        else:
            st.error("Data files not found! Please make sure 'twitter_training.csv' and 'twitter_validation.csv' are in the current directory.")
    
    elif page == "Model Training":
        st.header("🤖 Model Training")
        
        st.info("To train the model, run the following command in your terminal:")
        st.code("python sentiment_analysis.py", language="bash")
        
        st.markdown("""
        ### Training Process:
        1. **Data Loading**: Loads training and validation datasets
        2. **Text Preprocessing**: Cleans and preprocesses text data
        3. **Feature Extraction**: Uses TF-IDF vectorization
        4. **Model Training**: Trains a Random Forest classifier
        5. **Model Evaluation**: Evaluates on validation data
        6. **Model Saving**: Saves the trained model for future use
        
        ### Model Features:
        - **Text Cleaning**: Removes URLs, mentions, hashtags, special characters
        - **Stopword Removal**: Removes common English stopwords
        - **Lemmatization**: Reduces words to their base form
        - **TF-IDF Vectorization**: Converts text to numerical features
        - **Random Forest**: Ensemble method for robust predictions
        """)
        
        if st.button("Check Model Status"):
            model_loaded = analyzer.load_model('sentiment_model.pkl')
            if model_loaded:
                st.success("✅ Model is trained and ready to use!")
            else:
                st.warning("⚠️ Model not found. Please train the model first.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Twitter Sentiment Analysis App | Built with Streamlit & Scikit-learn</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
