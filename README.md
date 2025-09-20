# 🐦 Twitter Sentiment Analysis

A comprehensive Twitter sentiment analysis application built with machine learning and Streamlit. This project analyzes Twitter data to predict sentiment (Positive, Negative, Neutral, Irrelevant) using various machine learning algorithms.

## 🚀 Features

- **Real-time Sentiment Prediction**: Analyze any text input for sentiment
- **Interactive Data Visualization**: Explore dataset with interactive charts
- **Multiple ML Models**: Support for Random Forest, Logistic Regression, and SVM
- **Text Preprocessing**: Advanced text cleaning and preprocessing pipeline
- **Model Training Interface**: Easy model training and evaluation
- **Responsive UI**: Beautiful and user-friendly Streamlit interface

## 📊 Dataset

The project uses Twitter sentiment analysis datasets:
- **Training Data**: `twitter_training.csv` (74,683 samples)
- **Validation Data**: `twitter_validation.csv` (1,759 samples)

### Dataset Structure
- **ID**: Unique identifier
- **Company**: Company/topic (Borderlands, Facebook, Amazon, etc.)
- **Sentiment**: Target variable (Positive, Negative, Neutral, Irrelevant)
- **Text**: Tweet content

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (done automatically on first run):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## 🚀 Usage

### 1. Train the Model

First, train the machine learning model:

```bash
python sentiment_analysis.py
```

This will:
- Load and preprocess the training data
- Train a Random Forest classifier
- Evaluate the model on validation data
- Save the trained model as `sentiment_model.pkl`

### 2. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📱 App Features

### 1. Predict Sentiment
- Enter any text for sentiment analysis
- View prediction confidence scores
- See probability distribution across all sentiment classes

### 2. Data Analysis
- Interactive data visualization dashboard
- Sentiment distribution charts
- Company-wise sentiment analysis
- Text length analysis
- Sample data exploration

### 3. Model Training
- Instructions for training the model
- Model status checking
- Training process explanation

## 🔧 Technical Details

### Text Preprocessing Pipeline
1. **Lowercase Conversion**: Convert text to lowercase
2. **URL Removal**: Remove HTTP/HTTPS URLs
3. **Mention/Hashtag Removal**: Remove @mentions and #hashtags
4. **Special Character Removal**: Remove non-alphabetic characters
5. **Stopword Removal**: Remove common English stopwords
6. **Lemmatization**: Reduce words to base form
7. **Tokenization**: Split text into individual words

### Machine Learning Pipeline
1. **Feature Extraction**: TF-IDF vectorization with n-grams
2. **Model Training**: Random Forest classifier (default)
3. **Model Evaluation**: Accuracy, classification report, confusion matrix
4. **Model Persistence**: Save/load trained models

### Available Models
- **Random Forest**: Default ensemble method
- **Logistic Regression**: Linear classification
- **Support Vector Machine**: Kernel-based classification

## 📈 Performance

The model typically achieves:
- **Accuracy**: 85-90% on validation data
- **Precision/Recall**: Balanced across all sentiment classes
- **Training Time**: 2-5 minutes depending on hardware

## 🎨 Customization

### Adding New Models
To add a new model, modify the `train_model` method in `sentiment_analysis.py`:

```python
elif model_type == 'your_model':
    self.model = YourModelClass()
```

### Modifying Text Preprocessing
Edit the `clean_text` method in both files to customize preprocessing steps.

### UI Customization
Modify the CSS in `streamlit_app.py` to change the appearance.

## 📁 Project Structure

```
twitter-sentiment-analysis/
├── twitter_training.csv          # Training dataset
├── twitter_validation.csv        # Validation dataset
├── sentiment_analysis.py         # ML model training script
├── streamlit_app.py             # Streamlit web application
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── sentiment_model.pkl          # Trained model (generated)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🐛 Troubleshooting

### Common Issues

1. **Model not found error**:
   - Make sure to run `python sentiment_analysis.py` first
   - Check that `sentiment_model.pkl` exists in the project directory

2. **NLTK data not found**:
   - The app will automatically download required NLTK data
   - If issues persist, manually download: `nltk.download('all')`

3. **Memory issues with large datasets**:
   - Reduce the dataset size for testing
   - Increase system RAM or use cloud computing

4. **Streamlit not starting**:
   - Check if port 8501 is available
   - Try `streamlit run streamlit_app.py --server.port 8502`

## 📞 Support

For questions or issues, please:
1. Check the troubleshooting section
2. Review the code comments
3. Create an issue with detailed error information

## 🔮 Future Enhancements

- [ ] Real-time Twitter API integration
- [ ] Advanced deep learning models (LSTM, BERT)
- [ ] Sentiment trend analysis over time
- [ ] Multi-language support
- [ ] Model comparison dashboard
- [ ] Export functionality for predictions
- [ ] Batch processing capabilities

---

**Happy Analyzing! 🎉**
