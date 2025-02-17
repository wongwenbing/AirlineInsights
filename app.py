import streamlit as st
import dill
import nltk
import pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import wordnet
import re
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from textblob import TextBlob
from tqdm import tqdm
import numpy as np
import string
from sklearn.pipeline import Pipeline

tqdm.pandas()
# Download necessary NLTK resources (only if not already downloaded)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Title of the app
st.title("Airlines Prediction")

# Load the saved pipeline
with open('text_preprocessing.pkl', 'rb') as f: 
    text_preprocessing = dill.load(f)

with open('feature_engineering.pkl', 'rb') as f:
    feature_engineering_pipeline = dill.load(f)

with open('logreg.pkl', 'rb') as f: 
    model = pickle.load(f) 
with open('cv.pkl', 'rb') as f: 
    cv = pickle.load(f)
with open('scaling.pkl','rb') as f:
    scaler=pickle.load(f) 

# load functions 
#start off with converting to lowercase and remove any additional whitespaces
def lowercase(text): 
    text=text.lower()
    text=re.sub(r'\s\s+', ' ', text)
    return text

#remove starting words which are not relevant like trip verified/verified review 
def filter_startwords(text):
    text=re.sub(r'([^a-zA-Z]+trip verified[^a-zA-Z]+)', '', text)
    text=re.sub(r'([^a-zA-Z]+verified review[^a-zA-Z]+)', '', text)
    return text

def remove_stopwords(x): 
    stopwordss=stopwords.words('english')
    stopwordss.extend(['would','u', 'not'])
    words=word_tokenize(x)
    text=[]
    for word in words: 
        if word not in stopwordss: 
            text.append(word)
    return ' '.join(text)

def remove_punctuation(x):
    punct=list(string.punctuation)
    text=[]
    words=word_tokenize(x)
    for word in words: 
        if word not in punct: 
            text.append(word)
    return ' '.join(text)

def remove_others_lemmatise(x): 
    lemma=WordNetLemmatizer() #lemmatization
    wordslist=['``', '`']
    words=word_tokenize(x)
    txt=[]
    for word in words: 
        if word not in wordslist: 
            if word == "n't": 
                word='not'
                txt.append(word)
            else:
                word=lemma.lemmatize(word)
                txt.append(word)
    return ' '.join(txt)

def categorize_rating(rating):
    if rating <= 2:
        return 'Negative'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Positive'

def feature_engineering(df): 
    df['word_count']=df['cleaned_text'].apply(lambda x:len(str(x).split()))
    df['char_count']=df['cleaned_text'].apply(lambda x:len(x))
    df['avg_word_length'] = df['char_count'] / (df['word_count'] + 1)  # Avoid division by zero
    df['sentiment_score'] = df['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity_score'] = df['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    df['rating_category'] = df['rating'].apply(categorize_rating)
    df['sentiment_agreement'] = df.apply(lambda row: 1 if (row['sentiment_score'] > 0 and row['rating'] > 3) or
                                                       (row['sentiment_score'] < 0 and row['rating'] <= 2) 
                                          else 0, axis=1)
    return df

label_encoding=LabelEncoder()

def label_encoding(df): 
    df['rating_encoded']=label_encoding.transform(df['rating_category'])
    return df

# Input fields for user data
st.header("Enter Review Details:")

# Text input
user_text = st.text_area("Write your review here:", "We love Singapore Airlines")

# Rating input (range 1-5)
user_rating = st.slider("Rate the product (1-5):", min_value=1, max_value=5, value=3)

# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    'Text': [user_text],
    'rating': [user_rating]
})
print(input_data)
label_mapping={0: 'British Airways',
 1: 'Emirates',
 2: 'Ryanair',
 3: 'Scoot',
 4: 'Singapore Airlines',
 5: 'Southwest Airlines'}


# Prediction button
if st.button("Predict Sentiment"):
    # Use the pipeline to preprocess the input data and make predictions
    input_data['cleaned_text']=text_preprocessing.transform(input_data['Text'])
    test=feature_engineering_pipeline.transform(input_data)
    print(test)
    testset = test.drop(['Text', 'rating','rating_category'],axis=1)
    test_x=  testset[['cleaned_text','sentiment_score', 'subjectivity_score']]
    testx_text = test_x['cleaned_text']
    testx_numeric = test_x.drop('cleaned_text',axis=1)
    print(testx_numeric)
    testx_numeric_scaled=scaler.transform(testx_numeric)
    testx_numeric_scaled = pd.DataFrame(testx_numeric_scaled, columns=testx_numeric.columns, index=testx_numeric.index)
    testx_numeric = testx_numeric_scaled.reset_index()
    testx_cv = cv.transform(testx_text)
    testx_vectorised = pd.DataFrame(testx_cv.toarray(), columns=cv.get_feature_names_out())
    testx_vectorised.reset_index(drop=True,inplace=True)
    testing_set_x  = pd.concat([testx_vectorised,testx_numeric],axis=1)
    testing_set_x.drop('index',axis=1, inplace=True)
    # Step 3: Make predictions using the model
    prediction = model.predict(testing_set_x)
        
        # Map the numerical prediction to a label
    predicted_label = label_mapping[prediction[0]]
    # Display the result
    st.success(predicted_label)   
