import streamlit as st
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from gtts import gTTS
import os
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(page_title="CNN News Extractor", layout="wide")

def main():
    st.sidebar.title("CNN News Extractor and Audio Converter")
    st.sidebar.write("Extract CNN news articles, convert them to audio files, and visualize topics using Python.")
    
    # Step 1: Retrieve CNN Data
    st.sidebar.header('Step 1: Retrieve CNN Data')
    fetch_button = st.sidebar.button('Fetch CNN Homepage')
    if fetch_button:
        with st.spinner('Fetching CNN homepage...'):
            homepage_html = get_cnn_homepage()
        st.success('CNN homepage fetched successfully!')
    
        # Step 2: Extract Article Links
        st.sidebar.header('Step 2: Extract Article Links')
        with st.spinner('Extracting article links...'):
            article_links = extract_article_links(homepage_html)
        st.sidebar.write(f'Found {len(article_links)} articles.')
    
        # Step 3: Extract Article Details
        st.sidebar.header('Step 3: Extract Article Details')
        articles = []
        with st.spinner('Extracting article details...'):
            for link in article_links:
                try:
                    article_details = extract_article_details(link)
                    article_details['url'] = link  # Add the URL to the article details
                    articles.append(article_details)
                except Exception as e:
                    st.sidebar.error(f"Failed to process {link}: {e}")
    
        st.sidebar.write(f'Extracted details of {len(articles)} articles.')
    
        # Step 4: Convert Articles to Audio
        st.sidebar.header('Step 4: Convert Articles to Audio')
        if st.sidebar.button('Convert Articles to Audio'):
            audio_files = []
            with st.spinner('Converting articles to audio...'):
                for i, article in enumerate(articles):
                    try:
                        filename = f"article_{i + 1}.mp3"
                        text_to_speech(article['text'], filename)
                        audio_files.append((article['title'], filename, article['url']))
                    except Exception as e:
                        st.error(f"Failed to convert article '{article['title']}' to audio: {e}")
            st.sidebar.success('Audio conversion complete!')
    
            # Display articles and audio files
            st.header('Extracted Articles')
            for title, filename, url in audio_files:
                with st.expander(title):
                    st.markdown(f"[Read the full article on CNN]({url})")
                    st.audio(filename)
    
            # Step 5: Visualize Topics
            st.header('Visualize Topics')
            if st.button('Generate Topic Model'):
                article_texts = [article['text'] for article in articles]
                if article_texts:
                    with st.spinner('Performing topic modeling...'):
                        vis = topic_modeling(article_texts)
                    st.success('Topic modeling complete!')
                    components.html(pyLDAvis.prepared_data_to_html(vis), height=800, scrolling=True)

# Function to get the CNN homepage
def get_cnn_homepage():
    url = "https://www.cnn.com"
    response = requests.get(url)
    return response.text

# Function to extract article links from the homepage
def extract_article_links(html):
    soup = BeautifulSoup(html, 'html.parser')
    article_links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('/'):
            full_url = f"https://www.cnn.com{href}"
            if '/2024/' in full_url and full_url not in article_links:  # Filter for articles by date
                article_links.append(full_url)
    return article_links

# Function to extract article details using newspaper3k
def extract_article_details(url):
    article = Article(url)
    article.download()
    article.parse()
    return {
        'title': article.title,
        'authors': article.authors,
        'publish_date': article.publish_date,
        'text': article.text
    }

# Function to convert text to speech and save as MP3
def text_to_speech(text, filename):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    return filename

# Function to preprocess text for topic modeling
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
    return tokens

# Function to perform topic modeling and visualize topics
def topic_modeling(texts):
    texts = [preprocess(text) for text in texts]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    return vis

if __name__ == "__main__":
    main()
