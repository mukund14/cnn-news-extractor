import streamlit as st
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from gtts import gTTS
import os

st.title('CNN News Extractor and Audio Converter')
st.write('Extract CNN news articles and convert them to audio files using Python.')

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

# Main Streamlit app
def main():
    st.header('Step 1: Retrieve CNN Data')
    if st.button('Fetch CNN Homepage'):
        homepage_html = get_cnn_homepage()
        st.success('CNN homepage fetched successfully!')

        st.header('Step 2: Extract Article Links')
        article_links = extract_article_links(homepage_html)
        st.write(f'Found {len(article_links)} articles.')

        st.header('Step 3: Extract Article Details')
        articles = []
        for link in article_links:
            try:
                article_details = extract_article_details(link)
                articles.append(article_details)
                st.write(f"Extracted: {article_details['title']}")
            except Exception as e:
                st.error(f"Failed to process {link}: {e}")

        st.header('Step 4: Convert Articles to Audio')
        for i, article in enumerate(articles):
            try:
                filename = f"article_{i + 1}.mp3"
                text_to_speech(article['text'], filename)
                st.audio(filename)
                st.success(f"Audio content written to file '{filename}' for article '{article['title']}'")
            except Exception as e:
                st.error(f"Failed to convert article '{article['title']}' to audio: {e}")

if __name__ == "__main__":
    main()
