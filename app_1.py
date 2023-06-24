import pandas as pd
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from io import BytesIO, StringIO
import datetime
import plotly.express as px

nltk.download('vader_lexicon')

st.title('Sentiment Analysis using nltk')

page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

session = st.session_state

column_names = ['First name', 'Last name', 'Gender', 'Date', 'Review', 'Sentiment']

if 'df' not in session:
    session['df'] = pd.DataFrame(columns=['Name', 'Date', 'Review', 'Sentiment'])

first_name = st.text_input('Please enter your name')
last_name = st.text_input('Please enter your last name')
date_input = st.date_input('Please enter the date', datetime.date(2023, 6, 24))
gender_input = st.selectbox('Please select your gender', ['Male', 'Female', 'Trans', 'Rather not say'])
user_input = st.text_area('Please enter your review')

sid = SentimentIntensityAnalyzer()
sentiment = ''
compound_score = 0  

if st.button('Analyze'):
    score = sid.polarity_scores(user_input)
    compound_score = score['compound']

    if compound_score < 0:
        st.write('Negative')
        sentiment = 'Negative'
    else:
        st.write('Positive')
        sentiment = 'Positive'

    entry = {'First name': first_name, 'Last name' : last_name,'Gender' : gender_input , 'Date': date_input, 'Review': user_input, 'Sentiment': sentiment}
    session['df'] = session['df'].append(entry, ignore_index=True)

if st.button('View data'):
    st.dataframe(session['df'][column_names], use_container_width=True)
    sentiment_counts = session['df']['Sentiment'].value_counts()
    fig = px.bar(sentiment_counts, x=sentiment_counts.index, y=sentiment_counts.values, labels={'x': 'Sentiment', 'y': 'Count'})
    st.plotly_chart(fig)

if st.button('Download CSV'):
    csv_data = session['df'][column_names].to_csv(index=False)
    csv_bytes = BytesIO(csv_data.encode())
    st.download_button('Click here to download', csv_bytes, file_name='data.csv', mime='text/csv')


