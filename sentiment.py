import pandas as pd
import streamlit as st
import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import altair as alt


# hide the side menu from streamlit
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# add title with markdown
'''
# Reddit NFL VADER Sentiment Analysis
'''
# add dropdown box with the subs as the choices
option = st.sidebar.selectbox('Subreddit to Query', ('NFL', 'Patriots', 'eagles', 'GreenBayPackers',
                                                     'minnesotavikings', 'Seahawks', 'cowboys', '49ers', 'CHIBears',
                                                     'LosAngelesRams', 'steelers', 'Browns', 'DenverBroncos',
                                                     'detroitlions', 'NYGiants', 'KansasCityChiefs', 'falcons', 'Saints',
                                                     'bengals', 'panthers', 'buffalobills', 'Texans', 'ravens',
                                                     'washingtonNFL', 'raiders', 'nyjets', 'miamidolphins', 'Colts',
                                                     'AZCardinals', 'Jaguars', 'Chargers', 'Tennesseetitans',
                                                     'buccaneers'))
# add the slider
n_headlines = st.sidebar.slider('How many headlines to retrieve?', min_value=2, max_value=1000, step=1, value=50)

reddit = praw.Reddit(client_id='[your client id]',
                     client_secret='[your secret]',
                     user_agent='[username]')


# function for getting results and performing the sentiment analysis
def retrieve_results(option_in, num_results):
    headlines = set()

    # use the sentiment intensity analyzer (SIA) to categorize headlines and provide polarity scores
    # `compound` ranges from -1 (Extremely Negative) to 1 (Extremely Positive).
    sia = SIA()
    results = []

    for submission in reddit.subreddit(option_in).hot(limit=num_results):
        # print(submission.comments.list())
        headlines.add(submission.title)

    for line in headlines:
        pol_score = sia.polarity_scores(line)
        pol_score['headline'] = line
        results.append(pol_score)

    df_headline_labels = pd.DataFrame.from_records(results)

    # select the min and max for the compound score
    most_neg_out = df_headline_labels.loc[df_headline_labels['compound'] == df_headline_labels['compound'].min(), ['headline', 'compound']]
    most_pos_out = df_headline_labels.loc[df_headline_labels['compound'] == df_headline_labels['compound'].max(), ['headline', 'compound']]

    return most_pos_out, most_neg_out, df_headline_labels


most_pos, most_neg, df_headlines_out = retrieve_results(option_in=option, num_results=n_headlines)

# add markdown to show which sub is selected
'## Subreddit Selected: ', option

st.sidebar.write('Analyzing the top ' + str(n_headlines) + ' headlines')

'''
## **Most Positive Headline** 
'''
st.write(most_pos['headline'].values.item())
st.write('Compound Score: ', str(most_pos['compound'].values.item()))

'''
## **Most Negative Headline**
'''
st.write(most_neg['headline'].values.item())
st.write('Compound Score: ', str(most_neg['compound'].values.item()))

# use altair to plot the results
c = alt.Chart(df_headlines_out).mark_bar().encode(
    alt.X("compound:Q", bin=True),
    y='count()',
)

'''
## **VADER Compound Score Distribution**
'''

st.altair_chart(c, use_container_width=True)

