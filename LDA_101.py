import pandas as pd
from sklearn.datasets import fetch_20newsgroups
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
docs = dataset.data
# print(len(docs))

news_df = pd.DataFrame({'document':docs})

news_df['clean_doc'] = news_df['document'].str.replace('^a-zA-Z]', " ")
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x : x.lower())

# 불용어 제거
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
tokenized_doc = news_df['clean_doc'].apply(lambda x : x.split())
tokenized_doc = tokenized_doc.apply(lambda x : [item for item in x if item not in stop_words])
