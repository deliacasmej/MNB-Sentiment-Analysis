

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#using the preprocessed kindle review data from kaggle
data = pd.read_csv('pkr.csv')

data.head()

#dropping more columns since they won't be aused
data=data.drop(columns=['summary','Unnamed: 0'])

#replacing rating values where 1-3 equal neg (negative)
#and 4-5 equal pos (positive)
data.rating = data.rating.replace([1,2,3], 'neg')
data.rating = data.rating.replace([4,5],'pos')

data.head()

X = data['reviewText']
y = data['rating']

#splitting my dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

positive_text = ' '.join(data[data['rating'] == 'pos']['reviewText'])
positive_wordcloud = WordCloud(width=800, height=400, max_words=50, background_color='white', random_state=42).generate(positive_text)

neg_text = ' '.join(data[data['rating'] == 'neg']['reviewText'])
neg_wordcloud = WordCloud(width=800, height=400, max_words=50, background_color='white', random_state=42).generate(neg_text)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Positive Messages')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(neg_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Negative Messages')
plt.axis('off')


plt.show()

class_distribution = data['rating'].value_counts()
class_distribution.plot(kind='pie', autopct='%1.1f%%', colors=['#FF69B4','#FFC0CB'])
plt.title('Distribution of Rating Type in Kindle Review')
plt.show()

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

mnb = MultinomialNB(alpha=0.8, fit_prior=True, force_alpha=True)
mnb.fit(X_train_vec, y_train)

