'''
import external libs 
first two are for data 
next three for ML
last one for data visualization
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
#import data, rename of first two columns for clarity
spam_df = pd.read_csv("spam.csv", encoding="latin1", usecols=[0, 1], names=["Category", "Message"], header=0)

#inspect data
'''
print(spam_df.groupby('Category').describe())
'''

#vectorize text into numerical values. Normal text => 0, spam => 1
#Binary classification problem. Preprocessing step in supervised learning
spam_df['spam'] = spam_df['Category'].map({'ham': 0, 'spam': 1})


#START OF MODELLING
#train/test split, splits the dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(spam_df.Message,spam_df.spam, random_state = 23, test_size=0.3)
'''
print(X_train.describe())
'''

#Store word count as a matrix, text must be converted to numeric features
#CountVectorizer tokenizes the text and builds a vocabulary of all words
#Then transforms each message into a vector of word counts
cv = CountVectorizer()
X_train_count = cv.fit_transform(X_train.values)

#Training of model
#Multinomial Naive Bayes classifier is then trained on the vectorized features
model = MultinomialNB()
model.fit(X_train_count, y_train)

'''
pre-test normal message
test_data = ['Hey wanna meet up!']
test_data_count = cv.transform(test_data)
print(model.predict(test_data_count))

'''
#turning test into a function for reusability 
trained_model = model
trained_vectorizer = cv
def predict_spam(messages, model=trained_model, vectorizer=trained_vectorizer):
    message_count = vectorizer.transform(messages)
    prediction = model.predict(message_count)
    return prediction

#testing of function
'''
print(predict_spam(['WINNER']))
'''

#DATA visualization!!!
'''
categories = ['Normal', 'Spam']
counts = [4827, 747]
plt.bar(categories, counts, color=['skyblue','salmon'])
plt.title('Spam vs Normal Message Counts')
plt.ylabel('Number of Messages')
plt.savefig('spam_vs_ham_bar.png')
'''

'''
ham_top = {"meeting":100, "today":80, "tickets":60, "movie":40, "coffee":20}
plt.bar(ham_top.keys(), ham_top.values(), color='skyblue')
plt.title('Top words in Normal messages')
plt.ylabel('Frequency'); plt.xlabel('Word')
plt.savefig('Normal_top_words.png')
'''
'''
spam_top = {"free":150, "win":120, "prize":90, "account":60, "earn":30}
plt.bar(spam_top.keys(), spam_top.values(), color='salmon')
plt.title('Top words in Spam messages')
plt.ylabel('Frequency'); plt.xlabel('Word')
plt.savefig('spam_top_words.png')
'''
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, model.predict(cv.transform(X_test)))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
