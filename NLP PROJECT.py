#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import string, nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')



# In[2]:


nltk.download('omw-1.4')


# In[3]:


df = pd.read_csv('fake reviews dataset.csv')
df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df['rating'].value_counts()


# In[8]:


plt.figure(figsize=(15,8))
labels = df['rating'].value_counts().keys()
values = df['rating'].value_counts().values
explode = (0.1,0,0,0,0)
plt.pie(values,labels=labels,explode=explode,shadow=True,autopct='%1.1f%%')
plt.title('Proportion of each rating',fontweight='bold',fontsize=25,pad=20,color='crimson')
plt.show()


# In[9]:


def clean_text(text):
    nopunc = [w for w in text if w not in string.punctuation]
    nopunc = ''.join(nopunc)
    return  ' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])


# In[10]:


df['text_'][0], clean_text(df['text_'][0])


# In[11]:


df['text_'].head().apply(clean_text)


# In[12]:


df.shape


# In[13]:


#df['text_'] = df['text_'].apply(clean_text)


# In[14]:


df['text_'] = df['text_'].astype(str)


# In[15]:


def preprocess(text):
    return ' '.join([word for word in word_tokenize(text) if word not in stopwords.words('english') and not word.isdigit() and word not in string.punctuation])


# In[16]:


preprocess(df['text_'][4])


# In[17]:


df['text_'][:10000] = df['text_'][:10000].apply(preprocess)


# In[19]:


df['text_'][10001:20000] = df['text_'][10001:20000].apply(preprocess)


# In[20]:


df['text_'][20001:30000] = df['text_'][20001:30000].apply(preprocess)


# In[21]:


df['text_'][30001:32750] = df['text_'][30001:32750].apply(preprocess)


# In[22]:


df['text_'] = df['text_'].str.lower()


# In[23]:


stemmer = PorterStemmer()
def stem_words(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])
df['text_'] = df['text_'].apply(lambda x: stem_words(x))


# In[24]:


lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
df["text_"] = df["text_"].apply(lambda text: lemmatize_words(text))


# In[25]:


df['text_'].head()


# In[26]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# In[27]:


df.to_csv('Preprocessed dataset_s.csv')


# In[28]:


df = pd.read_csv('Preprocessed dataset_s.csv')
df.head()


# In[29]:


df.drop('Unnamed: 0',axis=1,inplace=True)


# In[30]:


df.head()


# In[31]:


df.dropna(inplace=True)


# In[32]:


df['length'] = df['text_'].apply(len)


# In[33]:


df.info()


# In[34]:


plt.hist(df['length'],bins=50)
plt.show()


# In[35]:


df.groupby('label').describe()


# In[36]:


df.hist(column='length',by='label',bins=50,color='blue',figsize=(12,5))
plt.show()


# In[37]:


df[df['label']=='OR'][['text_','length']].sort_values(by='length',ascending=False).head().iloc[0].text_


# In[38]:


df.length.describe()


# In[39]:


def text_process(review):
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[40]:


bow_transformer = CountVectorizer(analyzer=text_process)
bow_transformer


# In[41]:


bow_transformer.fit(df['text_'])
print("Total Vocabulary:",len(bow_transformer.vocabulary_))


# In[42]:


review4 = df['text_'][3]
review4


# In[43]:


bow_msg4 = bow_transformer.transform([review4])
print(bow_msg4)
print(bow_msg4.shape)


# # There are 6 unique words in the 4th review.

# In[44]:


print(bow_transformer.get_feature_names_out()[15841])
print(bow_transformer.get_feature_names_out()[23848])


# In[45]:


bow_reviews = bow_transformer.transform(df['text_'])


# In[46]:


print("Shape of Bag of Words Transformer for the entire reviews corpus:",bow_reviews.shape)
print("Amount of non zero values in the bag of words model:",bow_reviews.nnz)


# In[47]:


print("Sparsity:",np.round((bow_reviews.nnz/(bow_reviews.shape[0]*bow_reviews.shape[1]))*100,2))


# In[48]:


tfidf_transformer = TfidfTransformer().fit(bow_reviews)
tfidf_rev4 = tfidf_transformer.transform(bow_msg4)
print(bow_msg4)


# In[49]:


print(tfidf_transformer.idf_[bow_transformer.vocabulary_['mango']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['book']])


# In[50]:


tfidf_reviews = tfidf_transformer.transform(bow_reviews)
print("Shape:",tfidf_reviews.shape)
print("No. of Dimensions:",tfidf_reviews.ndim)


# # Creating training and testing data

# In[51]:


review_train, review_test, label_train, label_test = train_test_split(df['text_'],df['label'],test_size=0.35)


# In[52]:


pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB())
])


# # Training and testing Multinomial Naive Bayes Algorithm on the preprocessed data

# In[53]:


pipeline.fit(review_train,label_train)


# In[54]:


predictions = pipeline.predict(review_test)
predictions


# In[55]:


print('Classification Report:',classification_report(label_test,predictions))
print('Confusion Matrix:',confusion_matrix(label_test,predictions))
print('Accuracy Score:',accuracy_score(label_test,predictions))


# In[56]:


print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,predictions)*100,2)) + '%')


# In[57]:


pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',RandomForestClassifier())
])


# In[58]:


pipeline.fit(review_train,label_train)


# In[59]:


rfc_pred = pipeline.predict(review_test)
rfc_pred


# In[60]:


print('Classification Report:',classification_report(label_test,rfc_pred))
print('Confusion Matrix:',confusion_matrix(label_test,rfc_pred))
print('Accuracy Score:',accuracy_score(label_test,rfc_pred))
print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,rfc_pred)*100,2)) + '%')


# In[61]:


pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',DecisionTreeClassifier())
])


# In[62]:


pipeline.fit(review_train,label_train)


# In[63]:


dtree_pred = pipeline.predict(review_test)
dtree_pred


# In[64]:


print('Classification Report:',classification_report(label_test,dtree_pred))
print('Confusion Matrix:',confusion_matrix(label_test,dtree_pred))
print('Accuracy Score:',accuracy_score(label_test,dtree_pred))
print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,dtree_pred)*100,2)) + '%')


# In[65]:


pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',KNeighborsClassifier(n_neighbors=2))
])


# In[66]:


pipeline.fit(review_train,label_train)


# In[67]:


knn_pred = pipeline.predict(review_test)
knn_pred


# In[68]:


print('Classification Report:',classification_report(label_test,knn_pred))
print('Confusion Matrix:',confusion_matrix(label_test,knn_pred))
print('Accuracy Score:',accuracy_score(label_test,knn_pred))
print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,knn_pred)*100,2)) + '%')


# In[69]:


pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',SVC())
])


# In[70]:


pipeline.fit(review_train,label_train)


# In[71]:


svc_pred = pipeline.predict(review_test)
svc_pred


# In[72]:


print('Classification Report:',classification_report(label_test,svc_pred))
print('Confusion Matrix:',confusion_matrix(label_test,svc_pred))
print('Accuracy Score:',accuracy_score(label_test,svc_pred))
print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,svc_pred)*100,2)) + '%')


# In[73]:


pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',LogisticRegression())
])


# In[74]:


pipeline.fit(review_train,label_train)


# In[75]:


lr_pred = pipeline.predict(review_test)
lr_pred


# In[76]:


print('Classification Report:',classification_report(label_test,lr_pred))
print('Confusion Matrix:',confusion_matrix(label_test,lr_pred))
print('Accuracy Score:',accuracy_score(label_test,lr_pred))
print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,lr_pred)*100,2)) + '%')


# # Conclusion

# In[78]:


print('Performance of various ML models:')
print('\n')
print('Logistic Regression Prediction Accuracy:',str(np.round(accuracy_score(label_test,lr_pred)*100,2)) + '%')
print('K Nearest Neighbors Prediction Accuracy:',str(np.round(accuracy_score(label_test,knn_pred)*100,2)) + '%')
print('Decision Tree Classifier Prediction Accuracy:',str(np.round(accuracy_score(label_test,dtree_pred)*100,2)) + '%')
print('Random Forests Classifier Prediction Accuracy:',str(np.round(accuracy_score(label_test,rfc_pred)*100,2)) + '%')
print('Support Vector Machines Prediction Accuracy:',str(np.round(accuracy_score(label_test,svc_pred)*100,2)) + '%')
print('Multinomial Naive Bayes Prediction Accuracy:',str(np.round(accuracy_score(label_test,predictions)*100,2)) + '%')


# In[30]:


get_ipython().system('pip install torch')


# In[32]:


get_ipython().system('pip install torchmetrics')


# In[78]:


pip install transformers


# In[120]:


import torch
from transformers import BertTokenizer, BertForSequenceClassification


# In[121]:


# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'  
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  


# In[122]:


def tokenize_text(text):
    return tokenizer.encode_plus(text, add_special_tokens=True, max_length=128, padding='max_length', truncation=True, return_tensors='pt')

def predict_with_bert(text):
    inputs = tokenize_text(text)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_label = torch.argmax(probabilities, dim=1).item()
    return predicted_label


# In[123]:


# Function to preprocess text
def preprocess(text):
    # Tokenize the text and filter out stopwords, digits, and punctuation
    return ' '.join([word for word in word_tokenize(text) 
                     if word.lower() not in stopwords.words('english') 
                     and not word.isdigit() 
                     and word not in string.punctuation])


# In[125]:


from sklearn.metrics import accuracy_score

#Example evaluation
accuracy = accuracy_score(df['label'], df['predicted_label'])
print(f"BERT Model Accuracy: {accuracy}%")


# In[ ]:





# In[ ]:




