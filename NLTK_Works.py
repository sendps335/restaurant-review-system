import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

""" Some Pratice Lol """
sentence='Hello!!, How are you,Baby?'
st=word_tokenize(sentence)
cv=CountVectorizer()
word_cv=cv.fit_transform(st)
print(cv.vocabulary_)
print(st)
print(word_cv)


cv2=CountVectorizer(tokenizer=word_tokenize,token_pattern=None)
word_cv2=cv2.fit_transform(st)
print(word_cv2)
print(cv2.vocabulary_)

""" Real Work Starts """

df=pd.read_csv(r'C:\Users\DEBIPRASAD\Desktop\Git\Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
print(df.head())

""" Linear Model """
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

df['kfold']=-1
kf=StratifiedKFold(n_splits=5)
#Shuffle Randomly
df.sample(frac=1).reset_index(drop=True)
y=df.Liked.values
for t,(f_,v_) in enumerate(kf.split(X=df,y=y)):
    df.loc[v_,'kfold']=t

for fold in range(5):
    df_train=df[df['kfold']!=fold].reset_index(drop=True)
    df_cross=df[df['kfold']==fold].reset_index(drop=True)
    
    cv_vec=CountVectorizer(tokenizer=word_tokenize,token_pattern=None)
    cv_vec.fit(df_train.Review.values)
    xtrain=cv_vec.transform(df_train.Review.values)
    xcross=cv_vec.transform(df_cross.Review.values)
    
    lr=LogisticRegression()
    lr.fit(xtrain,df_train.Liked.values)
    y_train_pred=lr.predict(xtrain)
    y_cross_pred=lr.predict(xcross)
    
    accuracy1=metrics.accuracy_score(df_train.Liked.values,y_train_pred)
    accuracy2=metrics.accuracy_score(df_cross.Liked.values,y_cross_pred)
    f1=metrics.f1_score(df_cross.Liked.values,y_cross_pred)
    
    print(f"Fold : {fold}")
    print(f"Training Set Accuracy : {accuracy1}")
    print(f"Cross_Validation Set Accuracy : {accuracy2}")
    print(f"Cross_Validation Set F1 Score : {f1}")
    print("")
    
""" End of the Linear Model """