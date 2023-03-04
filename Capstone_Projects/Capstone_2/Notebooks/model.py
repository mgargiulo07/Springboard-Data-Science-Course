#basic python libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

#modeling 
from sklearn.neighbors import KNeighborsClassifier

#evaluation
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

#################################################################


#split our model 
def split(X,y,test_size=.30,random_state=0):
    return train_test_split(X,y,test_size=.30, random_state=0)

     

# add in transformers model
#create our list of unique data types
def transform_model(X_tr, X_ts, y_tr):   
    num_cols  = list(X_tr.select_dtypes(include='number').columns)
    cat_cols  = list(X_tr.select_dtypes(include='category').columns)
    sig_text  = ('signature')
    show_text =('showstopper')


    preprocesor = ColumnTransformer(
         transformers = [ 
            ('num',StandardScaler(),num_cols),                         #scale the numerical values
            ('cat',OneHotEncoder(handle_unknown = 'ignore'),cat_cols), #encode the categorical features
            ('text_sig',TfidfVectorizer(max_features=50),sig_text),    #freq counts for words
            ('text_show',TfidfVectorizer(max_features=50),show_text)   #freq counts for words
        ], remainder='passthrough')    


# fit and transform our train and test data
    X_tr = preprocesor.fit_transform(X_tr)
    X_ts = preprocesor.transform(X_ts)

 #add in our class resampling technique (smote)   
    smote=SMOTE()
    X_tr, y_tr = smote.fit_resample(X_tr,y_tr)

    return X_tr, X_ts, y_tr


#create a class that will build our model and evaluate it 
class build_model:
    def __init__(self,X_tr,X_ts,y_tr,y_ts):
        self.X_tr = X_tr
        self.X_ts = X_ts
        self.y_tr = y_tr
        self.y_ts = y_ts

#constructs knn model 
    def model(self):
        knn=KNeighborsClassifier(n_neighbors=2)
        knn.fit(self.X_tr,self.y_tr)
        prediction = knn.predict(self.X_ts)
        return prediction

#evaluates our model based on f1_score   
    def f1_score(self):
        score = f1_score(self.y_ts, self.model())
        return score 
