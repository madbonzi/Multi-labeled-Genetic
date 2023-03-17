import numpy as np
# =============================================================================
# from skmultilearn.adapt import MLkNN
# =============================================================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# =============================================================================
# from scipy.sparse import csr_matrix, lil_matrix
# =============================================================================
from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression

import pandas as pd


aspects_df = pd.read_csv('Multilabel_text.csv')
aspects_df.head()

x = aspects_df["text"]
y = np.asarray(aspects_df[aspects_df.columns[1:]])

vetorizar = TfidfVectorizer(max_features=3000, max_df=0.85)
vetorizar.fit(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

X_train_tfidf = vetorizar.transform(X_train)
X_test_tfidf = vetorizar.transform(X_test)

       
    # initialize classifier chains multi-label classifier
classifier = ClassifierChain(LogisticRegression())
    # Training logistic regression model on train data
classifier.fit(X_train_tfidf, y_train)
    # predict
predictions = classifier.predict( X_test_tfidf)
    # accuracy
accuracy = accuracy_score(y_test,predictions)


print("accuracy - Without GA:",accuracy) # (accuracy = objective func.)




