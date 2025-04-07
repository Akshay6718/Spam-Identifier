import pandas as pd
import langdetect
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#drops unnecessary columns
data = pd.read_csv("spam.csv", encoding='ISO-8859-1')
data.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)  #deleting empty columns


def get_lang(text):  #gets the language of an email
    try:
        return langdetect.detect(text)
    except:
        return "unknown"


#filters for english messages
data['lang'] = data["v2"].apply(get_lang)
pos = data[data['lang'] != "en"].index
english_data = data.drop(index=pos)

english_data['v1'].apply(lambda x: 1 if x == 'spam' else 0)
english_data.rename(columns={'v1': 'label', 'v2': "text"}, inplace=True)

#preprocessing
x_train, x_test, y_train, y_test = train_test_split(english_data["text"], english_data["label"], test_size=0.2)
list = x_train.to_list()
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
transformed_train = vectorizer.fit_transform(list)
transformed_test = vectorizer.transform(x_test)

#fitting and training the model
model = RandomForestClassifier()
model.fit(transformed_train, y_train)
y_pred = model.predict(transformed_test)
acc = accuracy_score(y_pred, y_test)
print(acc)
