import numpy as np
import pandas as pd
import seaborn as sns
import re
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from wordcloud import WordCloud
from PIL import Image
import shap

#Load the data
fake_news = pd.read_csv("C:\\Users\\Ümitcan\\Desktop\\Fake_news_detection\\fake.csv")
real_news = pd.read_csv("C:\\Users\\Ümitcan\\Desktop\\Fake_news_detection\\true.csv")

#Add a column to the dataframes to indicate the class
fake_news['class'] = 0
real_news['class'] = 1

#Concatenate the dataframes
df = pd.concat([fake_news,real_news],axis= 0)

fig = plt.figure(figsize=(10,5))

#Plot the count of fake and true news
graph = sns.countplot(x="class", data=df)
plt.title("Count of Fake and True News")

#removing boundary
graph.spines["right"].set_visible(False)
graph.spines["top"].set_visible(False)
graph.spines["left"].set_visible(False)

#annoting bars with the counts  
for p in graph.patches:
        height = p.get_height()
        graph.text(p.get_x()+p.get_width()/2., height + 0.2,height ,ha="center",fontsize=12)



#Plot the count of fake and true news by subject
fig = plt.figure(figsize=(10,5))

graph = sns.countplot(x="subject", data=df)
plt.title("Count of Subjecs")

#removing boundary
graph.spines["right"].set_visible(False)
graph.spines["top"].set_visible(False)
graph.spines["left"].set_visible(False)
 
for p in graph.patches:
        height = p.get_height()
        graph.text(p.get_x()+p.get_width()/2., height + 0.2,height ,ha="center",fontsize=12)

#Plot the word cloud for fake news
newspaper="C:\\Users\\Ümitcan\\Desktop\\Fake_news_detection\\news.png"
icon=Image.open(newspaper)
mask=Image.new(mode="RGB",size=icon.size, color=(255,255,255))
mask.paste(icon, box=icon)

rgb_array=np.array(mask)
plt.figure(figsize = (10,10))
Wc = WordCloud(mask=rgb_array,max_words = 2000 , width = 1600 ,
               height = 800)

Wc.generate(" ".join(df[df['class']== 0].text))
plt.axis("off")
plt.imshow(Wc , interpolation = 'bilinear')

#Plot the word cloud for true news
thumb="C:\\Users\\Ümitcan\\Desktop\\Fake_news_detection\\thumbs-up.png"
icon=Image.open(thumb)
mask=Image.new(mode="RGB",size=icon.size, color=(255,255,255))
mask.paste(icon, box=icon)

rgb_array1=np.array(mask)
plt.figure(figsize = (10,10))
Wc = WordCloud(mask=rgb_array1,max_words = 2000 , width = 1600 ,
               height = 800)

Wc.generate(" ".join(df[df['class']== 1].text))
plt.axis("off")
plt.imshow(Wc , interpolation = 'bilinear')

#Drop the columns that are not needed
df = df.drop(['title','subject','date'],axis=1)

#Check for missing values
df.isnull().sum()

#Remove the rows with missing values
df= df[df['text'] != ' ']

#Shuffle the data
df = df.sample(frac = 1)

#Reset the index
df.reset_index(inplace = True)
df.drop(['index'], axis = 1, inplace = True)


#Preprocess the text
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#Apply the function to the text column
df['text'] = df['text'].apply(wordopt)

#Split the data into X and y
X = df['text']
y = df['class']

#Split the data into training and testing sets for machine learning models
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.33,random_state=42)

#Vectorize the text
from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer(max_features=500)
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
xv_train_dense = xv_train.toarray()
xv_test_dense = xv_test.toarray()

#Logistic Regression
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(xv_train, y_train)

pred_lr = LR.predict(xv_test)
print("Logistic Regression: ")
print("Accuracy: ",accuracy_score(y_test, pred_lr))
print(classification_report(y_test, pred_lr))

#Desicion Tree
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)
print("Decision Tree: ")
print("Accuracy: ",accuracy_score(y_test, pred_dt))
print(classification_report(y_test, pred_dt))

#Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
GB = GradientBoostingClassifier(random_state=0)
GB.fit(xv_train, y_train)
pred_gb = GB.predict(xv_test)
print("Gradient Boosting: ")
print("Accuracy: ",accuracy_score(y_test, pred_gb))
print(classification_report(y_test, pred_gb))

#Random Forest
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(random_state=0)
RF.fit(xv_train, y_train)
pred_rf = RF.predict(xv_test)
print("Random Forest: ")
print("Accuracy: ",accuracy_score(y_test, pred_rf))
print(classification_report(y_test, pred_rf))


#RNN Model
#Tokenize the text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X)
vocab_size = len(tokenizer.word_index) + 1  
print(f'Vocab Size: {vocab_size}')

#find the max length of the sequences
sequences = tokenizer.texts_to_sequences(X)
lengths = [len(seq) for seq in sequences]
max_length = max(lengths) 
print(f'Max Length: {max_length}')

#Average length and 90th percentile length of texts
print(f'Average length: {np.mean(lengths)}')  
print(f'90th percentile length: {np.percentile(lengths, 90)}')  

#Set max length to 400
max_length = 400
#Pad the sequences
padded_x = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

#Split the data into training, validation and testing sets for RNN model
padded_x_train, padded_x_test, y_train, y_test = train_test_split(padded_x, y, test_size=0.33, random_state=42)

padded_x_train, padded_x_val, y_train, y_val = train_test_split(padded_x_train, y_train, test_size=0.2, random_state=42)

#Build the RNN model
from tensorflow.keras.layers import Embedding, LSTM, Dense,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

#learning rate
learning_rate = 0.001

#Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length))  
model.add(LSTM(128, return_sequences=False))  

#Dropout layer to prevent overfitting
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_x_train, y_train, epochs=12, batch_size=32, validation_data=(padded_x_val, y_val),callbacks=[early_stopping])

#Evaluate the model
loss, accuracy = model.evaluate(padded_x_test, y_test)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

from sklearn.metrics import roc_curve, auc

# Predict probabilities
y_pred_prob = model.predict(padded_x_test)

# Compute ROC curve and ROC area for each class
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Compute ROC area
roc_auc = auc(fpr, tpr)

# Optimal threshold
J = tpr - fpr
optimal_idx = np.argmax(J)
optimal_threshold = thresholds[optimal_idx]

print("Optimal Threshold (Youden's J):", optimal_threshold)

# Plot ROC curve for LSTM Model
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='LSTM (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


#Shap Explainer for Logistic Regression
explainer_LR = shap.LinearExplainer(LR,xv_test)
shap_values_LR = explainer_LR(xv_test)
shap.summary_plot(shap_values_LR, xv_test, feature_names=vectorization.get_feature_names_out())
shap.plots.bar(shap_values_LR)
shap.plots.beeswarm(shap_values_LR)

#Shap Explainer for Decision Tree
explainer_DT = shap.Explainer(DT, xv_train_dense)
shap_values_DT = explainer_DT(xv_test_dense)
shap.summary_plot(shap_values_DT, xv_test_dense, feature_names=vectorization.get_feature_names_out())

#Shap Explainer for Gradient Boosting
explainer_GB = shap.Explainer(GB, xv_train_dense)
shap_values_GB = explainer_GB(xv_test_dense)
shap.summary_plot(shap_values_GB, xv_test_dense, feature_names=vectorization.get_feature_names_out())

#Shap Explainer for Random Forest
explainer_RF = shap.Explainer(RF, xv_train_dense)
shap_values_RF = explainer_RF(xv_test_dense)
shap.summary_plot(shap_values_RF, xv_test_dense, feature_names=vectorization.get_feature_names_out())



# ROC Curve for 4 Models
y_pred_LR = LR.predict_proba(xv_test)[:, 1]
fpr_LR, tpr_LR, _ = roc_curve(y_test, y_pred_LR)
auc_LR = auc(fpr_LR, tpr_LR)

y_pred_DT = DT.predict_proba(xv_test)[:, 1]
fpr_DT, tpr_DT, _ = roc_curve(y_test, y_pred_DT)
auc_DT = auc(fpr_DT, tpr_DT)

y_pred_RF = RF.predict_proba(xv_test)[:, 1]
fpr_RF, tpr_RF, _ = roc_curve(y_test, y_pred_RF)
auc_RF = auc(fpr_RF, tpr_RF)

y_pred_GB = GB.predict_proba(xv_test)[:, 1]
fpr_GB, tpr_GB, _ = roc_curve(y_test, y_pred_GB)
auc_GB = auc(fpr_GB, tpr_GB)


# Plot ROC curve for 4 Models
plt.figure(figsize=(10, 7))
plt.plot(fpr_LR, tpr_LR, label=f'Logistic Regression (AUC = {auc_LR:.3f})', color='blue')
plt.plot(fpr_DT, tpr_DT, label=f'Decision Tree (AUC = {auc_DT:.3f})', color='green')
plt.plot(fpr_RF, tpr_RF, label=f'Random Forest (AUC = {auc_RF:.3f})', color='red')
plt.plot(fpr_GB, tpr_GB, label=f'Gradient Boost (AUC = {auc_GB:.3f})', color='orange')


plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

# Add labels
plt.title('ROC Curve for 4 Models')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.grid()
