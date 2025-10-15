import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Loading DataSet
ipl=pd.read_csv("ipl_data.csv")
print(ipl.head())

#Exploratory Data Analysis
data=ipl.copy()
matches_per_venue=data[['mid','venue']].drop_duplicates()
matches_count=matches_per_venue['venue'].value_counts()
print(matches_count)
plt.figure(figsize=(12,6))
sns.barplot(x=matches_count.values,y=matches_count.index,hue=matches_count.index,palette='rainbow')
plt.xlabel("Number of Matches")
plt.ylabel("Venue")
plt.title("Number of Matches per Venue")
plt.tight_layout()
plt.show()

runs_by_batsman=data.groupby('batsman')['runs'].max().sort_values(ascending=False).head(10)
print(f"Highest runs scorerd by batsmen\n{runs_by_batsman}")
plt.figure(figsize=(12,6))
sns.barplot(x=runs_by_batsman.values, y=runs_by_batsman.index,hue=runs_by_batsman.index,palette='rainbow')
plt.xlabel("Total runs")
plt.ylabel("Batsman")
plt.title("Top 10 Batsmen by total runs")
plt.show()

wickets_by_bowler=data.groupby('bowler')['wickets'].max().sort_values(ascending=False).head(10)
print(f'Highest wickets taken by bowlers\n{wickets_by_bowler}')
plt.figure(figsize=(12,6))
sns.barplot(x=wickets_by_bowler.values,y=wickets_by_bowler.index,hue=wickets_by_bowler.index,palette='rainbow')
plt.xlabel("Total Wickets")
plt.ylabel("Bowler")
plt.show()

#LabelEncoding
from sklearn.preprocessing import LabelEncoder
ipl_cols=['venue','bat_team','bowl_team','batsman','bowler']
label_encoders={}
data_encoded=data.copy()
for col in ipl_cols:
    le=LabelEncoder()
    data_encoded[col]=le.fit_transform(data_encoded[col])
    label_encoders[col]=le

#Feature Selection
data_corr=data_encoded.drop(columns=['mid','date'],axis=1)
sns.heatmap(data_corr.corr(),annot=True)
plt.show()

#Train and Test split
from sklearn.model_selection import train_test_split
feature_cols=['venue','bat_team','bowl_team','batsman','bowler','runs','wickets','overs','non-striker']
x=data_encoded[feature_cols]
y=data_encoded['total']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

#Building Neural Network
import tensorflow as tf
from tensorflow import keras
model=keras.Sequential([
    keras.layers.Input(shape=(x_train.shape[1],)),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(216,activation='relu'),
    keras.layers.Dense(1,activation='linear')
])
huber_loss=tf.keras.losses.Huber(delta=1.0)
model.compile(optimizer='adam',loss=huber_loss)

#Model Training
model.fit(x_train,y_train,epochs=10,batch_size=64,validation_data=(x_test,y_test))
model_losses=pd.DataFrame(model.history.history)
print(model_losses)
model_losses.plot()
plt.show()

#Model Evaluation
from sklearn.metrics import mean_absolute_error,mean_squared_error
predictions=model.predict(x_test)
print(f"Mean Absolute Error:{mean_absolute_error(y_test,predictions)}")
print(f"Mean Squared Error:{mean_squared_error(y_test,predictions)}")

#Saving Models,Scalers,Encoders
from tensorflow.keras.models import load_model


# Save model correctly
model.save("model.keras")

# Save encoders and scaler separately with joblib
import joblib
joblib.dump(label_encoders, "label_encoders.joblib")
joblib.dump(scaler, "scaler.joblib")
