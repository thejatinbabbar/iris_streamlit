import streamlit as st
# create title
st.title('Machine Learning with Iris Dataset')

# import data
from sklearn.datasets import load_iris
iris_data = load_iris()
st.header('About the dataset')
st.write(iris_data.DESCR)

# create dataframe
import pandas as pd
df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
df['class'] = [iris_data.target_names[i] for i in iris_data.target]
st.header('Feature Columns and Label')
st.write(df)

# visualizing the dataset
import seaborn as sns
import matplotlib.pyplot as plt
st.header('Visualizing the Dataset')
# fig = sns.pairplot(df, hue='class')
# st.pyplot(fig)

# split into train and test sets
from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(iris_data.data, iris_data.target, test_size=0.33, random_state=100, shuffle=True)
st.header('Splitting the Dataset into Train and Test sets')
st.subheader(f'Total Dataset: {len(df)} samples')
st.subheader(f'Training set size: {train_features.shape[0]} samples (Green points)')
st.subheader(f'Testing set size: {test_features.shape[0]} samples (Blue points)')

# visualizing train and test sets
fig, ax = plt.subplots(ncols=2, figsize=(10,5))
sns.scatterplot(x=train_features[:, 0], y=train_features[:, 1], color='green', ax=ax[0], label='Training set')
sns.scatterplot(x=test_features[:, 0], y=test_features[:, 1], color='blue', ax=ax[0], label='Testing set')
ax[0].set_xlabel('Sepal Length (cm)')
ax[0].set_ylabel('Sepal Width (cm)')
ax[0].set_title('Sepal Length and Width')
ax[0].legend()
sns.scatterplot(x=train_features[:, 2], y=train_features[:, 3], color='green', ax=ax[1], label='Training set')
sns.scatterplot(x=test_features[:, 2], y=test_features[:, 3], color='blue', ax=ax[1], label='Testing set')
ax[1].set_xlabel('Petal Length (cm)')
ax[1].set_ylabel('Petal Width (cm)')
ax[1].set_title('Petal Length and Width')
ax[1].legend()
st.pyplot(fig)
plt.clf()

# initialize model
from sklearn.linear_model import LogisticRegression
st.header('Using a Logisitic Regression Model')
model = LogisticRegression()
model.fit(train_features, train_labels)

# evaluate the model
st.subheader(f'Model Training Accuracy: {model.score(train_features, train_labels)}')
score = model.score(test_features, test_labels)
st.subheader(f'Model Testing Accuracy: {score}')

# visualizing model performance
from sklearn.model_selection import learning_curve
train_sizes, train_scores, valid_scores = learning_curve(model, train_features, train_labels, cv=10)
train_scores_mean = train_scores.mean(axis=1)
valid_scores_mean = valid_scores.mean(axis=1)

fig, ax = plt.subplots(figsize=(10, 5))
ax = sns.lineplot(train_sizes, train_scores_mean, label='Train score')
ax = sns.lineplot(train_sizes, valid_scores_mean, label='Valid score')
ax.set_title('Learning Curve')
ax.legend()
st.pyplot(fig)
