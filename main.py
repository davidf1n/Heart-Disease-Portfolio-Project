import pandas as pd  # to load and manipulate data
import matplotlib.pyplot as plt  # to draw graphs
from sklearn.tree import DecisionTreeClassifier  # to build a classification tree
from sklearn.tree import plot_tree  # to draw a classification tree
from sklearn.model_selection import train_test_split  # to split data into train test
from sklearn.metrics import confusion_matrix  # to create a confusion matrix
from sklearn.metrics import plot_confusion_matrix  # to draw the confusion matrix

# load the data and create a df
df = pd.read_csv("processed.cleveland.data", header=None)
print(f'{df.head()} \n')

# according to the information from the source of the data, we'll replace the column names
df.columns = ['age', 'sex', 'cp', 'restbp', 'chol', 'fbs', 'restecg', 'thelach', 'exang', 'oldpeak',
              'slope', 'ca', 'thal', 'hd']
print(f'{df.head()} \n')

# in order to look for missing data types - let's see the types of variables stored in the colunms
print(f'{df.dtypes} \n')

# looks like 'ca' and 'thal' are from type object. That might indicate that a mixture of values is present.
# let's look for unique values - in the same way that SELECT DISTINCT will work in SQL
print(f'{df["ca"].unique()} \n')
print(f'{df["thal"].unique()} \n')

# we need to handle the '?' values.
# let's understand how many rows contains '?' in both column
question_mark_rows_ca = df.loc[(df["ca"] == "?")]
question_mark_rows_thal = df.loc[(df["thal"] == "?")]
question_mark_rows_df = pd.concat([question_mark_rows_thal, question_mark_rows_ca]).drop_duplicates() \
    .reset_index(drop=True)  # union without repetition
print(f'{len(question_mark_rows_df)} \n')

# what percentage of the data is missing?
print(f'There are {round(len(question_mark_rows_df) / len(df) * 100)}% of the data with question marks\n')

# we will build the classification tree using the 98% percent of the data and remove the
# question mark rows. We will have 297 rows left, that's enough to build a well performing tree.
df_clean_first_step = df.loc[(df['ca'] != '?')]
df_clean = df_clean_first_step.loc[(df['thal'] != '?')]
print(f'(There are only {len(df_clean)} rows now \n')

# Let's define a dataframe of the data we will build the tree from
# let's define a dataframe of the column we want to predict
df_parameters = df_clean.iloc[:, 0:13]
df_labels = df_clean.iloc[:, 13:]

# now, we shall convert the python datatypes to categorical where it is needed (f.e. cp)
# we will use pandas get_dummies to split the columns into a binary columns representing
# 1 for the right category and zeros to the false categories (One-Hot Encoding)
# sex, fbs and exang are already binary, therefore, we should not do anything with them
print(f'cp values are: {df_parameters["cp"].unique()}')
print(f'restecg values are: {df_parameters["restecg"].unique()}')
print(f'slope values are: {df_parameters["slope"].unique()}')
print(f'thal values are: {df_parameters["thal"].unique()}')

# after we verified that they indeed categorical in the data, we'll convert them:
df_parameters_encoded = pd.get_dummies(df_parameters, columns=['cp', 'restecg', 'slope', 'thal'])
print(df_parameters_encoded.head())

# we are creating a tree that will classify if there is a disease or not, therefore, we will convert
# 1-4 values at the 'hd' column to 1 and leave the zero as a zero.
# thus, we create a binary classification for heart disease
not_zero_labels = df_labels > 0
df_labels_encoded = pd.DataFrame(df_labels, index=not_zero_labels['hd'].tolist())
df_labels_encoded.loc[True] = 1
df_labels_encoded.loc[False] = 0
print(df_labels_encoded['hd'].unique())

# let's move on to the classification tree and create a train-test splits
df_parameters_train, df_parameters_test, df_labels_train, df_labels_test = train_test_split(df_parameters_encoded,
                                                                                            df_labels_encoded,
                                                                                            random_state=42)
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt = clf_dt.fit(df_parameters_train, df_labels_train)
plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt, filled=True, rounded=True, class_names=["No HD", "Yes HD"],
          feature_names=df_parameters_encoded.columns)
plt.show()
plot_confusion_matrix(clf_dt, df_parameters_test, df_labels_test,
                      display_labels=["No HD", "Yes HD"])
plt.show()

# The decision tree is huge, and the results could be optimized, probably due to overfit
# let's show the results
print(f'Recall is: {round(26 / 33 * 100)}%, Precision is {round(26 / 37 * 100)}%, '
      f'definitely could be optimized. The tree might be overfitting.')
