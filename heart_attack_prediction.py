import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sb
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

#reading the csv file
df = pd.read_csv("heart.csv")

#renaming the columns to something the user would understand 
df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure',
              'cholesterol', 'fasting_blood_sugar', 'rest_ecg',
              'max_heart_rate_achieved', 'exercise_induced angina',
              'st_depression', 'st_slope', 'num_major_vessels',
              'thalassemia', 'target']

#create a heatmap with the correlation matrix to find features that are highly
#correlated so we can remove them
#clearly nothing to remove
f, ax = plt.subplots(figsize=(10,10))
sb.heatmap(df.corr(), annot=True, linewidth=0.5, fmt='.1f',ax=ax)
plt.show()

#violin plot is done for continuous variables
#checks for outliers so the model can learn accurately 
#run the following one at a time and comment out the rest as you look at the
#graphs for outliers
sb.violinplot(y ='resting_blood_pressure', data=df)
# sb.violinplot(y ='cholesterol', data=df)
# sb.violinplot(y ='max_heart_rate_achieved', data=df)
#we concluded that no outliers need to be removed

#split the set into test set and training set into 70:30 split
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', 1),
                                                    df['target'], test_size=0.3,
                                                    random_state=42)

model = RandomForestClassifier(max_depth=5)
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

cm = confusion_matrix(y_test, y_predict)
sb.heatmap(cm/np.sum(cm), annot=True, fmt=".2%",cmap='Blues')
plt.show()
ac = accuracy_score(y_test, y_predict)
print("Accuracy is: {}".format(ac*100))

y_pred_quant = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)

fig, ax = plt.subplots()
ax.plot(fpr,tpr)
ax.plot([0,1], [0,1], transform = ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

auc(fpr, tpr)









