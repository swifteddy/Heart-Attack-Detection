# Heart Disease Classification using Random Forest Classifier

In this project, we will be using the Random Forest Classifier to predict the presence of heart disease in a patient based on certain features.
 The dataset used is publicly available on Kaggle and contains various features related to heart disease.
 
## Data Preparation and Preprocessing
After reading the csv file, we renamed the columns to make it easier to understand. We then created a heatmap with the correlation matrix to find features that are 
highly correlated so we can remove them. However, we found that no features need to be removed. We then used violin plots to identify any outliers that might affect 
the model's performance, but concluded that no outliers need to be removed. Finally, we split the dataset into a training set and a test set with a 70:30 split.

## Model Building and Evaluation
We used the Random Forest Classifier to build the model with a maximum depth of 5. We then used the model to make predictions on the test set and evaluated its 
performance using the confusion matrix and accuracy score. We achieved an accuracy of 83.5%, which means the model correctly predicted 83.5% of the cases. We then 
plotted the ROC curve and calculated the AUC score, which measures how well the model can distinguish between positive and negative classes.

## Conclusion
Overall, the Random Forest Classifier performed well in predicting the presence of heart disease in patients. However, more feature engineering and data preprocessing 
could be done to improve the model's accuracy. Nonetheless, this project serves as a good starting point for anyone interested in using machine learning to predict heart
disease.

The Python code for this project is available in this repository.
