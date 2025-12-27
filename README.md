# CANCER-RISK-STRATIFICATION
Lifestyle correlated cancer risk factor prediction supported by unsupervised machine learning model using K-Means Clustering algorithm

## About Dataset
Name: Cancer Risk Factor (US Region). Columns Used: 'Age', 'Gender', 'Smoking', 'Alcohol_Use', 'Obesity', 'Family_History', 'Diet_Red_Meat', 'Diet_Salted_Processed', 'Fruit_Veg_Intake', 'Physical_Activity', 'Air_Pollution', 'Occupational_Hazards', 'BRCA_Mutation', 'H_Pylori_Infection', 'Calcium_Intake', 'BMI'. No of Rows: 2000 records

## Model Selection and it's reason:
Model: K-MEANS CLUSTERING (Unsupervised Model)
Description:
•	Without any guidance or supervision, the model learns the data and recognizes its own pattern to predict the proposed new data
•	It is efficient for grouping data into selected cluster

## Training method of data:
Since the model is an unsupervised machine learning K means clustering algorithm, the dataset X independent variables are utilized thoroughly for training.
•	Total Dataset used = 2000 records
•	Scaler used: StandardScaler. Used to normalize multiple features
•	3 clusters are created by mentioning n_clusters = 3
•	random state 42 is implemented to help model learn different possibilities

## Metrics used for Evaluation:
Metrics used:
	Silhouette Score: Utilized while developing a machine learning algorithm (specifically for K-Means Clustering), to measure the performance of clusters formed.
	It forms the clusters by using the formula:

$$s =\frac{b-a}{\max⁡(a,b)}$$

**Where:** 
- **\(a\)**=average distance from the concern point to all other points in the same cluster.
- **\(b\)**=average distance from the concern point to all points with nearest neighboring cluster.

**Interpretation**
- \(s \approx 0.60 to 1\) - Well - clustered data point
- \(s \approx 0.21 to 0.59\) - weak clustering
- \(s \approx 0 to 0.20\) - wrong clustering

It should be observed that the silhouette score should be ranged between 0 and 1. Scores between 0 to 0.20 are considered to be wrong clustering, between 0.21 to 0.59 are weak clustering, and between 0.60 to 1 ensures the model predicts the correct cluster.
Evaluated Value from the model: 
	Silhouette score = 0.165

## Strengths and Weakness of the model:
Strength: Finding the correlation between each feature, a K-Means Clustering model is trained without any guidance such that it tends to form a set of clusters
Weakness: As clusters increase, the model tends to confuse to which segment the latest data has to sit.

## Possible improvements / real-world applications:
Improvement: Model could be trained with precise values of features and increase the records of the data, that would drive the model for clear prediction.

real-world application: Could be supportive as a non contact follow-up statistical analysis of each patient. Spreading awareness to ensure proper work-life balance.

## Conclusion of the project:
The K-Means clustering approach predicted the unknown data to the approximate cancer risk level. Yet, the model has a poor accuracy value (0.165), and the severity levels are weakly grouped. The supplied features do not create any clearly defined groups. As a result, the model must be improved using both optimisation and boosting strategies so that it can strongly cluster to the necessary cancer risk factors.
