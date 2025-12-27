import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

# read the file

data = pd.read_csv("code and dataset/cancer-risk-factors.csv")

# Check the dataset
print(data.shape)
print(data.head())
print(data.describe())
print(data.dtypes)
print(data.isnull().sum())

data = data.drop('Patient_ID', axis=1)

# Encode categorical columns (Cancer_Type and Risk_Level)
le = LabelEncoder()
data['Risk_Encode'] = le.fit_transform(data['Risk_Level'])
data['Cancer_Encode'] = le.fit_transform(data['Cancer_Type'])

# assign input columns

X = data[["Age", "Smoking", "Alcohol_Use", "Diet_Red_Meat", "Diet_Salted_Processed", "Air_Pollution", "Overall_Risk_Score", "Cancer_Encode"]]

# prepocessing the input if they don't maintain similar units

scale = StandardScaler()
X_scaled = scale.fit_transform(X)

# Model structure

Kmeans = KMeans(n_clusters=3, random_state=42)
cluster = Kmeans.fit_predict(X_scaled)
data['Clusters'] = cluster

# Accuracy Metrics of K Means Clustering

from sklearn import metrics
labels = Kmeans.labels_
silhouette_avg = metrics.silhouette_score(X_scaled,labels)
print("=========== Model Accuracy ===============")
print(f"Silhouette coefficient: {silhouette_avg}")
print("==================================")

# CLUSTER MEANING REFERENCE TABLE

cluster_reference = {
    0: ("Low Risk", 
        "Your lifestyle shows minimal cancer-related risk factors.",
        "Maintain regular check-ups, healthy diet, and physical activity."
        ),

    1: ("Medium Risk", 
        "Some moderate risk patterns are detected.",
        "Improve dietary habits, reduce smoking/alcohol, exercise regularly."
        ),

    2: ("High Risk", 
        "Multiple strong risk indicators found.",
        "Consult a doctor, follow preventive screenings, modify lifestyle."
        )
}

# plotting the model output
plt.figure(figsize=(10,10))
plt.scatter(data['Age'], data['Overall_Risk_Score'], c=cluster, cmap='viridis')
plt.scatter(scale.inverse_transform(Kmeans.cluster_centers_)[:,0],
            scale.inverse_transform(Kmeans.cluster_centers_)[:,1],
            c='red', s=200, marker='X')
plt.title("K Means Clustering Output")
plt.xlabel("Age")
plt.ylabel("Overall Risk Score")
plt.grid(True)
plt.show()

# SIMPLE USER INPUT VALIDATION

age = int(input("Enter Age (years): "))
smoke = int(input("Enter Smoking Score (0–10): "))
alcohol = int(input("Enter Alcohol Use Score (0–10): "))
dietred = int(input("Enter diet meat Score (0–10): "))
dietsalt = int(input("Enter diet salt Score (0–10): "))
airp = int(input("Enter Air pollution level: "))
ORF = float(input("Enter Overall risk Score (0.0–1.0): "))

# Valid cancer types
valid_cancers = ["Breast", "Lung", "Colon", "Prostate", "Skin"]

print("\nAvailable Cancer Types:", valid_cancers)
cancer_input = input("Enter Cancer Type: ").strip().capitalize()

# INPUT VALIDATION

if age <= 0:
    print("Please enter a valid Age value.")
elif smoke < 0 or smoke > 10:
    print("Smoking score must be between 0 and 10.")
elif alcohol < 0 or alcohol > 10:
    print("Alcohol score must be between 0 and 10.")
elif dietred < 0 or dietred > 10:
    print("Obesity score must be between 0 and 10.")
elif dietsalt < 0 or dietsalt > 10:
    print("Air pollution score must be between 0 and 10.")
elif airp < 0 or airp > 10:
    print("Occupational hazard must be 0–10.")
elif ORF < 0 or ORF > 1:
    print("Calcium intake must be 0–10.")
elif cancer_input not in valid_cancers:
    print("Invalid Cancer Type! Please choose from:", valid_cancers)

else:
    print("\nAll inputs are valid! Proceeding with prediction...\n")
    
cancer_encoded_input = le.transform([cancer_input])[0]

# NEW DATA PREDICTION

#new_data = pd.DataFrame([[age, gender,smoke, alcohol, activity, diet, risk]],
#                        columns=["Age","Gender","Smoking","Alcohol_Use",
#                                 "Physical_Activity","Diet_Red_Meat",
#                                 "Overall_Risk_Score"])

new_data = pd.DataFrame([[
    age, smoke, alcohol, dietred, dietsalt, airp, ORF,
    cancer_encoded_input
]], columns=[
    "Age", "Smoking", "Alcohol_Use", "Obesity", "Air_Pollution", "Occupational_Hazards", "Overall_Risk_Score", "Cancer_Encode"
])

new_scaled = scale.transform(new_data)

pred_cluster = Kmeans.predict(new_scaled)[0]

risk_label, meaning, suggestion = cluster_reference[pred_cluster]

decoded_cancer_type = cancer_input

print("\n========== PATIENT RESULT ==========")
print(f"Cancer Type Entered : {decoded_cancer_type}")
print(f"Assigned Cluster    : {pred_cluster}")
print(f"Risk Category       : {risk_label}")
print(f"Interpretation      : {meaning}")
print(f"Suggestion          : {suggestion}")
print("====================================\n")

# USER INTERACTIVE FEATURES
print("Choose an option:\n"
      "1. Visualize risk score vs age comparison\n"
      "2. Show cluster centers\n"
      "3. Exit")

choice = input("Enter choice (1/2/3): ")

if choice == "1":
    plt.figure(figsize=(9,7))
    plt.scatter(data["Age"], data["Overall_Risk_Score"], 
                c=cluster, cmap="cividis", alpha=0.7)

    plt.scatter(age, "Overall_Risk_Score", c="blue", s=200, marker="D",
                label="Your Input")

    plt.xlabel("Age")
    plt.ylabel("Overall Risk Score")
    plt.title("Risk Distribution Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()
    
elif choice == "2":
    print("\nCluster Centers (Original Scale):\n")
    centers = scale.inverse_transform(Kmeans.cluster_centers_)
    for i, c in enumerate(centers):
        print(f"Cluster {i}: {c}")
else:
    print("Goodbye! Stay healthy. ✨")


#sample input: 45, 1, 4, 3, 6, 1, 5, 4, 6, 5, 3, 2, 0, 1, 7, 27.5, lung
