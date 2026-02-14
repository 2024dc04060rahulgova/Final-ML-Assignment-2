a) Problem Statement
The objective of this project is to build and compare multiple classification models to predict whether a patient has heart disease based on clinical attributes.
This project demonstrates an end-to-end ML workflow:
•	Data preprocessing
•	Model training
•	Evaluation
•	Deployment using Streamlit
b) Dataset Description
•	Instances: 1888
•	Features: 13 input features
•	Target: Binary classification (0 = No disease, 1 = Disease)
Features include:
•	age
•	sex
•	cp (chest pain type)
•	trestbps
•	chol
•	fbs
•	restecg
•	thalachh
•	exang
•	oldpeak
•	slope
•	ca
•	thal

c) Models Used
1.	Logistic Regression
2.	Decision Tree
3.	K-Nearest Neighbors
4.	Naive Bayes
5.	Random Forest (Ensemble)



Model  	Accuracy	AUC	Precision	Recall	F1    	MCC
Logistic Regression	0.72     0.83	0.70	0.81	0.75	0.45
Decision Tree       	0.97	0.97	0.97	0.97	0.97	0.95
KNN  			0.80	.78	79	83	81	60
Naive Bayes         	0.70	0.78	0.79	0.83	0.81	0.60
Random Forest       	0.97	0.99	0.97	0.98	0.97	0.95
XGBoost             	0.98	0.99	0.98	0.98	0.98	0.96


OBSERVATIONS
Model 	Observation
	
1.	Logistic Regression	Good baseline model but limited by linear decision boundary
2.	Decision Tree		Very high accuracy but risk of overfitting
3. 	KNN			Balanced performance but sensitive to scaling
3. 	Naive Bayes		Fast and simple but independence assumption reduces performance
5. 	Random Forest		Excellent overall performance and generalisation


