# ML Run Report
**Data:** data/Titanic-Dataset.csv
**Target:** Survived
## EDA
Nulls: {'PassengerId': 0, 'Survived': 0, 'Pclass': 0, 'Name': 0, 'Sex': 0, 'Age': 177, 'SibSp': 0, 'Parch': 0, 'Ticket': 0, 'Fare': 0, 'Cabin': 687, 'Embarked': 2}
Class balance: {0: 549, 1: 342}
## Candidates
- cuML Logistic Regression: acc = 0.8101, f1 = 0.8075
- cuML SVC: acc = 0.8101, f1 = 0.8059
- cuML Random Forest: acc = 0.7933, f1 = 0.7824

## Champion
**cuML Logistic Regression** with f1 = 0.8075, acc = 0.8101