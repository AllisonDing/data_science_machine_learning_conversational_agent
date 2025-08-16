# ML Run Report
**Data:** data/Titanic-Dataset-1.csv
**Target:** Survived
## EDA
Nulls: {'Unnamed: 0': 0, 'PassengerId': 0, 'Survived': 0, 'Pclass': 0, 'Name': 0, 'Sex': 0, 'Age': 198651, 'SibSp': 0, 'Parch': 0, 'Ticket': 0, 'Fare': 0, 'Cabin': 771051, 'Embarked': 2245}
Class balance: {0: 616170, 1: 383830}
## Candidates
- (cuML) Logistic Regression: acc = 1.0000, f1 = 1.0000
- (cuML) SVC: acc = 1.0000, f1 = 1.0000
- (cuML) Random Forest: acc = 0.9181, f1 = 0.9163

## Champion
**(cuML) Logistic Regression** with f1 = 1.0000, acc = 1.0000