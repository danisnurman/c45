import streamlit
import pandas as pd

# Train Model
from C45 import C45Classifier

data = pd.read_csv('https://raw.githubusercontent.com/danisnurman/psbnd2/main/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
X = data.drop(['target'], axis=1)
y = data['target']

model = C45Classifier()
model.fit(X, y)

# Predict
data_test = pd.read_csv('https://raw.githubusercontent.com/danisnurman/psbnd2/main/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
model.predict(data_test)

# Evaluate
data_test = pd.read_csv('data_test.csv')
X_test = data_test.drop(['target'], axis=1)
y_test = data_test['target']
streamlit.write(model.evaluate(X_test, y_test))

# Summary Model
streamlit.write(model.summary())

streamlit.write("Testing")