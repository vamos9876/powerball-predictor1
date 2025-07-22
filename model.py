from sklearn.ensemble import RandomForestClassifier
import numpy as np

def train_models(df):
    features = df[['White1', 'White2', 'White3', 'White4', 'White5']].values
    labels_white = df[['White1', 'White2', 'White3', 'White4', 'White5']]
    labels_power = df['Powerball']

    white_model = RandomForestClassifier(n_estimators=100, random_state=42)
    power_model = RandomForestClassifier(n_estimators=100, random_state=42)

    white_model.fit(features, labels_white)
    power_model.fit(features, labels_power)

    return white_model, power_model