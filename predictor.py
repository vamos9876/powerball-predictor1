import numpy as np

def predict_next_draw(model_white, model_power, X_recent):
    """
    Predict the next Powerball draw using trained models.
    """
    white_probs = model_white.predict_proba(X_recent)
    power_probs = model_power.predict_proba(X_recent)

    # Predict 5 white balls using model classes_
    selected_whites = set()
    while len(selected_whites) < 5:
        i = len(selected_whites)
        probs = white_probs[i][0]
        classes = model_white.estimators_[i].classes_
        probs = np.array(probs) / np.sum(probs)
        chosen = np.random.choice(classes, p=probs)
        selected_whites.add(chosen)

    # Predict Powerball
    power_p = power_probs[0]
    power_classes = model_power.classes_
    power_p = np.array(power_p) / np.sum(power_p)
    predicted_power = np.random.choice(power_classes, p=power_p)

    return sorted(selected_whites), int(predicted_power)
