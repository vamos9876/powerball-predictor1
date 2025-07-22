import pandas as pd

def engineer_features(df):
    """
    Extracts features (X), white ball targets (y_white), and Powerball targets (y_power)
    from the raw draw DataFrame.
    """
    features = []
    white_targets = []
    power_targets = []

    for i in range(len(df) - 1):
        current_draw = df.iloc[i]["white"]
        next_draw = df.iloc[i + 1]

        features.append(current_draw)
        white_targets.append(next_draw["white"])
        power_targets.append(next_draw["powerball"])

    X = pd.DataFrame(features)
    y_white = pd.DataFrame(white_targets)
    y_power = pd.Series(power_targets)

    return X, y_white, y_power
