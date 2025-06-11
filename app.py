import streamlit as st
import pandas as pd
import joblib

# Load model and feature order
model = joblib.load("cricket_win_predictor.pkl")
with open("feature_columns.txt") as f:
    feature_order = f.read().splitlines()

# Mappings
match_format_map = {"ODI": 0, "Test": 1, "T20": 2}
match_light_map = {"Day": 0, "Night": 1, "Day and Night": 2}
first_selection_map = {"Batting": 0, "Bowling": 1}
season_map = {"Rainy": 0, "Winter": 1}
opponent_map = {"England": 0, "Australia": 1, "Sri Lanka": 2}

# Suggest strategy to convert Loss into Win
def suggest_strategy(base_input):
    trial = base_input.copy()
    for bowlers in [3, 4, 5]:
        for all_rounders in [3, 4]:
            for age in [26.0, 27.0]:
                for scorer in [75, 85, 95]:
                    for bat_first in [0, 1]:
                        trial.update({
                            'Bowlers_in_team': bowlers,
                            'All_rounder_in_team': all_rounders,
                            'Avg_team_Age': age,
                            'player_highest_run': scorer,
                            'First_selection': bat_first
                        })
                        df_trial = pd.DataFrame([trial])[feature_order]
                        pred = model.predict(df_trial)[0]
                        if pred == 1:
                            return trial
    return None

# Streamlit UI
st.title("🏏 Cricket Match Outcome Predictor with Strategy Suggestion")

# Input from user
col1, col2 = st.columns(2)

with col1:
    match_format = st.selectbox("Match Format", ["ODI", "T20", "Test"])
    opponent = st.selectbox("Opponent", ["England", "Australia", "Sri Lanka"])
    match_light = st.selectbox("Match Light Type", ["Day", "Night", "Day and Night"])
    season = st.selectbox("Season", ["Rainy", "Winter"])
    offshore = st.radio("Match Offshore?", ["Yes", "No"])

with col2:
    avg_age = st.slider("Average Team Age", 20.0, 35.0, 26.0)
    bowlers = st.slider("Number of Bowlers", 1, 5, 3)
    all_rounders = st.slider("Number of All-Rounders", 1, 5, 3)
    first_selection = st.selectbox("India Chooses to", ["Batting", "Bowling"])

# Player performance
player_score = st.slider("Player Highest Run", 30, 150, 75)
player_wickets = st.slider("Player Highest Wicket", 0, 5, 3)

if st.button("Predict Outcome"):
    features = {
        'Avg_team_Age': avg_age,
        'Wicket_keeper_in_team': 1,
        'All_rounder_in_team': all_rounders,
        'Bowlers_in_team': bowlers,
        'First_selection': first_selection_map[first_selection],
        'Audience_number': 50000,
        'Max_run_scored_1over': 14,
        'Max_wicket_taken_1over': 3,
        'Extra_bowls_bowled': 1,
        'Min_run_given_1over': 2,
        'Min_run_scored_1over': 4,
        'Max_run_given_1over': 10,
        'extra_bowls_opponent': 0,
        'player_highest_run': player_score,
        'Players_scored_zero': 2,
        'player_highest_wicket': player_wickets,
        'Match_format': match_format_map[match_format],
        'Opponent': opponent_map[opponent],
        'Match_light_type': match_light_map[match_light],
        'Season': season_map[season],
        'Offshore': 1 if offshore == "Yes" else 0
    }

    input_df = pd.DataFrame([features])[feature_order]
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("🎉 Prediction: India is likely to WIN the match!")
    else:
        st.error("⚠️ Prediction: India is likely to LOSE.")
        st.info("Trying alternate strategies...")

        strategy = suggest_strategy(features)
        if strategy:
            st.success("✅ Strategy Found: India can WIN if you adjust these:")
            st.write({
                "Bowlers_in_team": strategy['Bowlers_in_team'],
                "All_rounder_in_team": strategy['All_rounder_in_team'],
                "Avg_team_Age": strategy['Avg_team_Age'],
                "player_highest_run": strategy['player_highest_run'],
                "First_selection": "Batting" if strategy['First_selection'] == 0 else "Bowling"
            })
        else:
            st.error("❌ No successful strategy found after multiple attempts.")
