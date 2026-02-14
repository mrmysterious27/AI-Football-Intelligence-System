import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from statsbombpy import sb
from sklearn.linear_model import LogisticRegression

st.set_page_config(layout="wide")
st.title("⚽ AI Football Intelligence System")

# -------------------------
# Load Competitions
# -------------------------
@st.cache_data
def load_competitions():
    return sb.competitions()

competitions = load_competitions()

# Select competition
competition_name = st.selectbox(
    "Select Competition",
    competitions["competition_name"].unique()
)

selected_comp = competitions[
    competitions["competition_name"] == competition_name
].iloc[0]

matches = sb.matches(
    competition_id=selected_comp["competition_id"],
    season_id=selected_comp["season_id"]
)

match_label = matches["home_team"] + " vs " + matches["away_team"]
selected_match_label = st.selectbox("Select Match", match_label)

match_row = matches[match_label == selected_match_label].iloc[0]
match_id = match_row["match_id"]

# -------------------------
# Load Events
# -------------------------
@st.cache_data
def load_events(match_id):
    return sb.events(match_id=match_id)

events = load_events(match_id)

st.subheader("Match Overview")
st.write(selected_match_label)

# -------------------------
# Passing Network
# -------------------------
st.subheader("Passing Network")

passes = events[
    (events["type"] == "Pass") &
    (events["pass_outcome"].isna())
]

G = nx.DiGraph()

for _, row in passes.iterrows():
    passer = row["player"]
    recipient = row["pass_recipient"]
    
    if pd.notna(recipient):
        if G.has_edge(passer, recipient):
            G[passer][recipient]["weight"] += 1
        else:
            G.add_edge(passer, recipient, weight=1)

plt.figure(figsize=(12,8))
pos = nx.spring_layout(G, k=0.5)
weights = [G[u][v]['weight'] for u,v in G.edges()]
nx.draw(G, pos, with_labels=True, width=weights, node_size=500)
st.pyplot(plt)

# -------------------------
# Player Performance
# -------------------------
st.subheader("Player Performance Scores")

performance = {}

players = events["player"].dropna().unique()

for player in players:
    player_data = events[events["player"] == player]
    
    completed_passes = len(player_data[
        (player_data["type"] == "Pass") &
        (player_data["pass_outcome"].isna())
    ])
    
    goals = len(player_data[
        (player_data["type"] == "Shot") &
        (player_data["shot_outcome"] == "Goal")
    ])
    
    tackles = len(player_data[
        (player_data["type"] == "Duel")
    ])
    
    score = completed_passes*1 + goals*5 + tackles*2
    
    performance[player] = score

perf_df = pd.DataFrame(
    performance.items(),
    columns=["Player","Score"]
).sort_values(by="Score", ascending=False)

st.dataframe(perf_df)

# -------------------------
# xG Model
# -------------------------
st.subheader("Expected Goals (xG)")

shots = events[events["type"] == "Shot"].copy()

if not shots.empty:
    shots = shots.dropna(subset=["location"])
    
    shots["x"] = shots["location"].apply(lambda x: x[0])
    shots["y"] = shots["location"].apply(lambda x: x[1])
    shots["goal"] = shots["shot_outcome"].apply(lambda x: 1 if x == "Goal" else 0)
    
    if len(shots) > 10:
        X = shots[["x","y"]]
        y = shots["goal"]
        
        model = LogisticRegression()
        model.fit(X,y)
        
        shots["xG"] = model.predict_proba(X)[:,1]
        
        st.dataframe(
            shots[["player","xG"]]
            .sort_values(by="xG", ascending=False)
        )
    else:
        st.write("Not enough shots to build xG model.")
else:
    st.write("No shots in this match.")
