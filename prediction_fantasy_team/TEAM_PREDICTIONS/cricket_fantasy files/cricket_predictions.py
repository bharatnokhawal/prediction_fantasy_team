import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")

# Load the CSV file into a pandas DataFrame
def load_data(transformed_file):
    data = pd.read_csv(transformed_file)
    return data

# Preprocess the data by encoding categorical features and ensuring correct data types
def preprocess_data(data):
    data['ball_faced'] = data['ball_faced'].astype(int)
    data['run_scored'] = data['run_scored'].astype(int)
    data['ball_delivered'] = data['ball_delivered'].astype(int)
    data['run_given'] = data['run_given'].astype(int)
    data['wicket'] = data['wicket'].astype(int)

    # One-hot encode the 'against_team' categorical feature
    encoder = OneHotEncoder(drop='first')
    against_team_encoded = encoder.fit_transform(data[['against_team']])
    against_team_encoded_df = pd.DataFrame(against_team_encoded.toarray(), columns=encoder.get_feature_names_out(['against_team']))

    data.reset_index(drop=True, inplace=True)
    against_team_encoded_df.reset_index(drop=True, inplace=True)

    # Combine original data with encoded categorical features
    data = pd.concat([data, against_team_encoded_df], axis=1)
    return data, encoder, against_team_encoded_df

# Train the models for predicting runs and wickets
def train_models(data, against_team_encoded_df):
    global features_runs, features_wickets

    # Define features and targets for runs prediction
    features_runs = data[['ball_faced'] + list(against_team_encoded_df.columns)]
    target_runs = data['run_scored']

    # Define features and targets for wickets prediction
    features_wickets = data[['ball_delivered', 'run_given'] + list(against_team_encoded_df.columns)]
    target_wickets = data['wicket']

    # Apply KMeans clustering for runs prediction
    kmeans_runs = KMeans(n_clusters=5, random_state=42)
    data['cluster_runs'] = kmeans_runs.fit_predict(features_runs)

    # Apply KMeans clustering for wickets prediction
    kmeans_wickets = KMeans(n_clusters=5, random_state=42)
    data['cluster_wickets'] = kmeans_wickets.fit_predict(features_wickets)

    # Train RandomForest models for each cluster in runs prediction
    models_runs = {}
    for cluster_id in range(kmeans_runs.n_clusters):
        cluster_data = data[data['cluster_runs'] == cluster_id]
        cluster_features = cluster_data[['ball_faced'] + list(against_team_encoded_df.columns)]
        cluster_target = cluster_data['run_scored']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(cluster_features, cluster_target)
        models_runs[cluster_id] = model

    # Train RandomForest models for each cluster in wickets prediction
    models_wickets = {}
    for cluster_id in range(kmeans_wickets.n_clusters):
        cluster_data = data[data['cluster_wickets'] == cluster_id]
        cluster_features = cluster_data[['ball_delivered', 'run_given'] + list(against_team_encoded_df.columns)]
        cluster_target = cluster_data['wicket']
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(cluster_features, cluster_target)
        models_wickets[cluster_id] = model

    return models_runs, models_wickets, kmeans_runs, kmeans_wickets

# Predict runs and wickets for given players against a specific team
def predict_runs_and_wickets(player_names, against_team, models_runs, models_wickets, data, encoder, kmeans_runs, kmeans_wickets, features_runs, features_wickets):
    predictions = {}
    for player_name in player_names:
        player_data = data[data['player'] == player_name]
        if not player_data.empty:
            against_team_encoded = encoder.transform([[against_team]])
            against_team_encoded_df = pd.DataFrame(against_team_encoded.toarray(), columns=encoder.get_feature_names_out(['against_team']))
            
            avg_balls_faced = player_data['ball_faced'].mean()
            avg_ball_delivered = player_data['ball_delivered'].mean()
            avg_run_given = player_data['run_given'].mean()
            
            input_data_runs = pd.DataFrame([[avg_balls_faced] + list(against_team_encoded_df.iloc[0])], columns=features_runs.columns)
            input_data_wickets = pd.DataFrame([[avg_ball_delivered, avg_run_given] + list(against_team_encoded_df.iloc[0])], columns=features_wickets.columns)
            
            cluster_id_runs = kmeans_runs.predict(input_data_runs)[0]
            cluster_id_wickets = kmeans_wickets.predict(input_data_wickets)[0]
            
            predicted_runs = models_runs[cluster_id_runs].predict(input_data_runs)[0]
            predicted_wickets = models_wickets[cluster_id_wickets].predict(input_data_wickets)[0]
            
            predictions[player_name] = {
                'predicted_runs': predicted_runs,
                'predicted_wickets': predicted_wickets
            }
        else:
            predictions[player_name] = {
                'predicted_runs': None,
                'predicted_wickets': None
            }
    return predictions

# Calculate impact scores for players based on their predicted runs and wickets
def calculate_impact_score(predictions):
    impact_scores = {}
    for player_name, prediction in predictions.items():
        if prediction['predicted_runs'] is not None and prediction['predicted_wickets'] is not None:
            impact_score = (prediction['predicted_runs'] * 1.4) + (prediction['predicted_wickets'] * 25)
            impact_scores[player_name] = impact_score
        else:
            impact_scores[player_name] = None
    return impact_scores

# Calculate the player's historical performance against a specific team
def performance_against_team(player_name, team_name, df):
    player_team_data = df[(df['player'] == player_name) & (df['against_team'] == team_name)]
    matches_played = player_team_data['match_id'].nunique()

    total_runs = player_team_data['run_scored'].sum()
    total_wickets = player_team_data['wicket'].sum()
    total_4s = player_team_data['4s'].sum()
    total_6s = player_team_data['6s'].sum()
    total_50s = len(player_team_data[player_team_data['run_scored'] >= 50])
    total_100s = len(player_team_data[player_team_data['run_scored'] >= 100])

    return {
        'matches_played': matches_played,
        'total_runs': total_runs,
        'total_wickets': total_wickets,
        'total_4s': total_4s,
        'total_6s': total_6s,
        'total_50s': total_50s,
        'total_100s': total_100s
    }

# Calculate fantasy points for a player based on their performance stats
def calculate_fantasy_points(player_stats):
    points = 0

    points += player_stats['total_runs'] * 1.4
    points += player_stats['total_4s'] * 1
    points += player_stats['total_6s'] * 2
    points += player_stats['total_wickets'] * 25
    points += player_stats['total_50s'] * 8
    points += player_stats['total_100s'] * 16

    return points

# Main function to execute the entire process
def main():
    transformed_file = 'transformed_match_data.csv'
    data = load_data(transformed_file)
    data, encoder, against_team_encoded_df = preprocess_data(data)
    models_runs, models_wickets, kmeans_runs, kmeans_wickets = train_models(data, against_team_encoded_df)

    teams = [['Abdul Samad','Abhishek Sharma','RA Tripathi','H Klaasen','TM Head','B Kumar', 'T Natarajan','JD Unadkatt','Shahbaz Ahmed','PJ Cummins','Nithish Kumar Reddy'], ['SV Samson', 'TK Cadmore', 'YBK Jaiswal', 'TA Boult', 'R Parag', 'SO Hetmyer', 'R Ashwin', 'Dhruv Jurel', 'Avesh Khan', 'R Powell', 'Sandeep Sharma']]
    against_teams = ['Rajasthan Royals', 'Sunrisers Hyderabad']

    all_impact_scores = {}
    for i, team in enumerate(teams):
        against_team = against_teams[i]
        predictions = predict_runs_and_wickets(team, against_team, models_runs, models_wickets, data, encoder, kmeans_runs, kmeans_wickets, features_runs, features_wickets)
        impact_scores = calculate_impact_score(predictions)
        all_impact_scores.update(impact_scores)



    df = pd.read_csv('transformed_match_data.csv')
    fantasy_points = {}
    for team in teams:
        for player_name in team:
            player_stats = performance_against_team(player_name, against_team, df)
            fantasy_points[player_name] = calculate_fantasy_points(player_stats)

    combined_scores = {}
    for player_name in fantasy_points.keys():
        impact_score = all_impact_scores.get(player_name, 0)
        if impact_score is None:
            impact_score = 0
        combined_score = fantasy_points[player_name] + impact_score
        combined_scores[player_name] = combined_score

    ranked_players = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    top_11_players = ranked_players[:11]
    return top_11_players

