from cricket_predictions import main


if __name__ == '__main__':
    top_11_players = main()
    print("Top 11 Players Based on Combined Scores:")
    for player_name, score in top_11_players:
        print(f"{player_name}: {score}")
