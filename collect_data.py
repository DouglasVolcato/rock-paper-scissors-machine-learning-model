from RPS_game import play, mrugesh, abbey, quincy, kris
from utils.data_extractor import DataExtractor
from RPS import player, player_decorator

play(player_decorator(player, DataExtractor('player1')), quincy, 100000, verbose=False)
play(player_decorator(player, DataExtractor('player2')), abbey, 100000, verbose=False)
play(player_decorator(player, DataExtractor('player3')), kris, 100000, verbose=False)
play(player_decorator(player, DataExtractor('player4')), mrugesh, 100000, verbose=False)
