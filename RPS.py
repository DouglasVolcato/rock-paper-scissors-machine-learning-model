# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

from utils.classification_model import ClassificationModel
from utils.data_extractor import DataExtractor
import random

model = ClassificationModel()

def player(prev_play, opponent_history=[]):
    opponent_history.append(prev_play)

    guess = random.choice(['R', 'P', 'S'])
    if len(opponent_history) > 10:
        guess = model.predict(opponent_history[-10:])

    return guess

def player_decorator(func: callable, data_extractor: DataExtractor):
    def inner(prev_play: str, opponent_history=[]):
        data_extractor.saveData(prev_play)
        return func(prev_play, opponent_history)
    return inner
