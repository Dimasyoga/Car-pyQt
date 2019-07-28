"""
Once a model is learned, use this to play it.
"""

from game import main_v2
import pygame
import numpy as np
from nn_v2 import neural_net

NUM_SENSORS = 14
NUM_ACTION = 4


def play(model):

    car_distance = 0
    game_state = main_v2.GameState()

    # Do nothing to get initial.
    _, state = game_state.frame_step((2))
    exit = False
    # Move.
    while not exit:
        car_distance += 1

        # Choose action.
        action = (np.argmax(model.predict(state, batch_size=1)))
        print(action)

        # Take action.
        _, state = game_state.frame_step(action)

        # Tell us something.
        if car_distance % 1000 == 0:
            print("Current distance: %d frames." % car_distance)

        # Event queue
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit = True


if __name__ == "__main__":
    saved_model = 'saved-models/128-128-64-64-50000-50000.h5'
    model = neural_net(NUM_ACTION, NUM_SENSORS, [128, 128, 64], saved_model)
    play(model)