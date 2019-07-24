"""
Once a model is learned, use this to play it.
"""

from game import main
import pygame
import numpy as np
from nn import neural_net

NUM_SENSORS = 9


def play(model):

    car_distance = 0
    game_state = main.GameState()

    # Do nothing to get initial.
    _, state = game_state.frame_step((2))
    exit = False
    # Move.
    while not exit:
        car_distance += 1

        # Choose action.
        action = (np.argmax(model.predict(state, batch_size=1)))

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
    saved_model = 'saved-models/32-32-40-50000-100000.h5'
    model = neural_net(NUM_SENSORS, [32, 32], saved_model)
    play(model)