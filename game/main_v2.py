import random
import math
import numpy as np
import sys

import pygame
from pygame.locals import *
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import *

# PyGame init
width = 1000
height = 700
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
pygame.event.get()

# Turn off alpha since we don't use it.
screen.set_alpha(None)

# Showing sensors and redrawing slows things down.
show_sensors = True
draw_screen = True


class GameState:
    def __init__(self):

        # Global-ish.
        self.crashed = False

        # Physics stuff.
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)
        self.space.damping = 0.8

        # Create the car.
        self.create_car(100, 100, 0.5)

        # Record steps.
        self.num_steps = 0

        # Create walls.
        static = [
            pymunk.Segment(
                self.space.static_body,
                (0, 1), (0, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, height), (width, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (width-1, height), (width-1, 1), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, 1), (width, 1), 1)
        ]
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS['red']
        self.space.add(static)

        # Create some obstacles, semi-randomly.
        # We'll create three and they'll move around to prevent over-fitting.
        self.obstacles = []
        self.obstacles.append(self.create_obstacle(200, 350, 100))
        self.obstacles.append(self.create_obstacle(700, 200, 125))
        self.obstacles.append(self.create_obstacle(600, 600, 35))

        # Create a cat.
        self.cat = []
        self.cat.append(self.create_cat(300, 400))
        self.cat.append(self.create_cat(400, 300))
        self.cat.append(self.create_cat(150, 150))

    def create_obstacle(self, x, y, r):
        c_body = pymunk.Body(10, 10, pymunk.Body.STATIC)
        c_shape = pymunk.Circle(c_body, r)
        c_shape.elasticity = 1.0
        c_body.position = x, y
        c_shape.color = THECOLORS["blue"]
        self.space.add(c_body, c_shape)
        return c_body

    def create_cat(self, x, y):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        cat_body = pymunk.Body(30, inertia)
        cat_body.position = x, y
        cat_shape = pymunk.Circle(cat_body, 30)
        cat_shape.color = THECOLORS["orange"]
        cat_shape.elasticity = 1.0
        cat_shape.angle = 0.5
        direction = Vec2d(1, 0).rotated(cat_body.angle)
        self.space.add(cat_body, cat_shape)
        return cat_body

    def create_car(self, x, y, r):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.car_body = pymunk.Body(50.0, inertia)
        self.car_body.position = x, y
        self.car_shape = pymunk.Circle(self.car_body, 25)
        self.car_shape.color = THECOLORS["green"]
        self.car_shape.elasticity = 1.0
        self.car_body.angle = r
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.apply_impulse_at_local_point(driving_direction)
        self.space.add(self.car_body, self.car_shape)

    def frame_step(self, action):
        if action == 0:  # Turn left.
            self.car_body.angle -= .2
        elif action == 1:  # Turn right.
            self.car_body.angle += .2
        elif action == 2:
            impulse = 200 * Vec2d(1,0)
            impulse.rotate(self.car_body.angle)
            self.car_body.apply_impulse_at_world_point(impulse, self.car_body.position)

        # Move obstacles.
        if self.num_steps % 100 == 0:
            self.move_obstacles()

        # Move cat.
        if self.num_steps % 5 == 0:
            self.move_cat()

        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        #self.car_body.velocity = 100 * driving_direction

        # Update the screen and stuff.
        screen.fill(THECOLORS["black"])
        #draw(screen, self.space)

        options = pymunk.pygame_util.DrawOptions(screen)
        self.space.debug_draw(options)

        self.space.step(1./10)
        if draw_screen:
            pygame.display.flip()
        clock.tick()

        # Get the current location and the readings there.
        x, y = self.car_body.position
        readings = self.get_sonar_readings(x, y, self.car_body.angle)
        normalized_readings = [(x-20.0)/20.0 for x in readings] 
        state = np.array([normalized_readings])

        # Set the reward.
        # Car crashed when any reading == 1
        if self.car_is_crashed(readings):
            self.crashed = True
            reward = -500
            self.recover_from_crash(driving_direction)
        else:
            reward = -100 + int(Vec2d.get_length(self.car_body.velocity))

        self.num_steps += 1

        return reward, state

    def move_obstacles(self):
        # Randomly move obstacles around.
        for obstacle in self.obstacles:
            speed = random.randint(1, 5)
            direction = Vec2d(1, 0).rotated(self.car_body.angle + random.randint(-2, 2))
            obstacle.velocity = speed * direction

    def move_cat(self):
        for cat in self.cat:
            cat.angle -= random.randint(-1, 1)
            direction = Vec2d(1, 0).rotated(cat.angle)
            impulse = 500 * Vec2d(1,0)
            impulse.rotate(cat.angle)
            cat.apply_impulse_at_world_point(impulse, cat.position)
            

    def car_is_crashed(self, readings):
        crash = False
        for i in range(17):
            if readings[i] == 1:
                crash = True
                break

        return crash

    def recover_from_crash(self, driving_direction):
        """
        We hit something, so recover.
        """
        while self.crashed:
            self.car_body.position = 100, 100
            self.crashed = False
            
            for i in range(10):
                screen.fill(THECOLORS["red"])  # Red is scary!

                options = pymunk.pygame_util.DrawOptions(screen)
                self.space.debug_draw(options)

                self.space.step(1./10)
                if draw_screen:
                    pygame.display.flip()
                clock.tick()

    def sum_readings(self, readings):
        """Sum the number of non-zero readings."""
        tot = 0
        for i in readings:
            tot += i
        return tot

    def get_sonar_readings(self, x, y, angle):
        readings = []
        """
        Instead of using a grid of boolean(ish) sensors, sonar readings
        simply return N "distance" readings, one for each sonar
        we're simulating. The distance is a count of the first non-zero
        reading starting at the object. For instance, if the fifth sensor
        in a sonar "arm" is non-zero, then that arm returns a distance of 5.
        """
        # Make our arms.
        arm_left = self.make_sonar_arm(x, y)
        arm_middle_left_0 = arm_left
        arm_middle_left_1 = arm_left
        arm_middle_left_2 = arm_left
        arm_middle_left_3 = arm_left
        arm_middle_left_4 = arm_left
        arm_middle_left_5 = arm_left
        arm_middle_left_6 = arm_left
        arm_middle = arm_left
        arm_middle_right_6 = arm_left
        arm_middle_right_5 = arm_left
        arm_middle_right_4 = arm_left
        arm_middle_right_3 = arm_left
        arm_middle_right_2 = arm_left
        arm_middle_right_1 = arm_left
        arm_middle_right_0 = arm_left
        arm_right = arm_left

        # Rotate them and get readings.
        readings.append(self.get_arm_distance(arm_left, x, y, angle, 1.8))
        readings.append(self.get_arm_distance(arm_middle_left_0, x, y, angle, 1.5))
        readings.append(self.get_arm_distance(arm_middle_left_1, x, y, angle, 1.2))
        readings.append(self.get_arm_distance(arm_middle_left_2, x, y, angle, 0.9))
        readings.append(self.get_arm_distance(arm_middle_left_3, x, y, angle, 0.7))
        readings.append(self.get_arm_distance(arm_middle_left_4, x, y, angle, 0.5))
        readings.append(self.get_arm_distance(arm_middle_left_5, x, y, angle, 0.3))
        readings.append(self.get_arm_distance(arm_middle_left_6, x, y, angle, 0.15))
        readings.append(self.get_arm_distance(arm_middle, x, y, angle, 0))
        readings.append(self.get_arm_distance(arm_middle_right_6, x, y, angle, -0.15))
        readings.append(self.get_arm_distance(arm_middle_right_5, x, y, angle, -0.3))
        readings.append(self.get_arm_distance(arm_middle_right_4, x, y, angle, -0.5))
        readings.append(self.get_arm_distance(arm_middle_right_3, x, y, angle, -0.7))
        readings.append(self.get_arm_distance(arm_middle_right_2, x, y, angle, -0.9))
        readings.append(self.get_arm_distance(arm_middle_right_1, x, y, angle, -1.2))
        readings.append(self.get_arm_distance(arm_middle_right_0, x, y, angle, -1.5))
        readings.append(self.get_arm_distance(arm_right, x, y, angle, -1.8))

        if show_sensors:
            pygame.display.update()

        return readings

    def get_arm_distance(self, arm, x, y, angle, offset):
        # Used to count the distance.
        i = 0

        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1

            # Move the point to the right spot.
            rotated_p = self.get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )

            # Check if we've hit something. Return the current i (distance)
            # if we did.
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= width or rotated_p[1] >= height:
                return i  # Sensor is off the screen.
            else:
                obs = screen.get_at(rotated_p)
                if self.get_track_or_not(obs) != 0:
                    return i

            if show_sensors:
                pygame.draw.circle(screen, (255, 255, 255), (rotated_p), 2)

        # Return the distance for the arm.
        return i

    def make_sonar_arm(self, x, y):
        spread = 10  # Default spread.
        distance = 20  # Gap before first sensor.
        arm_points = []
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        for i in range(1, 40):
            arm_points.append((distance + x + (spread * i), y))

        return arm_points

    def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + \
            (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - \
            (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = height - (y_change + y_1)
        return int(new_x), int(new_y)

    def get_track_or_not(self, reading):
        if reading == THECOLORS['black']:
            return 0
        else:
            return 1

if __name__ == "__main__":
    game_state = GameState()
    while True:
        game_state.frame_step((random.randint(0, 3)))
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                sys.exit(0)


    pygame.quit()