import math
import time

import Box2D
import gym
import numpy as np
import pyglet
# Dynamic binding
# noinspection PyUnresolvedReferences,PyUnresolvedReferences,PyUnresolvedReferences
from Box2D.b2 import (fixtureDef, polygonShape, contactListener)
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.utils import seeding
from pyglet import gl

# Easiest continuous control task to learn from pixels, a top-down racing environment.
# Discreet control is reasonable in this environment as well, on/off discretisation is
# fine.
#
# State consists of STATE_W x STATE_H pixels.
#
# Reward is -0.1 every frame and +1000/N for every track tile visited, where N is
# the total number of tiles in track. For example, if you have finished in 732 frames,
# your reward is 1000 - 0.1*732 = 926.8 points.
#
# Game is solved when agent consistently gets 900+ points. Track is random every episode.
#
# Episode finishes when all tiles are visited. Car also can go outside of PLAYFIELD, that
# is far off the track, then it will get -100 and die.
#
# Some indicators shown at the bottom of the window and the state RGB buffer. From
# left to right: true speed, four ABS sensors, steering wheel position, gyroscope.
#
# To play yourself (it's rather fast for humans), type:
#
# python gym/envs/box2d/car_racing.py
#
# Remember it's powerful rear-wheel drive car, don't press accelerator and turn at the
# same time.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1200
WINDOW_H = 1000

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50
ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)

MAX_SPEED = 101

TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]

ACTIVATE_PROFILING = False

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def projection(initial_point, point_to_project, vector_to_project_onto):
    return initial_point + np.dot(point_to_project - initial_point, vector_to_project_onto) * vector_to_project_onto

def cross_2d(v_2d, v_3d):
    return np.cross(np.concatenate([v_2d, [0]]), v_3d)[:2]

class Profiling():
    i = 0
    def __init__(self, name=""):
        self.name = name
        self.id = Profiling.i
        Profiling.i += 1

    def __enter__(self):
        self.time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if ACTIVATE_PROFILING:
            print(f'Block n_{self.id} ({self.name}): {(time.perf_counter() - self.time) * 1000:.2f}ms')

def profile(func):
    def wrapper(*args, **kwargs):
        with Profiling(func.__name__):
            return func(*args, **kwargs)
    return wrapper

class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        tile.color[0] = ROAD_COLOR[0]
        tile.color[1] = ROAD_COLOR[1]
        tile.color[2] = ROAD_COLOR[2]
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            # print tile.road_friction, "ADD", len(obj.tiles)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0 / len(self.env.track)
                self.env.tile_visited_count += 1
        else:
            obj.tiles.remove(tile)
            # print tile.road_friction, "DEL", len(obj.tiles) -- should delete to zero when on grass (this works)


class CarRacing(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second': FPS
    }

    def __init__(self, use_internal_state=False, draw_debug=False):
        # PEP8 prior declaration
        self.np_random = None
        self.tile_visited_count = None
        self.t = None
        self.road_poly = None
        self.human_render = None
        self.state = None
        self.score_label = None
        self.transform = None
        self.out_of_the_road = False
        self.previous_render_time = None
        self.steps_without_moving = 0

        self.seed()
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0

        self.closest_points = (0, 1)

        self.use_internal_state = use_internal_state
        self.draw_debug = draw_debug

        self.action_space = spaces.Box(np.array([-1, 0, 0]), np.array([+1, +1, +1]))  # steer, gas, brake
        if self.use_internal_state:
            self.observation_space = spaces.Box(low=-1, high=1, shape=(15,))
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        self.car.destroy()

    @profile
    def _create_track(self):
        nb_checkpoints = 12

        # Create checkpoints
        checkpoints = []
        for c in range(nb_checkpoints):
            alpha = 2.0 * math.pi * c / nb_checkpoints + self.np_random.uniform(0, 2 * math.pi * 1 / nb_checkpoints)
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)
            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == nb_checkpoints - 1:
                alpha = 2 * math.pi * c / nb_checkpoints
                self.start_alpha = 2 * math.pi * (-0.5) / nb_checkpoints
                rad = 1.5 * TRACK_RAD
            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))

        # print "\n".join(str(h) for h in checkpoints)
        # self.road_poly = [ (    # uncomment this to see checkpoints
        #    [ (tx,ty) for a,tx,ty in checkpoints ],
        #    (0.7,0.7,0.9) ) ]
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while 1:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi
            while True:  # Find destination from checkpoints
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break
                if not failed:
                    break
                alpha -= 2 * math.pi
                continue
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            proj = r1x * dest_dx + r1y * dest_dy  # destination vector projected on rad
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break
        # print "\n".join([str(t) for t in enumerate(track)])

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = track[i][0] > self.start_alpha >= track[i - 1][0]
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        # print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1:i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2])) +
            np.square(first_perp_y * (track[0][3] - track[-1][3])))
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (x1 - TRACK_WIDTH * math.cos(beta1), y1 - TRACK_WIDTH * math.sin(beta1))
            road1_r = (x1 + TRACK_WIDTH * math.cos(beta1), y1 + TRACK_WIDTH * math.sin(beta1))
            road2_l = (x2 - TRACK_WIDTH * math.cos(beta2), y2 - TRACK_WIDTH * math.sin(beta2))
            road2_r = (x2 + TRACK_WIDTH * math.cos(beta2), y2 + TRACK_WIDTH * math.sin(beta2))
            t = self.world.CreateStaticBody(fixtures=fixtureDef(
                shape=polygonShape(vertices=[road1_l, road1_r, road2_r, road2_l])
            ))
            t.userData = t
            c = 0.01 * (i % 3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = False
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (x1 + side * TRACK_WIDTH * math.cos(beta1), y1 + side * TRACK_WIDTH * math.sin(beta1))
                b1_r = (x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                        y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1))
                b2_l = (x2 + side * TRACK_WIDTH * math.cos(beta2), y2 + side * TRACK_WIDTH * math.sin(beta2))
                b2_r = (x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                        y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2))
                self.road_poly.append(([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0)))
        self.track = np.array(track)
        return True

    @profile
    def reset(self):
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []
        self.human_render = False
        self.out_of_the_road = False
        self.steps_without_moving = 0
        self.closest_points = (0, 1)

        while True:
            success = self._create_track()
            if success:
                break
            # print("retry to generate track (normal if there are not many of this messages)")
        self.car = Car(self.world, *self.track[0][1:4])

        return self.step(None)[0]

    @profile
    def step(self, action):
        if action is not None:
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.update_closest_points()

        if self.use_internal_state:
            self.state = self.get_input()
        else:
            self.state = self.render("state_pixels")

        step_reward = 0
        done = False
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            if not self.use_internal_state:
                step_reward = self.reward - self.prev_reward
            else:
                # road_direction = np.concatenate((self.road_direction, [0]))
                # car_velocity = np.concatenate((self.car_velocity, [0]))
                # up = np.array([0, 0, 1])
                #
                # # right vector is cross(road_direction, up) and velocity on right is dot(velocity, right)
                # # so velocity_y = velocity . (road_direction x up) => Triple product => det(a,b,c)
                #
                step_reward = np.dot(self.car_velocity, self.road_direction) * (1 - 2 * np.abs(self.state[0])) / MAX_SPEED
                #               - np.abs(np.linalg.det([car_velocity, road_direction, up])) \

            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track):
                done = True
            x, y = self.car.hull.position
            if np.linalg.norm(self.car_velocity) < 2:
                self.steps_without_moving += 1
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD or self.out_of_the_road or self.steps_without_moving > 3*FPS:
                done = True
                step_reward = -1

        return self.state, step_reward, done, {}

    @profile
    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label('0000', font_size=36,
                                                 x=20, y=WINDOW_H * 2.5 / 40.00, anchor_x='left', anchor_y='center',
                                                 color=(255, 255, 255, 255))
            self.transform = rendering.Transform()

        if self.t is None:
            return  # reset() not called yet

        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)  # Animate zoom first second
        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W / 2 - (scroll_x * zoom * math.cos(angle) - scroll_y * zoom * math.sin(angle)),
            WINDOW_H / 4 - (scroll_x * zoom * math.sin(angle) + scroll_y * zoom * math.cos(angle)))
        self.transform.set_rotation(angle)

        self.car.draw(self.viewer, mode != "state_pixels")

        arr = None
        win = self.viewer.window
        if mode != 'state_pixels':
            win.switch_to()
            win.dispatch_events()
        if mode == "rgb_array" or mode == "state_pixels":
            win.clear()
            t = self.transform
            if mode == 'rgb_array':
                vp_w = VIDEO_W
                vp_h = VIDEO_H
            else:
                vp_w = STATE_W
                vp_h = STATE_H
            gl.glViewport(0, 0, vp_w, vp_h)
            t.enable()
            self.render_road()
            if self.draw_debug:
                self.render_debug()
            for geom in self.viewer.onetime_geoms:
                geom.render()
            t.disable()
            self.render_indicators(WINDOW_W, WINDOW_H)  # TODO: find why 2x needed, wtf
            image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            arr = arr.reshape(vp_h, vp_w, 4)
            arr = arr[::-1, :, 0:3]

        # agent can call or not call env.render() itself when recording video.
        if mode == "rgb_array" and not self.human_render:
            win.flip()

        if mode == 'human':
            self.human_render = True
            win.clear()
            t = self.transform
            gl.glViewport(0, 0, WINDOW_W, WINDOW_H)
            t.enable()
            self.render_road()
            if self.draw_debug:
                self.render_debug()
            for geom in self.viewer.onetime_geoms:
                geom.render()
            t.disable()
            self.render_indicators(WINDOW_W, WINDOW_H)
            win.flip()

        self.viewer.onetime_geoms = []
        return arr

    @property
    def car_position(self):
        return np.array(self.car.hull.position)

    @property
    def car_velocity(self):
        return np.array(self.car.hull.linearVelocity)

    @property
    def car_direction(self):
        angle = self.car.hull.angle
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 1])
        right = np.cos(angle) * x + np.sin(angle) * y
        return -np.cross(right, np.array([0, 0, 1]))[:-1]

    @property
    def road_direction(self):
        index1, index2 = self.find_closest_point_and_next()
        direction = self.track[index2][2:] - self.track[index1][2:]
        assert direction.any()
        return direction / np.linalg.norm(direction)

    def update_closest_points(self):
        point1 = self.track[self.closest_points[0] - 1][2:]
        point2 = self.track[self.closest_points[0]][2:]
        point3 = self.track[self.closest_points[1]][2:]
        point4 = self.track[(self.closest_points[1] + 1) % len(self.track)][2:]

        dist1 = np.linalg.norm(self.car_position - point1)
        dist2 = np.linalg.norm(self.car_position - point2)
        dist3 = np.linalg.norm(self.car_position - point3)
        dist4 = np.linalg.norm(self.car_position - point4)

        dists = [dist1, dist2, dist3, dist4]
        min_dist = min(*dists)

        if min_dist == dist1 or (min_dist == dist2 and dist1 < dist3):
            # Go back
            self.closest_points = (self.closest_points[0] - 1, self.closest_points[0])
        elif min_dist == dist4 or (min_dist == dist3 and (dist4 < dist2)):
            # Go further
            self.closest_points = (self.closest_points[1], (self.closest_points[1] + 1) % len(self.track))

    def find_closest_point_and_next(self):
        return self.closest_points

    def get_points_further_ahead(self, index, starting_point, ahead_distances):
        previous = starting_point
        dist = 0
        points = []
        for ahead_distance in ahead_distances:
            while dist < ahead_distance:
                point = self.track[index][2:]
                index = (index + 1) % len(self.track)
                new_dist = dist + np.linalg.norm(point - previous)
                if new_dist > ahead_distance:
                    # Do a linear interpolation
                    removing_dist = new_dist - ahead_distance
                    direction = (point - previous)
                    direction /= np.linalg.norm(direction)
                    points.append(point - removing_dist * direction)
                dist = new_dist
                previous = point
        return points

    def projection_on_road(self):
        index1, index2 = self.find_closest_point_and_next()
        point1 = self.track[index1][2:]
        point2 = self.track[index2][2:]
        direction = point2 - point1
        assert direction.any()
        direction /= np.linalg.norm(direction)

        position = self.car_position
        return projection(point1, position, direction), index2

    def projections_further_ahead(self, index, starting_point, ahead_distances):
        points_ahead = self.get_points_further_ahead(index, starting_point, ahead_distances)
        position = self.car_position
        direction = self.car_direction

        projs = list(map(lambda p: projection(position, p, direction), points_ahead))

        return projs, points_ahead

    @profile
    def get_input(self):
        projection, index = self.projection_on_road()

        velocity = self.car_velocity / MAX_SPEED

        road_direction = self.road_direction
        angle = np.arccos(np.dot(self.car_direction, road_direction))

        top = np.array([0, 0, 1])
        left = cross_2d(road_direction, top)

        velocity_x = np.dot(velocity, road_direction)
        velocity_y = np.dot(velocity, left)

        distance_from_road = np.linalg.norm(self.car_position - projection) * \
                             np.sign(np.dot(self.car_position - projection, left)) / \
                             TRACK_WIDTH

        angle *= np.sign(np.dot(self.car_direction, left)) / np.pi

        if np.linalg.norm(velocity) > 0.05:
            drift_angle = np.dot(self.car_direction, self.car_velocity / np.linalg.norm(self.car_velocity))
        else:
            drift_angle = 1

        res = [distance_from_road, angle, velocity_x, velocity_y, drift_angle]

        distances = np.array((5, 20, 35))
        projs, points_ahead = self.projections_further_ahead(index, projection, distances)
        projs = np.array(projs)
        points_ahead = np.array(points_ahead)

        v_vectors = projs - self.car_position
        h_vectors = points_ahead - projs
        v = np.linalg.norm(v_vectors, axis=1) * np.sign(np.dot(v_vectors, self.car_direction)) / distances
        h = np.linalg.norm(h_vectors, axis=1) * np.sign(np.cross(h_vectors, self.car_direction)) / distances
        res += list(v) + list(h)

        res.extend([w.omega / 360.0 for w in self.car.wheels])

        self.out_of_the_road = abs(distance_from_road) > 1

        return np.array(res)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def render_road(self):
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0.4, 0.8, 0.4, 1.0)
        gl.glVertex3f(-PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, -PLAYFIELD, 0)
        gl.glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)
        gl.glColor4f(0.4, 0.9, 0.4, 1.0)
        k = PLAYFIELD / 20.0
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                gl.glVertex3f(k * x + k, k * y + 0, 0)
                gl.glVertex3f(k * x + 0, k * y + 0, 0)
                gl.glVertex3f(k * x + 0, k * y + k, 0)
                gl.glVertex3f(k * x + k, k * y + k, 0)
        for poly, color in self.road_poly:
            gl.glColor4f(color[0], color[1], color[2], 1)
            for p in poly:
                gl.glVertex3f(p[0], p[1], 0)
        gl.glEnd()

    def render_debug(self):
        gl.glBegin(gl.GL_LINES)
        gl.glColor4f(0, 1, 1, 1)
        for i in range(len(self.track)):
            a1, b1, x1, y1 = self.track[i - 1]
            a2, b2, x2, y2 = self.track[i]
            gl.glVertex3f(x1, y1, 0)
            gl.glVertex3f(x2, y2, 0)
        gl.glColor4f(1, 1, 0, 1)
        proj, index = self.projection_on_road()
        gl.glVertex3f(*proj, 0)
        gl.glVertex3f(*self.car_position, 0)
        gl.glColor4f(0, 0, 1, 1)
        gl.glVertex3f(*self.car_position, 0)
        gl.glVertex3f(*(self.car_position + self.car_velocity), 0)
        gl.glColor4f(1, 0, 1, 1)
        distances = [5, 20, 35]
        projs, points_ahead = self.projections_further_ahead(index, proj, distances)
        gl.glVertex3f(*self.car_position, 0)
        gl.glVertex3f(*projs[2], 0)
        for proj_, point_ahead in zip(projs, points_ahead):
            gl.glVertex3f(*proj_, 0)
            gl.glVertex3f(*point_ahead, 0)
        gl.glEnd()

    def display_input_debug(self):
        text = ['Distance', 'Angle', 'Velocity x', 'Velocity y', 'Drift Angle',
                '5 meter v',
                '20 meter v',
                '35 meter v',
                '5 meter h',
                '20 meter h',
                '35 meter h',
                'omega_1', 'omega_2', 'omega_3', 'omega_4']

        if self.state is None or len(self.state) != len(text):
            input_ = self.get_input()
        else:
            input_ = self.state

        y = WINDOW_H - 20

        for i, data in enumerate(input_):
            pyglet.text.Label(f'{text[i]}: {data:.2f}', font_size=30,
                              x=10, y=y, anchor_x='left', anchor_y='center',
                              color=(255, 255, 255, 255)).draw()
            y -= 50

    def render_indicators(self, w, h):
        gl.glBegin(gl.GL_QUADS)
        s = w / 40.0
        h = h / 40.0
        gl.glColor4f(0, 0, 0, 1)
        gl.glVertex3f(w, 0, 0)
        gl.glVertex3f(w, 5 * h, 0)
        gl.glVertex3f(0, 5 * h, 0)
        gl.glVertex3f(0, 0, 0)

        def vertical_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place + 0) * s, h + h * val, 0)
            gl.glVertex3f((place + 1) * s, h + h * val, 0)
            gl.glVertex3f((place + 1) * s, h, 0)
            gl.glVertex3f((place + 0) * s, h, 0)

        def horiz_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place + 0) * s, 4 * h, 0)
            gl.glVertex3f((place + val) * s, 4 * h, 0)
            gl.glVertex3f((place + val) * s, 2 * h, 0)
            gl.glVertex3f((place + 0) * s, 2 * h, 0)

        true_speed = np.sqrt(np.square(self.car.hull.linearVelocity[0]) + np.square(self.car.hull.linearVelocity[1]))
        vertical_ind(5, 0.02 * true_speed, (1, 1, 1))
        vertical_ind(7, 0.01 * self.car.wheels[0].omega, (0.0, 0, 1))  # ABS sensors
        vertical_ind(8, 0.01 * self.car.wheels[1].omega, (0.0, 0, 1))
        vertical_ind(9, 0.01 * self.car.wheels[2].omega, (0.2, 0, 1))
        vertical_ind(10, 0.01 * self.car.wheels[3].omega, (0.2, 0, 1))
        horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle, (0, 1, 0))
        horiz_ind(30, -0.8 * self.car.hull.angularVelocity, (1, 0, 0))
        gl.glEnd()
        self.score_label.text = "%04i" % self.reward
        self.score_label.draw()

        if self.draw_debug:
            self.display_input_debug()


if __name__ == "__main__":
    from pyglet.window import key

    a = np.array([0.0, 0.0, 0.0])


    def key_press(k, _):
        global restart
        if k == 0xff0d:
            restart = True
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[2] = +0.8  # set 1.0 for wheels to block to zero rotation


    def key_release(k, _):
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0


    env_ = CarRacing(use_internal_state=True, draw_debug=True)
    env_.render()
    record_video = False
    if record_video:
        env_.monitor.start('/tmp/video-test', force=True)
    env_.viewer.window.on_key_press = key_press
    env_.viewer.window.on_key_release = key_release
    for _ in range(2):
        env_.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            state, r, done_, info = env_.step(a)
            total_reward += r
            if steps % 200 == 0 or done_:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                # import matplotlib.pyplot as plt
                # plt.imshow(s)
                # plt.savefig("test.jpeg")
            steps += 1
            if not record_video:  # Faster, but you can as well call env.render() every time to play full window.
                env_.render()
            if done_ or restart:
                break
    env_.close()
