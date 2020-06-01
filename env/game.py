import sys
import pygame
from objects import *
from configs import *
from ple.games import base

# ==============================================
# Car Simulator Class for RL Algorithm Training
# ==============================================

class CarSimulator(base.PyGameWrapper):
    def __init__(self, args, width=SCREEN_WIDTH, height=SCREEN_HEIGHT):
        base.PyGameWrapper.__init__(self, width, height)

        self.args = args
        self.screen = pygame.display.set_mode(self.getScreenDims(), 0, 32)
        pygame.display.set_caption(CAPTION)
        self.clock = pygame.time.Clock()

    def reset(self):
        self.reward = 0
        self.car = Car(0, START_X, START_Y) # TODO: change this
        self.road = Road()
        self.sensors = Sensors(self.car, self.road)
        self.info = Info(self.sensors)

    @property
    def _done(self): return self.done

    @property
    def state(self):
        # ==============================================
        # State:
        # - pose, acceleration, velocity, road friction level
        # - distance to center lane and lane curvature
        # ==============================================
        pose = self.car.pose
        moments = [self.car.v, self.car.dv]
        friction = [self.car.Bv, self.car.Bw]

        measurements = []
        for sensor in self.sensors.sensors:
            measurements.append(sensor.measurement)

        state = pose + moments + friction + measurements
        return state

    def _handle_player_events(self):
        action = [0, 0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = [0, LATERAL_FORCE]
        if keys[pygame.K_RIGHT]:
            action = [LATERAL_FORCE, 0]
        if keys[pygame.K_UP]:
            action = [FORWARD_FORCE, FORWARD_FORCE]
        if keys[pygame.K_DOWN]:
            action = [-FORWARD_FORCE, -FORWARD_FORCE]

        return action

    def collision_check(self):
        pose = self.car.pose[:2]
        pose = [int(ele) for ele in pose]
        return self.road.field_mask.overlap(self.car.img_mask, pose)

    def update_friction(self):
        pose = self.car.pose[:2]
        pose = [int(ele) for ele in pose]

        if not self.car.img_mask: return

        # Set rotational and translational coefficients of vehicle
        for texture, dic in self.road.texture_map.items():
            if self.road.texture_map[texture]['mask'].overlap(self.car.img_mask, pose):
                self.car.set_friction(dic['friction_level'], 1)
                break
            else:
                self.car.set_friction(1, 1)

    def step(self, dt):
        self.screen.fill(WHITE)
        action = self._handle_player_events()

        # ==============================================
        # Update friction and update all of the env objects
        # ==============================================
        self.update_friction()
        self.road.blit(self.screen)
        self.car.blit(self.screen, action, dt)
        self.sensors.blit(self.screen)

        done = False
        if self.collision_check() is not None:
            done = True
        else:
            self.reward += 1

        import collections
        pose = self.car.pose
        moments = [self.car.v, self.car.dv]
        friction = [self.car.Bv, self.car.Bw]
        info = collections.OrderedDict(
            pose=pose,
            moments=moments,
            friction=friction,
            reward=self.reward,
            done=done
        )

        self.info.blit(self.screen, info)

        return self.state, self.reward, done, info

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=4, help="random seed")
    parser.add_argument('--verbose', type=int, default=0, help="verbose")

    args, unknown = parser.parse_known_args()
    args = vars(args)

    pygame.init()
    game = CarSimulator(args)
    game.reset()

    while True:
        dt = game.clock.tick_busy_loop(30) / 1000.0 # in milliseconds
        state, reward, done, info = game.step(dt)

        if done: game.reset()
        pygame.display.update()
