import pygame
import math
from configs import *
import os
import numpy as np

# ==============================================
# Car Class
# ==============================================
class Car(pygame.sprite.Sprite):
    def __init__(self, init_angle, init_x, init_y):
        self.image = pygame.image.load(CAR_FILE)
        self.img = None
        self.img_mask = None
        self.img_rect = None

        self.x = init_x
        self.y = init_y
        self.theta = init_angle
        self.v = 0                  # center velocity
        self.w = 0                  # angular velocity
        self.dv = 0                 # acceleration
        self.dw = 0                 # angular acceleratino
        self.l = 1                  # half the length of axis between the wheel
        self.M = 1                  # mass of unicycle
        self.J = 1                  # inertia moment
        self.Bv = 1                 # translational friction coefficient
        self.Bw = 1                 # rotational friction coefficient

        pygame.sprite.Sprite.__init__(self)

    def reset(self):
        self.theta = 0
        self.x = self.init_x
        self.y = self.init_y

    @property
    def xpos(self): return self.x

    @property
    def ypos(self): return SCREEN_HEIGHT - self.y

    @property
    def theta_degrees(self):
        return self.theta*(180/math.pi)

    @property
    def pose(self): return [self.xpos, self.ypos, self.theta_degrees]

    @property
    def center(self): return [self.xpos + self.img_rect.center[0], self.ypos + self.img_rect.center[1]]

    def set_friction(self, trans_coef, rot_coef):
        self.Bv = trans_coef
        self.Bw = rot_coef

    def step(self, action, dt):
        # ================================
        # Unicycle Dynamics Model
        # ================================

        if action is None: return
        F = action[0] + action[1]
        T = self.l * (action[1] - action[0])

        # the acceleration
        self.dw = (T - self.Bw*self.w)/self.J
        self.dv = (F - self.Bv*self.v)/self.M

        # distance
        ds = self.v*dt + 0.5*self.dv*(dt**2)
        dtheta = self.w*dt + 0.5*self.dw*(dt**2)

        # update state
        self.x = self.x + ds*math.cos(self.theta)
        self.y = self.y + ds*math.sin(self.theta)
        self.theta = self.theta + dtheta

        # update velocity
        self.w = self.w + self.dw*dt
        self.v = self.v + self.dv*dt

    # Rotate an image while keeping its center and size
    def rotate(self):
        orig_rect = self.image.get_rect()
        rot_image = pygame.transform.rotate(self.image, self.theta_degrees)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image


    def blit(self, screen, action, dt):
        self.img = self.rotate()
        self.img_mask = pygame.mask.from_surface(self.img)
        self.img_rect = self.img.get_rect()
        screen.blit(self.img, self.pose[:2])
        self.step(action, dt)

# ==============================================
# Game Info class
# ==============================================
class Info:
    def __init__(self, sensors):
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        self.sensors = sensors

    def process_values(self, v):
        subtext = None

        if type(v) == list:
            v = [round(ele, 3) for ele in v]
            subtext = " {} " * len(v)
            subtext = subtext.format(*v)
        elif type(v) == int or type(v) == float or isinstance(v, np.float64):
            subtext = " {} "
            subtext = subtext.format(round(float(v), 3))
        elif type(v) == bool or type(v) == str:
            subtext = " {} "
            subtext = subtext.format(v)
        else:
            print(type(v))
            raise NotImplementedError

        return subtext

    def blit(self, screen, info):
        num_keys = len(info)

        for i, (k, v) in enumerate(info.items()):
            subtext = self.process_values(v)
            text = "{}:".format(k) + subtext
            label = self.font.render(text, 1, WHITE)
            label_size = self.font.size(text)
            screen.blit(label, (SCREEN_WIDTH - label_size[0], SCREEN_HEIGHT - label_size[1] * (num_keys - i)))

        num_sensors = len(self.sensors)
        for i, sensor in enumerate(self.sensors._sensors):
            subtext = self.process_values(sensor.measurement)
            sensor_text = "{}:".format(sensor.name) + subtext
            label = self.font.render(sensor_text, 1, WHITE)
            label_size = self.font.size(sensor_text)
            screen.blit(label, (SCREEN_WIDTH - label_size[0], label_size[1] * i))

# ==============================================
# Sensor base class
# ==============================================
from abc import ABCMeta, abstractmethod

class Sensor(object):
    def __init__(self, car, road):
        self.car = car
        self.road = road

    @abstractmethod
    def measure(self):
        pass

    @abstractmethod
    def blit(self, screen):
        pass

# ==============================================
# Sensor for computing distance from car to center of lane
# ==============================================
class CenterLaneSensor(Sensor):
    def __init__(self, _id, car, road):
        super(CenterLaneSensor, self).__init__(car, road)
        self.name = 'CenterLaneSensor' + str(_id)

    def measure(self):
        car_center = np.array(self.car.center)[np.newaxis, :]

        mask = pygame.surfarray.array2d(self.road.center_lane)
        coords = np.array(mask.nonzero()).transpose()

        import scipy.spatial as sp

        distances = sp.distance.cdist(car_center, coords)
        point = coords[distances.argmin()]
        dist = distances.min()

        self.measurement = "P: {}, D: {}".format(point, round(dist, 3))
        self.closest_point = point
        return point, dist

    def blit(self, screen):
        pygame.draw.circle(screen, BLUE, tuple(self.closest_point), 10)

# ==============================================
# Sensor for computing lane curvature
# ==============================================
class LaneCurvatureSensor(Sensor):
    def __init__(self, _id, car, road):
        super(LaneCurvatureSensor, self).__init__(car, road)
        self.name = 'LaneCurvatureSensor' + str(_id)

    def measure(self):
        car_center = np.array(self.car.center)[np.newaxis, :]

        mask = pygame.surfarray.array2d(self.road.center_lane)
        coords = np.array(mask.nonzero()).transpose()

        import scipy.spatial as sp

        distances = sp.distance.cdist(car_center, coords).squeeze(0)

        closest_point = coords[distances.argmin()]
        x, y = closest_point

        win_size = 50

        # Take a region around the closest point
        window = mask[x-win_size:x+win_size, y-win_size:y+win_size]

        from itertools import product
        prod = np.array(list(product(range(x-win_size,x+win_size), range(y-win_size,y+win_size))))
        coords = prod[window.flatten().nonzero()]
        self.coords = coords

        # ==============================================
        # Some code to compute lane curvature
        # ==============================================

        dx_dt = np.gradient(coords[:, 0])
        dy_dt = np.gradient(coords[:, 1])
        velocity = np.stack((dx_dt, dy_dt),axis=1)

        ds_dt = np.sqrt(dx_dt**2 + dy_dt**2)
        tangent = np.array([1/ds_dt] * 2).transpose() * velocity

        tangent_x = tangent[:, 0]
        tangent_y = tangent[:, 1]

        deriv_tangent_x = np.gradient(tangent_x)
        deriv_tangent_y = np.gradient(tangent_y)

        dT_dt = np.stack((deriv_tangent_x, deriv_tangent_y),axis=1)

        length_dT_dt = np.sqrt(deriv_tangent_x ** 2 + deriv_tangent_y **2)

        normal = np.array([1/length_dT_dt] * 2).transpose() * dT_dt

        d2s_dt2 = np.gradient(ds_dt)
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)

        # use formula for curvature: |x'' y' - x'y''| / (x'^2 + y'^2) ^(3/2)
        curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**(3/2)

        t_component = np.array([d2s_dt2] * 2).transpose()
        n_component = np.array([curvature * ds_dt * ds_dt] * 2).transpose()

        acceleration = t_component * tangent + n_component * normal

        self.measurement = curvature[int(len(curvature) / 2)]
        return self.measurement

    def blit(self, screen):
        for point in self.coords:
            pygame.draw.circle(screen, ORANGE, tuple(point), 2)

# ==============================================
# Sensor ray trace distance to edge of the road
# ==============================================
class RangeSensor(Sensor):
    def __init__(self, _id, car, road, deg):
        self.deg = deg
        self.name = 'RangeSensor' + str(_id)
        super(RangeSensor, self).__init__(car, road)

    def measure(self):
        car_center = self.car.center

        length = 0
        for i in range(1500):
            self.deg_x = car_center[0] + math.cos(math.radians(self.deg + self.car.theta_degrees)) * length
            self.deg_y = car_center[1] - math.sin(math.radians(self.deg + self.car.theta_degrees)) * length

            line_surface = pygame.Surface((2, 2), pygame.SRCALPHA)
            line_rect = line_surface.get_rect()
            line_rect.topleft = self.deg_x, self.deg_y
            line_surface.fill((255,0,0))
            if self.road.field_mask.overlap(pygame.mask.from_surface(line_surface),
                                             (int(line_rect[0]), int(line_rect[1]))) is not None:
                self.measurement = length
                break
            length += 1

    def blit(self, screen):
        area_clear = 200
        if self.measurement > area_clear:
            red = 0
            green = 255
        else:
            ratio = self.measurement / area_clear
            value = 510 * ratio
            if value <= 255:
                red = 255
                green = value
            else:
                red = 255 - (value - 255)
                green = 255

        pygame.draw.line(screen, (int(red), int(green), 0), tuple(self.car.center), (self.deg_x, self.deg_y), 2)

class Sensors:
    def __init__(self, car, road):
        self.car = car
        self.road = road
        self.num_range_sensors = NUM_RANGE_SENSORS
        self.density = ANGLE_BETWEEN_SENSORS
        self.range_sensors = self.init_range_sensors()
        self.center_lane_sensor = CenterLaneSensor(0, self.car, self.road)
        self.lane_curvature_sensor = LaneCurvatureSensor(0, self.car, self.road)

        self._sensors = self.range_sensors + [self.center_lane_sensor, self.lane_curvature_sensor]

    def __len__(self):
        return len(self.range_sensors)

    @property
    def sensors(self):
        return self._sensors

    def init_range_sensors(self):
        sensors = []
        angle_range = (self.num_range_sensors - 1) * self.density
        start_angle = -angle_range // 2

        for i, deg in enumerate(range(start_angle + angle_range, start_angle - 1, -self.density)):
            sensors.append(RangeSensor(i, self.car, self.road, deg))
        return sensors

    def blit(self, screen):
        for sensor in self.sensors:
            sensor.measure()
            sensor.blit(screen)

# ==============================================
# Road Class
# ==============================================
class Road:
    def __init__(self):
        self.img = pygame.image.load(TRACK_FILE)
        self.img = pygame.transform.smoothscale(self.img, (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.field_mask = pygame.mask.from_threshold(self.img, self.img.get_at((0,0)), (2, 2, 2, 255))
        self.img_rect = self.img.get_rect()

        # ==============================================
        # Load in different textures for the road
        # ==============================================
        self.texture_map = {}

        for name, info in TEXTURES.items():
            path, translation_coef = info
            texture = pygame.image.load(path)
            texture = pygame.transform.smoothscale(texture, (SCREEN_WIDTH, SCREEN_HEIGHT))
            texture_mask = pygame.mask.from_surface(texture)

            self.texture_map[name] = dict(
                texture=texture,
                mask=texture_mask,
                friction_level=translation_coef
            )

        # ==============================================
        # Load center lane
        # ==============================================
        self.center_lane = pygame.image.load(CENTER_LANE_FILE)
        # self.center_lane = pygame.transform.smoothscale(texture, (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.center_lane_mask = pygame.mask.from_surface(self.center_lane)

    def blit(self, screen):
        screen.blit(self.center_lane, (0, 0))

        # Render track first
        screen.blit(self.img, (0, 0))

        # Render textures on top
        for texture, dic in self.texture_map.items():
            screen.blit(dic['texture'], (0, 0))
