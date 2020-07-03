import os
import re
import numpy as np
import pygame
import math
from .utils import *
from .env_configs import *
import scipy.spatial as sp

# ==============================================
# Car Class
# ==============================================
class Car(pygame.sprite.Sprite):
    def __init__(self, init_angle, init_x, init_y, fps=60):
        self.image = pygame.image.load(CAR_FILE)
        self.img = None
        self.img_mask = None
        self.img_rect = None
        self.clock = pygame.time.Clock()
        self.fps = fps

        self.x = init_x
        self.y = init_y
        self.theta = init_angle
        self.v = 0                  # center velocity
        self.w = 0                  # angular velocity
        self.dv = 0                 # acceleration
        self.dw = 0                 # angular acceleration
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
    # pose of car with bottom left origin
    def pose(self): return [self.xpos, self.y, self.theta_degrees]

    @property
    # pose of car to render in pygame coordinates
    def to_screen_coords(self): return [self.xpos, self.ypos, self.theta_degrees]

    @property
    def center(self): return [self.xpos + self.img_rect.center[0], self.ypos + self.img_rect.center[1]]

    def set_friction(self, trans_coef, rot_coef):
        self.Bv = trans_coef
        self.Bw = rot_coef

    def step(self, action):
        # ================================
        # Unicycle Dynamics Model
        # ================================
        dt = self.clock.tick_busy_loop(self.fps) / 1000.0 # in milliseconds

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
        self.theta = wrap2pi(self.theta + dtheta)

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


    def blit(self, screen, action):
        self.img = self.rotate()
        self.img_mask = pygame.mask.from_surface(self.img)
        self.img_rect = self.img.get_rect()
        screen.blit(self.img, self.to_screen_coords[:2])
        self.step(action)

# ==============================================
# Game Info class
# ==============================================
class Info:
    def __init__(self, font_size=24):
        self.font = pygame.font.SysFont("monospace", font_size, bold=True)

    def set_sensors(self, sensors):
        self.sensors = sensors

    def process_values(self, v):
        subtext = None

        if type(v) == list or type(v) == np.ndarray:
            subtext = " {:.3f} " * len(v)
            subtext = subtext.format(*v)
        elif type(v) == int or type(v) == float or isinstance(v, np.float64):
            subtext = " {:.3f} "
            subtext = subtext.format(v)
        elif type(v) == bool or type(v) == str:
            subtext = " {} "
            subtext = subtext.format(v)
        else:
            print(type(v))
            raise NotImplementedError

        return subtext

    def blit_info(self, name, value):
        subtext = self.process_values(value)
        sensor_text = "{}:".format(name) + subtext
        label = self.font.render(sensor_text, 1, RED)
        label_size = self.font.size(sensor_text)
        return subtext, sensor_text, label, label_size

    def blit(self, screen, info):
        width, height = screen.get_size()
        num_keys = len(info)

        for i, (k, v) in enumerate(info.items()):
            subtext = self.process_values(v)
            text = "{}:".format(k) + subtext
            label = self.font.render(text, 1, RED)
            label_size = self.font.size(text)
            screen.blit(label, (width - label_size[0], height - label_size[1] * (num_keys - i)))

        if hasattr(self, 'sensors'):
            num_sensors = len(self.sensors)
            for i, sensor in enumerate(self.sensors._sensors):
                subtext, sensor_text, label, label_size = self.blit_info(sensor.name, sensor.text)
                screen.blit(label, (width - label_size[0], label_size[1] * i))

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

    @property
    def text(self): return self.measurement

# ==============================================
# Sensor for computing distance from car to center of lane
# ==============================================
class CenterLaneSensor(Sensor):
    def __init__(self, _id, car, road):
        super(CenterLaneSensor, self).__init__(car, road)
        self.name = 'CenterLaneSensor' + str(_id)
        self.measure()

    def measure(self):
        car_center = np.array(self.car.center)[np.newaxis, :]
        mask = pygame.surfarray.array2d(self.road.center_lane)
        coords = np.array(mask.nonzero()).transpose()

        distances = sp.distance.cdist(car_center, coords)
        point = coords[distances.argmin()]
        dist = distances.min()

        screen_point = to_screen_coords(point)
        self.closest_point = screen_point
        self.visual_closest_point = point

        # closest distance
        self.measurement = [dist, *screen_point]
        return screen_point, dist

    @property
    def text(self): return f"P: {self.closest_point}, D: {round(self.measurement[0], 3)}"

    def blit(self, screen):
        pygame.draw.circle(screen, BLUE, tuple(self.visual_closest_point), 10)


# ==============================================
# Sensor for computing lane direction
# ==============================================
class LaneDirectionSensor(Sensor):
    def __init__(self, _id, car, road, oriented_curve, num_future_info=0):
        super().__init__(car, road)
        self.name = 'LaneDirectionSensor'
        self.oriented_curve = oriented_curve
        self.num_future_info = num_future_info
        self.win = 5
        self.measure()

    def measure(self):
        lane_dir, indx = self.measure_lane_direction()
        angle_diff = self.measure_angle_diff()

        self.measurement = [lane_dir, angle_diff]

        if self.num_future_info > 0:
            self.measurement.extend(self.add_future_info(indx))

        return self.measurement

    def measure_lane_direction(self):
        car_center = np.array(self.car.center)[np.newaxis, :]
        mask = pygame.surfarray.array2d(self.road.center_lane)
        distances = sp.distance.cdist(car_center, self.oriented_curve).squeeze(0)
        indx = distances.argmin()

        p1 = self.oriented_curve[wrap(self.oriented_curve, indx, -self.win)]
        p2 = self.oriented_curve[wrap(self.oriented_curve, indx, self.win)]

        self.p1 = p1
        self.p2 = p2

        pos1 = to_screen_coords(p1)
        pos2 = to_screen_coords(p2)
        road_dir_angle = vec2angle(pos2 - pos1)

        car2center_point = to_screen_coords(car_center.squeeze()) - to_screen_coords(self.oriented_curve[indx])
        car_theta_rads = self.car.theta
        car_vec = np.array([math.cos(car_theta_rads), math.sin(car_theta_rads)])

        dot = car_vec[0]*-car2center_point[1] + car_vec[1]*car2center_point[0]

        # dot > 0 => center is on the right
        # left is negative, right is positive
        if dot < 0:
            road_dir_angle *= -1

        return road_dir_angle, indx

    def measure_angle_diff(self):
        car_theta_rads = self.car.theta
        car_vec = np.array([math.cos(car_theta_rads), math.sin(car_theta_rads)])
        import copy
        p1, p2 = copy.deepcopy(self.p1), copy.deepcopy(self.p2)
        p2[1] = SCREEN_HEIGHT - p2[1]
        p1[1] = SCREEN_HEIGHT - p1[1]
        road_vec = np.array(p2 - p1)

        # compute angle between two vectors
        u1 = car_vec / np.linalg.norm(car_vec)
        u2 = road_vec / np.linalg.norm(road_vec)
        angle_diff = np.arccos(np.dot(u1, u2)) # in radians

        return angle_diff

    def add_future_info(self, indx):
        future_dir_angle = []

        # Append the future lane curvature information
        for i in range(self.num_future_info):
            p1 = self.oriented_curve[wrap(self.oriented_curve, indx+i+1, -self.win)]
            p2 = self.oriented_curve[wrap(self.oriented_curve, indx+i+1, self.win)]
            pos1 = to_screen_coords(p1)
            pos2 = to_screen_coords(p2)
            road_dir_angle = vec2angle(pos2 - pos1)
            future_dir_angle.append(road_dir_angle)

        return future_dir_angle

    def blit(self, screen):
        pygame.draw.circle(screen, BLACK, tuple(self.p1), 2)
        pygame.draw.circle(screen, RED, tuple(self.p2), 2)

    @property
    def text(self):
        angle, angle_diff = self.measurement[:2]

        side = "left" if angle < 0 else "right"
        return f"road_dir: {angle:.2f}, angle_diff: {angle_diff:.2f}, side: {side}"


# ==============================================
# Sensor for computing lane curvature
# ==============================================
class LaneCurvatureSensor(Sensor):
    def __init__(self, _id, car, road):
        super(LaneCurvatureSensor, self).__init__(car, road)
        self.name = 'LaneCurvatureSensor' + str(_id)
        self.measure()

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
        super(RangeSensor, self).__init__(car, road)
        self.deg = deg
        self.name = 'RangeSensor' + str(_id)
        self.measure()

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
    def __init__(self, car, road, state_sources=['lane_direction', 'center_lane'], num_future_info=0):
        self.car = car
        self.road = road
        self.state_sources = state_sources
        self._sensors = []
        self.num_range_sensors = 0
        self.range_sensors = []
        range_sensor = list(filter(lambda x: bool(re.match('range_.*', x)), self.state_sources))
        if range_sensor:
            self.num_range_sensors = int(range_sensor[0].split('_')[-1])

            self.density = ANGLE_BETWEEN_SENSORS
            self.range_sensors = self.init_range_sensors()
            self._sensors.extend(self.range_sensors)

        if 'center_lane' in self.state_sources:
            self.center_lane_sensor = CenterLaneSensor(0, self.car, self.road)
            self._sensors.append(self.center_lane_sensor)

        if 'lane_curvature' in self.state_sources:
            self.lane_curvature_sensor = LaneCurvatureSensor(0, self.car, self.road)
            self._sensors.append(self.lane_curvature_sensor)

        if 'lane_direction' in self.state_sources:
            oriented_curve = self.contour_tracing()
            self.lane_direction_sensor = LaneDirectionSensor(0, self.car, self.road, oriented_curve, num_future_info)
            self._sensors.append(self.lane_direction_sensor)

    def __len__(self):
        return len(self.range_sensors)

    def contour_tracing(self):
        b = []
        self.mask = pygame.surfarray.array2d(self.road.center_lane)
        starting_pixel = list(zip(*np.where(self.mask != 0)))
        b_x = starting_pixel[0][0]
        b_y = starting_pixel[0][1]
        b.append(starting_pixel[0])

        b_x = starting_pixel[0][0]
        b_y = starting_pixel[0][1]-1

        d_x = b_x - b_x
        d_y = b_y - b_y

        neighbors = [(0,0),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1)]
        count = 1
        # contour = np.zeros((*img.shape, 3))
        while count <= 2:
            tmp = 0
            for i in range(9):
                if d_x == neighbors[i][0] and d_y == neighbors[i][1]:
                    tmp = i
                    break

            while True:
                if(tmp == 8):
                    tmp = 0

                c_x = b_x + neighbors[tmp+1][0]
                c_y = b_y + neighbors[tmp+1][1]
                if (self.mask[c_x][c_y]!=0):
                    b_x = c_x
                    b_y = c_y
                    temp = (b_x,b_y)

                    if (temp in b)==True:
                        count += 1
                    b.append((b_x,b_y))
                    d_x = prev_x-b_x
                    d_y = prev_y-b_y
                    break

                prev_x = c_x
                prev_y = c_y
                tmp += 1
        return np.array(b)
        # for pixel in range(len(B)):
        #       contour[B[pixel][0]][B[pixel][1]] = [0,255,0]

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
    def __init__(self, use_textures=False):
        self.img = pygame.image.load(TRACK_FILE)
        self.img = pygame.transform.smoothscale(self.img, (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.field_mask = pygame.mask.from_threshold(self.img, self.img.get_at((0,0)), (2, 2, 2, 255))
        self.img_rect = self.img.get_rect()

        # ==============================================
        # Load in different textures for the road
        # ==============================================
        self.texture_map = {}
        self.use_textures = use_textures

        if self.use_textures:
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
        self.center_lane_mask = pygame.mask.from_surface(self.center_lane)

    def blit(self, screen):
        screen.blit(self.center_lane, (0, 0))

        # Render track first
        screen.blit(self.img, (0, 0))

        # Render textures on top
        for texture, dic in self.texture_map.items():
            screen.blit(dic['texture'], (0, 0))
