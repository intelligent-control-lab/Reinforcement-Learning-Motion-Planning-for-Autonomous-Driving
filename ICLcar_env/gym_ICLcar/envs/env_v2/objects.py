import os
import re
import numpy as np
import pygame
import math
from .utils import *
from .env_configs import *
import scipy.spatial as sp
from collections import defaultdict as dd

# ==============================================
# Car Class
# ==============================================

class Car(pygame.sprite.Sprite):
    """
        Car Pygame sprite object.

        Parameters
        ----------
        init_angle: int
        init_x: int
        init_y: int
        fps: int
    """
    def __init__(self, init_angle, init_x, init_y, car_file, fps=60):
        self.image = pygame.image.load(car_file)
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
        self.Bw = 25                 # rotational friction coefficient

        pygame.sprite.Sprite.__init__(self)

    def reset(self):
        """
            Reset car member variables.
        """
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
    def to_screen_coords(self):
        """Convert pose of car from car from to pygame coordinates
        """
        return [self.xpos, self.ypos, self.theta_degrees]

    @property
    def center(self):
        """Compute direct center of egovehicle
        """
        return [self.xpos + self.img_rect.center[0], self.ypos + self.img_rect.center[1]]

    def set_friction(self, trans_coef, rot_coef):
        self.Bv = trans_coef
        self.Bw = rot_coef

    def step(self, action):
        """Take a step in the environment
        Parameters
        ----------
        action: numpy.array
        """
        # ================================
        # Unicycle Dynamics Model
        # ================================
        # dt = self.clock.tick_busy_loop(self.fps) / 1000.0 # in milliseconds
        dt = 0.033 # cancel the adaptive time interval settings staff

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

    def rotate(self):
        """Rotate an image while keeping its center and size
        """
        orig_rect = self.image.get_rect()
        rot_image = pygame.transform.rotate(self.image, self.theta_degrees)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image


    def blit(self, screen, action):
        """Render car in pygame display
        """
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
    def __init__(self, _id, car, road, oriented_curve, num_future_info=0):
        super(CenterLaneSensor, self).__init__(car, road)
        self.name = 'CenterLaneSensor' + str(_id)
        self.oriented_curve = oriented_curve
        self.win = 5
        self.num_future_info = num_future_info
        self.measure()

    def measure(self):
        car_center = np.array(self.car.center)[np.newaxis, :]
        coords = self.road.center_lane

        distances = sp.distance.cdist(car_center, coords)
        point = coords[distances.argmin()]
        dist = distances.min()
        import copy
        dist_unsign = copy.deepcopy(dist)

        screen_point = to_screen_coords(point)
        self.visual_closest_point = point

        # compute the relative pos of car wrt. lane
        # determine the car relative pos
        cross_screen_coord, indx = self.measure_car_relativepos_lane()
        if cross_screen_coord > 0: # car on the left lane
            dist *= -1

        # compute the coords of cloest point in car frame
        # compute the angle difference first wrt. car heading direction
        angle_diff_unsign, cross_car_coord = self.measure_angle_diff_unsign(indx)
        # compute the x and y pos in car frame
        if cross_car_coord > 0: # center in left of car
            ypos = dist_unsign*math.cos(angle_diff_unsign)
            xpos = -1*dist_unsign*math.sin(angle_diff_unsign)
        else: # center in the right of car
            ypos = dist_unsign*math.cos(angle_diff_unsign)
            xpos = dist_unsign*math.sin(angle_diff_unsign)
        center_pos_car_coord = [xpos, ypos]

        # closest distance
        self.closest_point = center_pos_car_coord

        # get the future lane pos information
        future_center_pos_car_coord = self.measure_future_lane_pos(indx)

        # measurement
        # self.measurement=[dist, *future_center_pos_car_coord]
        self.measurement = [dist, *center_pos_car_coord, *future_center_pos_car_coord]
        return center_pos_car_coord, dist

    def measure_future_lane_pos(self, indx):
        future_center_pos_car_coord = [] # in car frame

        # get car position and vector
        car_center = np.array(self.car.center)[np.newaxis, :]
        # car_screen_coord = to_screen_coords(car_center)
        # car vector
        car_theta_rads = self.car.theta
        car_vec = np.array([math.cos(car_theta_rads), math.sin(car_theta_rads)])
        # car vector normal direction vector
        u1 = car_vec / np.linalg.norm(car_vec)
        # Append the future lane center points information
        for i in range(self.num_future_info):
            # get the future point
            point = self.oriented_curve[wrap(self.oriented_curve, indx+i*5+1, 0)]
            # get distance
            dist_unsign = sp.distance.cdist(car_center, np.expand_dims(point, axis=0)).min()
            # get unsigned angle difference wrt. car
            angle_diff_unsign, cross_car_coord = self.measure_angle_diff_unsign(indx+i*5+1)
            # compute the x and y pos in car frame
            if cross_car_coord > 0: # center in left of car
                ypos = dist_unsign*math.cos(angle_diff_unsign)
                xpos = -1*dist_unsign*math.sin(angle_diff_unsign)
            else: # center in the right of car
                ypos = dist_unsign*math.cos(angle_diff_unsign)
                xpos = dist_unsign*math.sin(angle_diff_unsign)
            center_pos_car_coord = [xpos, ypos]
            future_center_pos_car_coord.extend(center_pos_car_coord)

        return future_center_pos_car_coord

    def measure_car_relativepos_lane(self):
        # find closest point on oriented curve
        car_center = np.array(self.car.center)[np.newaxis, :]
        distances = sp.distance.cdist(car_center, self.oriented_curve).squeeze(0)
        indx = distances.argmin()
        # get closest point coordinates
        p1 = self.oriented_curve[wrap(self.oriented_curve, indx, -self.win)]
        p2 = self.oriented_curve[wrap(self.oriented_curve, indx, self.win)]
        # to world coordinates
        pos1 = to_screen_coords(p1)
        pos2 = to_screen_coords(p2)
        # road vec and center2car vec
        center2car = to_screen_coords(car_center.squeeze()) - to_screen_coords(self.oriented_curve[wrap(self.oriented_curve, indx, 0)])
        road_vec = pos2 - pos1
        # compute cross product of road vec and center2car vec
        cross = road_vec[0]*center2car[1] - center2car[0]*road_vec[1]
        # cross > 0: car is on left, cross < 0: car is on right
        return cross, indx

    def measure_angle_diff_unsign(self, indx):
        # cloest center of lane lane direction vector
        car_center = np.array(self.car.center)[np.newaxis, :]

        # car to lane center vector
        car2center = to_screen_coords(self.oriented_curve[wrap(self.oriented_curve, indx, 0)]) - to_screen_coords(car_center.squeeze())

        # car vector
        car_theta_rads = self.car.theta
        car_vec = np.array([math.cos(car_theta_rads), math.sin(car_theta_rads)])

        # compute angle between two vectors
        u1 = car_vec / np.linalg.norm(car_vec)
        u2 = car2center / np.linalg.norm(car2center)
        angle_diff = np.arccos(np.dot(u2, u1)) # in radians

        # compute left/right of center point in car frame
        # cross < 0: center in right, corss > 0: center in left
        cross = u1[0]*u2[1] - u1[1]*u2[0]

        return angle_diff, cross

    @property
    def text(self): return f"P: {self.closest_point[0]:.2f} {self.closest_point[1]:.2f}, D: {round(self.measurement[0], 3)}"


    def blit(self, screen):
        pygame.draw.circle(screen, BLUE, tuple(self.visual_closest_point), 10)


# ==============================================
# Sensor for computing lane direction
# ==============================================
class LaneDirectionSensor(Sensor):
    def __init__(self, _id, car, road, oriented_curve, num_future_info=0):
        super().__init__(car, road)
        self.name = 'LaneDirectionSensor'
        self.oriented_curve = road.center_lane
        self.num_future_info = num_future_info
        self.win = 5
        self.relative_pos = 1 # relative position from car to road, 1 for right of lane, -1 for left
        self.measure()

    def measure(self):
        lane_dir, indx, cross = self.measure_lane_direction()

        # update the relative position between car and lane
        if cross > 0:
            self.relative_pos = -1 # car on left indicator
        else:
            self.relative_pos = 1 # car on right indicator

        angle_diff = self.measure_angle_diff()

        self.measurement = [angle_diff]

        if self.num_future_info > 0:
            self.measurement.extend(self.add_future_info(indx))

        return self.measurement

    def measure_lane_direction(self):
        car_center = np.array(self.car.center)[np.newaxis, :]
        distances = sp.distance.cdist(car_center, self.oriented_curve).squeeze(0)
        indx = distances.argmin()

        p1 = self.oriented_curve[wrap(self.oriented_curve, indx, -self.win)]
        p2 = self.oriented_curve[wrap(self.oriented_curve, indx, self.win)]

        self.p1 = p1
        self.p2 = p2

        pos1 = to_screen_coords(p1)
        pos2 = to_screen_coords(p2)
        road_dir_angle = vec2angle(pos2 - pos1)

        center2car = to_screen_coords(car_center.squeeze()) - to_screen_coords(self.oriented_curve[indx])
        road_vec = pos2 - pos1
        # compute cross product of road vec and center2car vec
        cross = road_vec[0]*center2car[1] - center2car[0]*road_vec[1]

        return road_dir_angle, indx, cross

    def measure_angle_diff(self):
        car_theta_rads = self.car.theta
        car_vec = np.array([math.cos(car_theta_rads), math.sin(car_theta_rads)])
        import copy
        p1, p2 = copy.deepcopy(self.p1), copy.deepcopy(self.p2)
        p2[1] = SCREEN_HEIGHT - p2[1]
        p1[1] = SCREEN_HEIGHT - p1[1]
        road_vec = np.array(p2 - p1)

        # compute angle difference between two vectors
        u1 = car_vec / np.linalg.norm(car_vec)
        u2 = road_vec / np.linalg.norm(road_vec)
        angle_diff = np.arccos(np.dot(u1, u2)) # in radians

        # compute the signed angle difference
        cross = u1[0]*u2[1] - u1[1]*u2[0] # u1 cross u2
        if cross < 0:
            angle_diff *= -1

        return angle_diff

    def add_future_info(self, indx):
        future_dir_angle = [] # in car frame

        # car vector
        car_theta_rads = self.car.theta
        car_vec = np.array([math.cos(car_theta_rads), math.sin(car_theta_rads)])
        # car vector normal direction vector
        u1 = car_vec / np.linalg.norm(car_vec)
        # Append the future lane curvature information
        for i in range(self.num_future_info):
            p1 = self.oriented_curve[wrap(self.oriented_curve, indx+i*5+1, -self.win)]
            p2 = self.oriented_curve[wrap(self.oriented_curve, indx+i*5+1, self.win)]
            pos1 = to_screen_coords(p1)
            pos2 = to_screen_coords(p2)
            # road vector
            road_vec_tmp = np.array(pos2 - pos1)
            # compute angle difference between two vectors
            u2 = road_vec_tmp / np.linalg.norm(road_vec_tmp)
            angle_diff_tmp = np.arccos(np.dot(u1, u2)) # in radians
            # compute the signed angle difference
            cross = u1[0]*u2[1] - u1[1]*u2[0] # u1 cross u2
            if cross < 0: # negative means road dir on the right half plane (clockwise)
                angle_diff_tmp *= -1

            future_dir_angle.append(angle_diff_tmp)

        return future_dir_angle

    def blit(self, screen):
        pygame.draw.circle(screen, BLACK, tuple(self.p1), 2)
        pygame.draw.circle(screen, RED, tuple(self.p2), 2)

    @property
    def text(self):
        angle_diff = self.measurement[0]

        side = "left" if self.relative_pos < 0 else "right"
        return f"angle_diff: {angle_diff:.2f}, side: {side}"


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

        oriented_curve = self.road.center_lane

        if range_sensor:
            self.num_range_sensors = int(range_sensor[0].split('_')[-1])

            self.density = ANGLE_BETWEEN_SENSORS
            self.range_sensors = self.init_range_sensors()
            self._sensors.extend(self.range_sensors)

        if 'center_lane' in self.state_sources:
            self.center_lane_sensor = CenterLaneSensor(0, self.car, self.road, oriented_curve, num_future_info)
            self._sensors.append(self.center_lane_sensor)

        if 'lane_curvature' in self.state_sources:
            self.lane_curvature_sensor = LaneCurvatureSensor(0, self.car, self.road)
            self._sensors.append(self.lane_curvature_sensor)

        if 'lane_direction' in self.state_sources:
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
    def __init__(self,
        center_lane,
        textures,
        texture_frictions,
        road_sections,
        road_types
    ):
        self.center_lane = center_lane
        self.textures = textures
        self.texture_frictions = texture_frictions
        self.texture_colors = [(173, 216, 230), (181, 101, 29), (211, 211, 211)]
        self.texture_metadata = {
            t: {'friction': fric, 'color': c} for (t, fric, c) in zip(textures, texture_frictions, self.texture_colors)
        }
        self.road_sections = road_sections
        self.road_types = road_types

    def setup_texture_map(self, screen):
        self.texture_map = {k: dd(list) for k in self.textures}

        for texture in self.textures:
            for i, indx in enumerate(self.road_types):
                if self.textures[indx] == texture:
                    point = self.road_sections[i][len(self.road_sections[indx])//2]
                    point = [int(point[0]), int(point[1])]
                    # import ipdb; ipdb.set_trace()
                    texture_mask = pygame.mask.from_threshold(screen, self.texture_metadata[texture]['color'], (2,2,2,255))

                    self.texture_map[texture]['mask'].append(texture_mask)

    def blit(self, screen):
        radius = 40
        width = 600
        height = 400
        for i, section in enumerate(self.road_sections):
            info = self.texture_metadata[self.textures[self.road_types[i]]]
            # import ipdb; ipdb.set_trace()
            # trace the point to construct different polygon to represent the track 
            # for pointidx in np.arange(section.shape[0])[:-radius//3][::radius//3]:

            # ====================
            # newest version 
            # ====================
            for pointidx in np.arange(section.shape[0])[:-1]:
                tmp_idx = pointidx
                next_idx = pointidx + 1
                assert(next_idx < section.shape[0])
                point = section[tmp_idx,:]
                next_point = section[next_idx,:]
                # compute the polygon 
                xcoord0, ycoord0 = point[0], point[1]
                xcoord1, ycoord1 = next_point[0], next_point[1]

                vector = np.array([xcoord1 - xcoord0, ycoord1 - ycoord0])
                theta = np.arctan2(vector[1], vector[0])
                normal = theta + np.pi/2
                from numpy import cos,sin
                cp1 = np.array([xcoord0 + radius*cos(normal), ycoord0 + radius*sin(normal)])
                cp2 = np.array([xcoord0 - radius*cos(normal), ycoord0 - radius*sin(normal)])
                cp3 = np.array([xcoord1 + radius*cos(normal), ycoord1 + radius*sin(normal)])
                cp4 = np.array([xcoord1 - radius*cos(normal), ycoord1 - radius*sin(normal)])

                # first fill the crack between the end of last and begin of current 
                if pointidx >= 1 or i >=1:
                    precp1 = lastending["cp1"]
                    precp2 = lastending["cp2"]
                    pygame.draw.polygon(screen, info['color'], ((precp1[0],precp1[1]),(precp2[0],precp2[1]),(cp2[0],cp2[1]),(cp1[0],cp1[1]))) 

                # preserve the previous thing 
                lastending = {"cp1": cp3, "cp2": cp4}
                # draw the polygon 
                pygame.draw.polygon(screen, info['color'], ((cp1[0],cp1[1]),(cp2[0],cp2[1]),(cp4[0],cp4[1]),(cp3[0],cp3[1])))

           