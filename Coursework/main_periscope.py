# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 21:12:34 2020

@author: user
"""
import os
workdir="C:\\Users\\Nika\\Desktop\\Mag\\3\\Компьютерные сети\\coursework_project"
os.chdir(workdir)

from screen import Screen
from sensor import Sensor, DesignatedRouter
#import time
from multiprocessing import Queue, Array
from Receiver_and_Sender import Receiver, Sender, Transmitter, Result


import numpy as np
from helpers import (random_pos, random_vel, random_width, random_angle,
                     reflect_vector, cross, get_x, get_y, normalize, real_attr,
                     make_shared_arr, modify_shared_arr, array_from_shared,
                     shared_from_array, shared_from_double, double_from_shared)
from communication import (Connection, HelloMessage, AskEnergyMessage, RespondEnergyMessage,
                           StartCorrectionMessage, StopCorrectionMessage)

from matplotlib import pyplot as plt

from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager
import multiprocessing as mp
import ctypes

from time import time, sleep
import os

    

class Light:
    """
    Light source (Sun)
    """
    def __init__(self, pos, vel):
        self.pos = shared_from_array(pos) # pos
        self.vel = shared_from_array(vel) # vel
        self.start_time = shared_from_double(time())
        
        
    def update(self):
        # self.pos += self.vel * dt
        dt = time() - self.start_time
        real_attr(self, "start_time").value = time()

        if np.abs(get_x(self.pos)) > 10.0:
            def upd_vel(v):
                v[0] *= -1
            modify_shared_arr(upd_vel, real_attr(self, "vel"))
        
        def upd(pos):
            pos += self.vel * dt
        modify_shared_arr(upd, real_attr(self, "pos"))
        
    
    def __getattribute__(self, attr):
        if attr == "__dict__": # probably pickling in Windows uses __dict__
            return object.__getattribute__(self, attr)
        if callable(real_attr(self, attr)):
            return real_attr(self, attr)
        if type(real_attr(self, attr)) == float:
            return real_attr(self, attr)
        if type(real_attr(self, attr)) == mp.Value:
            return double_from_shared(real_attr(self, attr))
        
#         try:
#             res = array_from_shared(real_attr(self, attr))
#             return res
#         except Exception as ex:
#             print(ex)
#             print(f"Exception with attr = {attr}")
#             raise ex
        
        return array_from_shared(real_attr(self, attr))
    
    # def get_pos(self):
    #    return array_from_shared(self.pos)
        
        
    def draw(self):
        plt.scatter(get_x(self.pos), get_y(self.pos), c='y')

        
class Ray:
    def __init__(self, origin, direction, energy=1.0):
        object.__setattr__(self, "origin", shared_from_array(origin))
        object.__setattr__(self, "direction", shared_from_array(direction))
        self.energy = energy
        
    
    def __getattribute__(self, attr):
        if attr == "origin" or attr == "direction":
            return array_from_shared(real_attr(self, attr))
        return real_attr(self, attr)


    def __setattr__(self, attr, value):
        if attr == "origin" or attr == "direction":
            with real_attr(self, attr).get_lock(): # synchronize access
                arr = np.frombuffer(real_attr(self, attr).get_obj()) # no data copying
                arr[:] = value[:]
        else:
            object.__setattr__(self, attr, value)
    
    
    def intersect_horizontal(self, y):
        """
        Find x-coordinate of ray intersection with horizontal line
        Returns None if ray does not intersect the line
        """
        if get_y(self.direction) * (y - get_y(self.origin)) < 0:
            return None

        tg = get_x(self.direction) / get_y(self.direction)
        return get_x(self.origin) + tg * (y - get_y(self.origin))
        

class Agent:
    def __init__(self):
        self.in_conns = []
        self.out_conns = []


class Focus(Agent):
    def loop(self):
        total_time = self.total_time # 30
        start_time = time()
        snapshot_made = False
        
        timeout = 0.5
        attempts = 5
        # threshold = 2.0 * len(self.world.mirrors)
        threshold = 1000.0
        
        # all_mirrors_ids = list(range(len(self.world.mirrors)))
        all_mirrors_ids = [i for i, mirror in enumerate(self.world.mirrors) if not mirror.is_static]
        available_mirrors_ids = set(all_mirrors_ids)
        
        def get_next_correction_idx(ids):
            next_correction_idx = np.random.choice(list(ids))
            ids.remove(next_correction_idx)
            if len(ids) == 0:
                ids.update(all_mirrors_ids)
            
            return next_correction_idx
        
        last_correction_idx = get_next_correction_idx(available_mirrors_ids)
            
        
        for out_conn in self.out_conns:
            out_conn.send(HelloMessage(-1, out_conn.to_idx, data='I am a Focus'))
            out_conn.send(StartCorrectionMessage(-1, last_correction_idx, data=time()))
        
        cur_energy, _ = self.calc_energy()

        while attempts > 0:
            for in_conn in self.in_conns:
                if in_conn.poll(timeout / len(self.in_conns)):
                    msg = in_conn.recv()
                    # print(f"{self}: Got message: {msg}")

                    if type(msg) == AskEnergyMessage:
                        # send response
                        to_idx = msg.sender_idx
                        # print(f"SEND RESPONSE TO {to_idx}")
                        self.world.light.update() # update the light position
                        energy, _ = self.calc_energy()
                        if energy > threshold:
                            print(f"{self}: Energy > {threshold}")
                            return
                        
                        response = RespondEnergyMessage(sender_idx=self.idx, recepient_idx=to_idx,
                                                        data=energy)
                        for out_conn in self.out_conns:
                            out_conn.send(response)
                    elif type(msg) == StopCorrectionMessage:
                        # last_correction_idx = (last_correction_idx + 1) % len(self.world.mirrors)
                        new_energy, _ = self.calc_energy()
                        if new_energy <= cur_energy:
                            last_correction_idx = get_next_correction_idx(available_mirrors_ids)
                        cur_energy = new_energy
                        for out_conn in self.out_conns:
                            out_conn.send(StartCorrectionMessage(-1, last_correction_idx, data=time()))
                    break
            else:
                print(f"{self}: No input connections")
                attempts -= 1
                sleep(timeout)
            if time() - start_time > total_time:
                print(f"{self}: Done")
                return
            if not snapshot_made and time() - start_time > total_time / 2:
                snapshot_made = True
                print(f"{self}: Save plot")
                self.world.draw(title="after_half")
        
        print(f"{self}: Done")
    
    
    def __init__(self, pos, width, world=None, idx=-1, loss_function=None, total_time=30):
        super().__init__()
        self.pos = pos
        self.width = width
        self.halfwidth = width / 2.0
        self.world = world
        self.idx = idx
        self.loss_function = loss_function
        self.total_time = total_time
        
    
    def __repr__(self):
        return f"<#{self.idx} Focus>"
    
    
    def get_light(self):
        return self.world.light
    
    
    def get_mirrors(self):
        return self.world.mirrors
    
    
    def intersect(self, ray):
        return ray.intersect_horizontal(get_y(self.pos))
    
    
    def mirror_contribution(self, rays):
        intersections = list(filter(lambda x: x is not None, [self.intersect(ray) for ray in rays]))
        if len(intersections) <= 1:
            return 0

        left_pt  = np.min(intersections)
        right_pt = np.max(intersections)
        
        left  = np.maximum(left_pt, get_x(self.pos) - self.halfwidth)
        right = np.minimum(right_pt, get_x(self.pos) + self.halfwidth)
        
        # diff = right - left
        # diff = len(list(filter(lambda x: get_x(self.pos) - self.halfwidth <= x <= get_x(self.pos) + self.halfwidth, intersections)))
        diff = 0

        for ray in rays:
            x = self.intersect(ray)
            if x is None:
                continue

            if get_x(self.pos) - self.halfwidth <= x <= get_x(self.pos) + self.halfwidth:
                decay = 1.0 - np.abs(x - get_x(self.pos)) / self.halfwidth
                diff += ray.energy * decay
        
        return diff if diff > 0 else 0
    
    
    def calc_energy(self):
        if self.loss_function is not None:
            return self.loss_function(self)
        
        light = self.get_light()
        contribs = [self.mirror_contribution(mirror.get_rays(light))
                                             for mirror in self.get_mirrors()]
        return np.sum(contribs), [ray for mirror in self.get_mirrors() for ray in mirror.get_rays(light)]

    
    def draw(self):
        plt.scatter(get_x(self.pos), get_y(self.pos), c='r')
        
        left = self.pos.copy()
        right = self.pos.copy()
        left[0] -= self.halfwidth
        right[0] += self.halfwidth
        
        plt.plot([left[0], right[0]], [left[1], right[1]], c='r')

        energy, _ = self.calc_energy()
        plt.text(get_x(self.pos), get_y(self.pos) + 0.3, f"E={energy:.3f}")
 

class Mirror(Agent):
    @staticmethod
    def loop(mirror, focus):
        if mirror.is_static:
            print(f"{mirror}: Static mode")
            return

        print(f"{mirror}: Init")

        timeout = 0.1
        
        rot_dir = [1]
        def rotate_mirror(angle):
            delta = mirror.delta # 0.05
            left_bound = 0.0
            right_bound = 2.0 * np.pi
            
            angle += delta * rot_dir[0]
            if angle < left_bound:
                rot_dir[0] = 1
                angle = left_bound + delta
                return angle
            if angle > right_bound:
                rot_dir[0] = -1
                angle = right_bound - delta
            return angle
        
        def get_focus_energy():
            for out_conn in mirror.out_conns:
                out_conn.send(AskEnergyMessage(mirror.idx, -1))
            while True:
                for in_conn in mirror.in_conns:
                    if in_conn.poll(timeout / len(mirror.in_conns)):
                        msg = in_conn.recv()
                        if msg.recepient_idx == mirror.idx:
                            if type(msg) == RespondEnergyMessage:
                                # print(f"{mirror} Got energy {msg.data}")
                                return msg.data
                            else:
                                print(f"{mirror}: Got incorrect message = {msg}")
                        else:
                            # send message to the next agent
                            for out_conn in mirror.out_conns:
                                if out_conn.to_idx == in_conn.from_idx:
                                    continue # don't send messages back
                                # print(f"{mirror}: Resend message to {out_conn.to_idx} = {msg}")
                                out_conn.send(msg)
        
        def resend_messages():
            intercepted_energy = None
            # resend messages
            for in_conn in mirror.in_conns:
                if in_conn.poll(timeout / len(mirror.in_conns)):
                    msg = in_conn.recv()
                    if type(msg) == RespondEnergyMessage:
                        intercepted_energy = msg.data
                    if msg.recepient_idx == mirror.idx:
                        continue # ignore incoming messages
                    else:
                        # send message to the next agent
                        for out_conn in mirror.out_conns:
                            if out_conn.to_idx == in_conn.from_idx:
                                continue # don't send messages back
                            # print(f"{mirror}: Resend message to {out_conn.to_idx} = {msg}")
                            out_conn.send(msg)
            return intercepted_energy
        
        
        def send_start_correction_message(data):
            for out_conn in mirror.out_conns:
                out_conn.send(StartCorrectionMessage(mirror.idx, -1, data=data))
            return True
        
        def get_correction_message():
            for in_conn in mirror.in_conns:
                if in_conn.poll(timeout / len(mirror.in_conns)):
                    msg = in_conn.recv()
                    if type(msg) == StartCorrectionMessage:
                        if msg.sender_idx != mirror.idx:
                            return msg
                    return None
            return None
        
        def do_correction():
            """
            Returns if correction was successful
            """
            initial_energy = None
            end_energy = None
            
            intercepted_message = None
            for i in range(5):
                if intercepted_message is None:
                    prev_energy = get_focus_energy() # focus.calc_energy()
                else:
                    # don't send unnecessary request
                    prev_energy = intercepted_message
                
                if initial_energy is None:
                    initial_energy = prev_energy
                
                intercepted_message = None
                mirror.angle = rotate_mirror(mirror.angle)
                new_energy = get_focus_energy() # focus.calc_energy()
                if new_energy < prev_energy:
                    rot_dir[0] *= -1
                intercepted_message = resend_messages()
                
                end_energy = new_energy
                
            if end_energy > initial_energy:
                print(f"{mirror}: Successful correction from {initial_energy:.3f} to {end_energy:.3f}")
            else:
                print(f"{mirror}: No positive effect")
            
            return end_energy > initial_energy
                
        def send_stop_correction_message(corr_res=False):
            for out_conn in mirror.out_conns:
                if corr_res:
                    out_conn.send(StopCorrectionMessage(mirror.idx, -1, data=[mirror.pos, mirror.angle]))
                else:
                    out_conn.send(StopCorrectionMessage(mirror.idx, -1))
            return True
        
        currently_rotating = [False]
        def can_rotate():
            """
            Listen for rotation messages
            """
            # current_correction = get_correction_message()
            # return current_correction is None
            for in_conn in mirror.in_conns:
                if in_conn.poll(timeout / len(mirror.in_conns)):
                    msg = in_conn.recv()
                    if type(msg) == StartCorrectionMessage:
                        currently_rotating[0] = True
                        return False
                    if type(msg) == StopCorrectionMessage:
                        currently_rotating[0] = False
                        return True
            return not currently_rotating[0]
        
        MODE = 'adaptive' # 'random'
        if MODE == 'random':
            while True:
                do_correction()
                resend_messages()
            return
        
        while True:
            # listen for commands
            for in_conn in mirror.in_conns:
                if in_conn.poll(timeout / len(mirror.in_conns)):
                    msg = in_conn.recv()
                    if type(msg) == StartCorrectionMessage:
                        if msg.recepient_idx == mirror.idx:
                            # print(f"{mirror}: Do correction")
                            corr_res = do_correction()
                            send_stop_correction_message(corr_res)
                            break
                    if type(msg) == StopCorrectionMessage:
                        if msg.data is not None:
                            # print(f"{mirror}: Found successful correction")
                            pass
                    for out_conn in mirror.out_conns:
                        if out_conn.to_idx == in_conn.from_idx:
                            continue # don't send messages back
                        # print(f"{mirror}: Resend message to {out_conn.to_idx} = {msg}")
                        out_conn.send(msg)
                                
        
        while False:
            # listen for rotation messages
            if can_rotate():
                cur_time = time()
                send_start_correction_message(cur_time)
                
                give_up = False
                while True: # check all correction messages
                    other_rotation_message = get_correction_message()
                    if other_rotation_message is not None:
                        if other_rotation_message.data < cur_time:
                            resend_messages()
                            give_up = True# continue # give up
                            break
                    else:
                        break
                
                if give_up:
                    break
                
                send_start_correction_message(cur_time)
                # rotate
                do_correction()
                send_stop_correction_message()
                currently_rotating[0] = True

            resend_messages()
            

        
    def __repr__(self):
        return f"<#{self.idx}: angle={self.angle:.2f}; pos={self.pos}>"
        
    def __init__(self, pos, width, angle=0, idx=None, weight=1.0, delta=0.05, n_rays=50, is_static=False):
        super().__init__()
        object.__setattr__(self, "pos", shared_from_array(pos))
        self.width = width
        self.halfwidth = width / 2.0
        object.__setattr__(self, "angle", shared_from_double(angle))
        self.idx = idx
        self.weight = weight
        self.delta = delta
        self.n_rays = n_rays
        self.is_static = is_static
        
        
    def __getattribute__(self, attr):
        if attr == "pos":
            return array_from_shared(real_attr(self, attr))
        elif attr == "angle":
            return double_from_shared(real_attr(self, attr))
        else:
            return real_attr(self, attr)
        
    
    def __setattr__(self, attr, value):
        if attr == "pos":
            with real_attr(self, attr).get_lock(): # synchronize access
                arr = np.frombuffer(real_attr(self, attr).get_obj()) # no data copying
                arr[:] = value[:]
        elif attr == "angle":
            with real_attr(self, attr).get_lock(): # synchronize access
                real_attr(self, attr).value = value
        else:
            object.__setattr__(self, attr, value)
    
        
    def get_normal(self):
        return np.array([np.cos(self.angle), np.sin(self.angle)])
        
        
    def get_rays(self, light=None, ray=None):
        normal = self.get_normal()
        normal_cross = cross(normal)

        if light is not None:
            light_dir = light.pos - self.pos
            mirrored_light_dir = reflect_vector(light_dir, normal)
            mirrored_light = mirrored_light_dir + self.pos

            ray_beam = [] # ray cluster
            alpha_space = np.linspace(-1, 1, self.n_rays)
            for alpha in alpha_space:
                left_pos = self.pos + normal_cross * self.halfwidth * alpha
                left_ray = Ray(left_pos, left_pos - mirrored_light, self.weight)
                ray_beam.append(left_ray)

            # return [Ray(self.pos, normalize(-mirrored_light_dir))]
            # return [Ray(self.pos, normalize(-mirrored_light_dir)), left_ray, right_ray]
            return ray_beam
        elif ray is not None:
            relative_origin = ray.origin - self.pos
            mirrored_origin_dir = reflect_vector(relative_origin, normal)
            mirrored_origin = mirrored_origin_dir + self.pos # mirrored relative origin

            mirrored_direction = reflect_vector(ray.direction, normal)

            # shift origin along direction to the mirror
            p = self.pos - mirrored_origin
            n_dir = normalize(mirrored_direction)
            alpha = np.dot(normal, p) / np.dot(normal, n_dir)
            shifted_mirrored_origin = mirrored_origin + n_dir * alpha
            # return [Ray(mirrored_origin, mirrored_direction)]

            # check if origin is inside the mirror
            # if not, don't cast ray

            center_to_reflected_origin = shifted_mirrored_origin - self.pos
            if np.abs(np.dot(normal_cross, center_to_reflected_origin)) <= self.halfwidth:
                return [Ray(shifted_mirrored_origin, n_dir, ray.energy * self.weight)]
            else:
                return []
        else:
            raise ValueError("Please, specify at least one named argument")
            
        # first_direction = (-mirrored_light_dir) + normal_cross * self.halfwidth
        # first = Ray(self.pos + normal_cross * self.halfwidth, first_direction)
        # 
        # second_direction = (-mirrored_light_dir) - normal_cross * self.halfwidth
        # second = Ray(self.pos - normal_cross * self.halfwidth, second_direction)
        # 
        # return [first, second]
    
    
    def draw(self, world):
        plt.scatter(get_x(self.pos), get_y(self.pos), c='b', s=10)
        
        normal = self.get_normal()
        normal_cross = cross(normal)
        
        left = self.pos + normal_cross * self.halfwidth
        right = self.pos - normal_cross * self.halfwidth
        
        # print(f"left: {left}, right: {right}")
        plt.plot([left[0], right[0]], [left[1], right[1]], c='g' if not self.is_static else 'b') #, s=5)
        plt.text(get_x(self.pos), get_y(self.pos) - 0.05, f"w={self.weight}")
        
        # for ray in self.get_rays(world.light):
        #     origin = ray.origin
        #     to = origin + ray.direction
        #     plt.arrow(*(list(ray.origin) + list(ray.direction)), shape='full', linestyle='-', color='r', alpha=0.1)
        

class World:
    def __init__(self, light=None, focus=None, mirrors=None, plots_prefix="results/plots_weights/"):
        self.light = light
        self.focus = focus
        self.mirrors = mirrors
        
        if focus is not None:
            focus.world = self

        self.plots_prefix = plots_prefix


    def draw_rays(self):
        _, rays = self.focus.calc_energy()
        for ray in rays: 
            origin = ray.origin
            to = origin + ray.direction
            plt.arrow(*(list(ray.origin) + list(ray.direction)), shape='full', linestyle='-', color='r', alpha=getattr(ray, "alpha", 0.5))

            
    def draw(self, title="scheme"):
        plt.figure()
        self.light.draw()
        self.focus.draw()
        [mirror.draw(self) for mirror in self.mirrors]
        self.draw_rays()
        # print([mirror.angle for mirror in self.mirrors])
        # plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('square')
        plt.savefig(os.path.join(self.plots_prefix, f"{title}.png"), dpi=270)

                                             
def make_chain_connections(agents):
    prev = 0 # first is Focus
    
    # available = set(range(len(agents)))
    available = set([i for i, agent in enumerate(agents) if not getattr(agent, "is_static", False)])
    available.remove(prev)
    
    while len(available) > 0:
        lst = list(available)
        following = np.argmin([np.linalg.norm(agents[prev].pos - agents[i].pos)
                               for i in lst])
        agents[prev].following = agents[lst[following]]
        agents[lst[following]].prev = agents[prev]
        
        print(f"Wire {agents[prev]} <-> {agents[lst[following]]}")
        prev_to_following = Connection(agents[prev].idx, agents[lst[following]].idx)
        agents[prev].out_conns.append(prev_to_following.sender())
        agents[lst[following]].in_conns.append(prev_to_following.receiver())
        
        following_to_prev = Connection(agents[lst[following]].idx, agents[prev].idx)
        agents[prev].in_conns.append(following_to_prev.receiver())
        agents[lst[following]].out_conns.append(following_to_prev.sender())
        
        available -= set([lst[following], prev])
        prev = lst[following]
            

def flatten_rays(rays_groups):
    """
    Convert list [[ray, ...], ...] to [ray]
    """
    return [ray for group in rays_groups for ray in group]


def weighted_loss(foc):
    """
    Custom loss function with mirrors weights.
    Light ray goes through a mirror chain from mirror to mirror.
    
    Loss function should be defined at a top level in order to permit pickling (!)
    Returns a pair of calculated energy and a list of rays (for visualization)
    """
    light = foc.get_light()

    mirrors_iter = iter(foc.get_mirrors())
    first_mirror = next(mirrors_iter)
    outgoing_rays = first_mirror.get_rays(light=light)
    all_rays_groups = [outgoing_rays]

    for mirror in mirrors_iter:
        outgoing_rays = [ray for incoming_ray in outgoing_rays for ray in mirror.get_rays(ray=incoming_ray)]

        if len(outgoing_rays) == 0:
            return 0.0, flatten_rays(all_rays_groups)

        all_rays_groups.append(outgoing_rays)

        # for visualization: rescale rays to collide with mirror
        normal = mirror.get_normal()
        for ray in all_rays_groups[-2]:
            n_dir = normalize(ray.direction)
            p = mirror.pos - ray.origin
            alpha = np.dot(normal, p) / np.dot(normal, n_dir)
            new_direction = n_dir * alpha
            center_to_ray_end = new_direction + ray.origin - mirror.pos
            if np.abs(np.dot(cross(normal), center_to_ray_end)) <= mirror.halfwidth:
                ray.direction = new_direction
            else:
                ray.direction = n_dir
                ray.alpha = 0.05
    
    # contribs = [foc.mirror_contribution(mirror.get_rays(light)) * getattr(mirror, "weight", 1.0)
    #                                      for mirror in foc.get_mirrors()]
    energy = foc.mirror_contribution(all_rays_groups[-1])
    # print(f"Calculated energy = {energy}")
    return energy, flatten_rays(all_rays_groups)

            
def run_experiment(centre, n_mirrors=None):
    
    coords = lambda x, y: np.array([x, y])
    
    CONFIG = {
        "path_prefix": "results/plots_periscope/",

        # objects configuration
        "light_pos": coords(0.0, 10.0),
        "light_vel": coords(1e-1, 0.0),
        "mirrors_pos": [
            coords(0.0, 5.0),
            coords(5.0, 5.0),
            coords(5.0, 4.0),
            coords(2.5, 4.0),

            # coords(2.5, 7.0),
            # coords(7.0, 4.0),

            # coords(5.0, 5.0)
        ],
        "n_rays": 100, # 100, # number of test rays from the light
        # "width": 1.5,
        "width": 1.25,
        "angle_delta": 0.01, # rotation angle delta
        # "focus_pos": coords(5.0, 0.0),
        "focus_pos": coords(centre[0], centre[1]),

        "total_time": 20,
    }
    
    # light = Light(random_pos(kind='light'), random_vel())
    # mirrors = [Mirror(random_pos(kind='mirror'), random_width(), random_angle(), idx=i, weight=1.0)
    #            for i in range(n_mirrors)]
    # focus = Focus(random_pos(kind='focus'), random_width(), loss_function=weighted_loss)
    # world = World(light, focus, mirrors)
    
    light = Light(CONFIG["light_pos"], CONFIG["light_vel"])
    mirrors = [Mirror(pos, CONFIG["width"], random_angle(), idx=i, weight=0.9, delta=CONFIG["angle_delta"], n_rays=CONFIG["n_rays"])
               for i, pos in enumerate(CONFIG["mirrors_pos"])]
    focus = Focus(CONFIG["focus_pos"], CONFIG["width"], loss_function=weighted_loss, total_time=CONFIG["total_time"])
    world = World(light, focus, mirrors, plots_prefix=CONFIG["path_prefix"])

    
    for mirror in mirrors:
        # set initial angles
        to_light = normalize(light.pos - mirror.pos)
        to_focus = normalize(focus.pos - mirror.pos)
        normal = normalize(to_light + to_focus)
        mirror.angle = np.arctan2(get_y(normal), get_x(normal))

    # emulate spurious glow
    # mirrors[0].weight = -1.0

    # mirrors[0].angle = np.pi / 4 + 0.05
    # mirrors[1].angle = 5 * np.pi / 4 + 0.09

    # mirrors[0].angle = np.pi / 3 + 0.1
    # mirrors[1].angle = 3 * np.pi / 2
    # mirrors[2].angle = np.pi + 0.1

    mirrors[0].angle = np.pi / 4
    mirrors[1].angle = 5 * np.pi / 4
    mirrors[2].angle = 3 * np.pi / 4
    mirrors[3].angle = 7 * np.pi / 4

    mirrors[1].is_static = True
    mirrors[2].is_static = True

    # mirrors[1].halfwidth = 1.0
    # mirrors[2].halfwidth = 1.0
        
    
    world.draw(title="before")
    
    make_chain_connections([focus] + mirrors)
    
    processes = []
    for mirror in mirrors:
        p = Process(target=Mirror.loop, args=[mirror, focus])
        processes.append(p)
        
    for p in processes:
        p.start()
        
    focus.loop()
    # while True:
    #     if not any(p.is_alive() for p in processes):
    #         break
    #     print(f"Energy: {focus.calc_energy()}")
    
    for p in processes:
        p.terminate()
        p.join()
        
    world.draw(title="after")


if __name__ == "__main__":
    screen_centre = [3,1]
    screen_height = 2
    spot_diam = 1.1
    move_step = 0.2

    screen = Screen(screen_centre, screen_height, spot_diam, move_step)
    router = DesignatedRouter(screen_centre, screen_height)

    screen.randMoveSpot()
    router.setIntersectionsForSensors(*screen.intersectionsWithSensors())
    screen.makePlotSpotAndIntersections()
    router.processGetDataFromSensors(0, 2)
    new_centre = router.calculateNewCentre()
    print('Центр окружности: ', new_centre)

    n_mirrors = 5
    run_experiment(new_centre, n_mirrors=n_mirrors)


