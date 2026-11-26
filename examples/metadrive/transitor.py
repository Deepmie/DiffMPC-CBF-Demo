from metadrive.envs import MetaDriveEnv
from metadrive.component.vehicle.vehicle_type import BaseVehicle, DefaultVehicle
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.component.lane.pg_lane import PGLane
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.geometry.base import BaseGeometry
from typing import List, Tuple, Union
import numpy as np
from numpy import ndarray

class Transitor:
    def __init__(
            self,
            env: MetaDriveEnv,
            T: int,
            delta_t: float,
        ):
        self._env = env
        self._ego: DefaultVehicle = env.agent
        self._traffic: List[BaseVehicle] = list(env.engine.traffic_manager.traffic_vehicles)
        self._T = T
        self._delta_t = delta_t

    def _pavement(self) -> BaseGeometry:
        lanes = []
        for ends in self._env.engine.current_map.road_network.graph.values():
            for parallel in ends.values():
                lanes.extend(parallel)
        return unary_union([lane.shapely_polygon for lane in lanes]).buffer(0.2)

    def _clearance(
            self,
            pref: ndarray,
            psiref: ndarray,
            pavement: BaseGeometry,
            lane_half_width: float,
            step=0.25,
            max_dist=40.0
        ) -> Tuple[ndarray, ndarray]:
        nleft = np.array([-np.sin(psiref), np.cos(psiref)])
        def reach(direction: ndarray) -> ndarray:
            dist = 0.0
            while dist + step <= max_dist:
                q = pref + (dist + step) * direction
                if not pavement.covers(Point(float(q[0]), float(q[1]))):
                    break
                dist += step
            return dist
        return max(reach(nleft), lane_half_width), max(reach(-nleft), lane_half_width)

    def _route_lanes(self, vehicle: BaseVehicle) -> List[PGLane]:
        navi: NodeNetworkNavigation = vehicle.navigation
        network: NodeRoadNetwork = navi.map.road_network
        nodes = navi.checkpoints
        lane: PGLane = vehicle.lane
        start_i = navi._target_checkpoints_index[0]
        for i in range(len(nodes) - 1):
            if nodes[i] == lane.index[0] and nodes[i+1] == lane.index[1]:
                start_i = i
                break
        lanes: List[PGLane] = []
        prev = None
        for i in range(start_i, len(nodes) - 1):
            parallel = network.graph[nodes[i]][nodes[i + 1]]
            if prev is None:
                chosen = lane if lane in parallel else parallel[min(lane.index[-1], len(parallel) - 1)]
            else:
                linked = [c for c in parallel if prev.is_previous_lane_of(c)]
                same = [c for c in linked if c.index[-1] == prev.index[-1]]
                chosen = same[0] if same else (linked[0] if linked else parallel[min(prev.index[-1], len(parallel) - 1)])
            lanes.append(chosen)
            prev = chosen
        return lanes

    def _path_planning(self, vref: float) -> Tuple[ndarray, ndarray]:
        lanes = self._route_lanes(self._ego)
        pavement = self._pavement()
        s, _    = lanes[0].local_coordinates(self._ego.position)
        lane_i  = 0
        pref    = np.zeros([self._T + 1, 2])
        psiref  = np.zeros([self._T + 1, 1])
        nleft  = np.zeros([self._T + 1, 1])
        nright = np.zeros([self._T + 1, 1])
        for i in range(self._T + 1):
            lane = lanes[lane_i]
            while s > lane.length and lane_i + 1 < len(lanes):
                s -= lane.length
                lane_i += 1
                lane = lanes[lane_i]
            s_clip = min(max(s, 0.0), lane.length)
            pref[i] = lane.position(s_clip, 0.0)
            psiref[i, 0] = lane.heading_theta_at(s_clip)
            nleft[i, 0], nright[i, 0] = self._clearance(
                pref[i], psiref[i, 0], pavement, lane.width_at(s_clip) / 2.0,
            )
            s += vref * self._delta_t
        return pref, psiref, nleft, nright

    def _predict_traffic_info(self, nobs: int) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
        self._traffic.sort(key=lambda v: np.linalg.norm(np.asarray(v.position) - np.asarray(self.ego.position)))
        chosen = self._traffic[:nobs]

        # default value far from ego
        pos     = np.empty([self._T + 1, nobs, 2])
        pos[:]  = np.asarray(self._ego.position) + np.array([1.0e4, 0.0])
        heading = np.zeros([self._T + 1, nobs])
        length  = np.ones([nobs])
        width   = np.ones([nobs])
        mask    = np.zeros([nobs])
        
        for j, v in enumerate(chosen):
            length[j] = float(v.LENGTH); width[j] = float(v.WIDTH); mask[j] = 1.0
            # v's attr always None
            if v.lane is None or v.navigation is None:
                p = np.asarray(v.position, dtype=float)
                vel = np.asarray(v.velocity, dtype=float)
                for k in range(self._T + 1):
                    pos[k, j] = p + vel * (k * self._delta_t)
                continue
            lanes = self._route_lanes(v)
            long, lat = lanes[0].local_coordinates(v.position)
            for k in range(self._T + 1):
                s = long + float(v.speed) * k * self._delta_t
                lane_i = 0
                lane = lanes[lane_i]
                while s > lane.length and lane_i + 1 < len(lanes):
                    s -= lane.length
                    lane_i += 1
                    lane = lanes[lane_i]
                s_clip = min(max(s, 0.0), lane.length)
                pos[k, j] = lane.position(s_clip, lat)
                heading[k, j] = lane.heading_theta_at(s_clip)
        return pos, heading, length, width, mask