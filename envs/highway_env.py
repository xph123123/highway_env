import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.road.lane import LineType, StraightLane, SineLane

# 0:original 1:static obstacle 2:two lane two car overtaking 3: two lane two car static 4: two lanes overtaking slow car but left car accerlerate
# 5：two lanes, keep in second lane and left car cut in 6：monte carlo random generate two lane two cars overtaking
SCENARIO_OPTION = 5

class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        if SCENARIO_OPTION == 0 or SCENARIO_OPTION == 1:
            """Create a road composed of straight adjacent lanes."""
            self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                             np_random=self.np_random, record_history=self.config["show_trajectories"])
        elif SCENARIO_OPTION == 2 or SCENARIO_OPTION == 3 or SCENARIO_OPTION == 4 or SCENARIO_OPTION == 5 or SCENARIO_OPTION == 6:
            # 2 straight line
            net = RoadNetwork()
            ends = [150,150]
            c,s,n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
            y = [0, 3.5]
            line_type = [[c,s],[n,c]]
            for i in range(2):
                net.add_lane("a", "b",StraightLane([0, y[i]], [sum(ends[:1]), y[i]], line_types=line_type[i]))
                net.add_lane("b", "c",StraightLane([sum(ends[:1]), y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))
            road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
            self.road = road

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        if SCENARIO_OPTION == 0:
            for others in other_per_controlled:
                controlled_vehicle = self.action_type.vehicle_class.create_random(
                    self.road,
                    speed=25,
                    lane_id=self.config["initial_lane_id"],
                    spacing=self.config["ego_spacing"]
                )
                self.controlled_vehicles.append(controlled_vehicle)
                self.road.vehicles.append(controlled_vehicle)

                for _ in range(others):
                    vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                    vehicle.randomize_behavior()
                    self.road.vehicles.append(vehicle)

        elif SCENARIO_OPTION == 1:
            for others in other_per_controlled:
                controlled_vehicle = self.action_type.vehicle_class.create_random(
                    self.road,
                    speed=0,
                    lane_id=3,
                    spacing=self.config["ego_spacing"]
                )
                self.controlled_vehicles.append(controlled_vehicle)
                self.road.vehicles.append(controlled_vehicle)
                for _ in range(1):
                    self.road.vehicles.append(
                        other_vehicles_type(self.road, [70, 12], speed=0.0, target_speed=0.0, enable_lane_change= False)
                        #other_vehicles_type.make_on_lane(cls, road: Road, lane_index: LaneIndex, longitudinal: float, speed: float = 0)
                    )
                    self.road.vehicles.append(
                        other_vehicles_type(self.road, [100, 4], speed=0.0, target_speed=0.0, enable_lane_change=False)
                        # other_vehicles_type.make_on_lane(cls, road: Road, lane_index: LaneIndex, longitudinal: float, speed: float = 0)
                    )

        elif SCENARIO_OPTION == 2:
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(
                self.road,
                lane_index=('a','b',1),
                longitudinal=2
            )
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)
            self.road.vehicles.append(
                other_vehicles_type(self.road, [20, 0], speed=10.0, target_speed=20.0, enable_lane_change= False)
                #other_vehicles_type.make_on_lane(cls, road: Road, lane_index: LaneIndex, longitudinal: float, speed: float = 0)
            )
            self.road.vehicles.append(
                other_vehicles_type(self.road, [100, 3.5], speed=10.0, target_speed=20.0, enable_lane_change= False, route=[('a','b',1),('b','c',0)])
                #other_vehicles_type.make_on_lane(cls, road: Road, lane_index: LaneIndex, longitudinal: float, speed: float = 0)
            )

        elif SCENARIO_OPTION == 3:
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(
                self.road,
                lane_index=('a', 'b', 0),
                longitudinal=2
            )
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)
            self.road.vehicles.append(
                other_vehicles_type(self.road, [60, 0], speed=0, target_speed=0, enable_lane_change=False)
                # other_vehicles_type.make_on_lane(cls, road: Road, lane_index: LaneIndex, longitudinal: float, speed: float = 0)
            )
            self.road.vehicles.append(
                other_vehicles_type(self.road, [120, 3.5], speed=0, target_speed=0, enable_lane_change=False,
                                    route=[('a', 'b', 1), ('b', 'c', 0)])
                # other_vehicles_type.make_on_lane(cls, road: Road, lane_index: LaneIndex, longitudinal: float, speed: float = 0)
            )

        elif SCENARIO_OPTION == 4:
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(
                self.road,
                lane_index=('a', 'b', 1),
                longitudinal=2
            )
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)
            self.road.vehicles.append(
                other_vehicles_type(self.road, [1, 0], speed=10, target_speed=30, enable_lane_change=False)
                # other_vehicles_type.make_on_lane(cls, road: Road, lane_index: LaneIndex, longitudinal: float, speed: float = 0)
            )
            self.road.vehicles.append(
                other_vehicles_type(self.road, [50, 3.5], speed=10, target_speed=15, enable_lane_change=False,
                                    route=[('a', 'b', 1), ('b', 'c', 1)])
                # other_vehicles_type.make_on_lane(cls, road: Road, lane_index: LaneIndex, longitudinal: float, speed: float = 0)
            )

        elif SCENARIO_OPTION == 5:
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(
                self.road,
                lane_index=('a', 'b', 1),
                longitudinal=35
            )
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

            self.road.vehicles.append(
                other_vehicles_type(self.road, [50, 0], speed=10, target_speed=20, enable_lane_change=False,
                                    route=[('a', 'b', 0), ('b', 'c', 1)])
                # other_vehicles_type.make_on_lane(cls, road: Road, lane_index: LaneIndex, longitudinal: float, speed: float = 0)
            )
        elif SCENARIO_OPTION == 6:
            d_array = [90,60,45]
            D_array = [70,50,35]
            delta_v2_v1_array = [-5,-8,-12]
            delta_v3_v1_array = [-1,0,10]
            v1_array = [20,25,30]

            random_d = random.randint(0,2)
            random_D = random.randint(0,2)
            random_delta_v2_v1 = random.randint(0,2)
            random_delta_v3_v1 = random.randint(0,2)
            random_v1 = random.randint(0,2)

            d = d_array[random_d]
            D = D_array[random_D]
            delta_v2_v1 = delta_v2_v1_array[random_delta_v2_v1]
            delta_v3_v1 = delta_v3_v1_array[random_delta_v3_v1]
            v1 = v1_array[random_v1]
            d2 = 90
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(
                self.road,
                lane_index=('a','b',1),
                longitudinal=d2 - D ,
                speed=v1
            )
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)
            self.road.vehicles.append(
                other_vehicles_type(self.road, [d2 - d, 0], speed=v1 + delta_v3_v1, target_speed=v1 + delta_v3_v1, enable_lane_change= False)
                #other_vehicles_type.make_on_lane(cls, road: Road, lane_index: LaneIndex, longitudinal: float, speed: float = 0)
            )
            self.road.vehicles.append(
                other_vehicles_type(self.road, [d2, 3.5], speed=v1 + delta_v2_v1, target_speed=v1 + delta_v2_v1, enable_lane_change= False, route=[('a','b',1),('b','c',1)])
                #other_vehicles_type.make_on_lane(cls, road: Road, lane_index: LaneIndex, longitudinal: float, speed: float = 0)
            )

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                          [self.config["collision_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 20,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False


register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)

register(
    id='highway-fast-v0',
    entry_point='highway_env.envs:HighwayEnvFast',
)
