from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random
import math
import numpy as np
import pandas as pd
import os
_PLAYER_HOSTILE = 4

ACTION_DO_NOTHING = 'donothing'

ACTION_BUILD_SPAWNINGPOOL = 'buildspawningpool'
ACTION_BUILD_ROACH_WARREN = 'buildroachwarren'
ACTION_BUILD_EXTRACTOR = 'buildextractor'
ACTION_BUILD_HATCHERY = 'buildhatchery'
ACTION_TRAIN_ZERGLING = 'trainzergling'
ACTION_TRAIN_ROACH = 'trainroach'
ACTION_TRAIN_OVERLORD = 'trainoverlord'
ACTION_DRONE_HGAS = 'harvestgas'
ACTION_DRONE_HARVEST = 'droneharvest'
ACTION_ATTACK = 'attack'
ACTION_TRAIN_DRONE = 'traindrone'

DATA_FILE = 'sparse_agent_data'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_DRONE_HARVEST,
    ACTION_DRONE_HGAS,
    ACTION_TRAIN_OVERLORD,
    ACTION_TRAIN_DRONE,
    ACTION_TRAIN_ZERGLING,
    ACTION_TRAIN_ROACH,
    ACTION_BUILD_SPAWNINGPOOL,
    ACTION_BUILD_ROACH_WARREN,
    ACTION_BUILD_EXTRACTOR,
]

for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 2) % 8 == 0 and (mm_y + 2) % 8 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 4) + '_' + str(mm_y - 4))

for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 6) % 4 == 0 and (mm_y + 6) % 4 == 0:
            smart_actions.append(ACTION_BUILD_HATCHERY + '_' + str(mm_x - 2) + '_' + str(mm_y - 2))


class ZergAgentAttack(base_agent.BaseAgent):
    def __init__(self):
        super(ZergAgentAttack, self).__init__()
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        self.previous_action = None
        self.previous_state = None
        self.htc_x = None
        self.htc_y = None
        self.player_x = None
        self.player_y = None
        self.move_number = 0
        self.build_here = []
        self.harvest_here = []
        self.select=[]
        self.should_select=False
        self.should_harvest = False
        self.geyser_taken = False
        self.base_top_left = None
        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')
            self.qlearn.q_table.to_csv(DATA_FILE + '.csv')

    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        return [x + x_distance, y + y_distance]

    def transformLocation(self, x, y):
        if not self.base_top_left:
            return [64 - x, 64 - y]
        return [x, y]

    def get_geysers(self, obs):
        geysers = self.get_units_by_type(obs, units.Neutral.VespeneGeyser)
        if not geysers:
            geysers = self.get_units_by_type(obs, units.Neutral.ProtossVespeneGeyser)
            if not geysers:
                geysers = self.get_units_by_type(obs, units.Neutral.ShakurasVespeneGeyser)
                if not geysers:
                    geysers = self.get_units_by_type(obs, units.Neutral.SpacePlatformGeyser)
                    if not geysers:
                        geysers = self.get_units_by_type(obs, units.Neutral.PurifierVespeneGeyser)
                        if not geysers:
                            geysers = self.get_units_by_type(obs, units.Neutral.RichVespeneGeyser)
        return geysers

    def calculateDistance(self, x1, y1, x2, y2):
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist

    def splitAction(self, action_id):
        smart_action = smart_actions[action_id]
        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')
        return (smart_action, x, y)

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

    def unit_type_is_selected(self, obs, unit_type):
        if len(obs.observation.single_select) > 0 and obs.observation.single_select[0].unit_type == unit_type:
            return True
        if len(obs.observation.multi_select) > 0 and obs.observation.multi_select[0].unit_type == unit_type:
            return True
        return False

    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def step(self, obs):
        super(ZergAgentAttack, self).step(obs)
        if obs.last():
            reward = obs.reward
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')
            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
            self.previous_action = None
            self.previous_state = None
            self.move_number = 0
            return actions.FUNCTIONS.no_op()

        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                                  features.PlayerRelative.SELF).nonzero()
            self.player_x = player_x.mean()
            self.player_y = player_y.mean()
            xmean = player_x.mean()
            ymean = player_y.mean()
            self.base_top_left = 1 if player_y.any() and ymean <= 31 else 0

            self.htc_y = self.get_units_by_type(obs, units.Zerg.Hatchery)
            self.htc_y = self.htc_y[0].y
            self.htc_x = self.get_units_by_type(obs, units.Zerg.Hatchery)
            self.htc_x = self.htc_x[0].x

        hatcheries = self.get_units_by_type(obs, units.Zerg.Hatchery)
        htc_count = len(hatcheries)
        overlords = self.get_units_by_type(obs, units.Zerg.Overlord)
        overlords_count = len(overlords)
        spawning_pools = self.get_units_by_type(obs, units.Zerg.SpawningPool)
        spawning_pools_count = len(spawning_pools)
        supply_limit = obs.observation.player.food_cap
        supply_used = obs.observation.player.food_used
        worker_supply = obs.observation.player.food_workers
        army_limit = obs.observation['player'][5]
        roach_warren = self.get_units_by_type(obs, units.Zerg.RoachWarren)
        roach_warren_count = len(roach_warren)
        extractors = self.get_units_by_type(obs, units.Zerg.Extractor)
        extractor_count = len(extractors)
        free_supply = obs.observation.player.food_cap - obs.observation.player.food_used
        army_supply = obs.observation.player.food_army
        if self.move_number == 0:
            self.move_number += 1
            current_state = np.zeros(388)
            current_state[0] = overlords_count
            current_state[1] = spawning_pools_count
            current_state[2] = supply_limit
            current_state[3] = army_limit

            hot_squares = np.zeros(64)
            enemy_y, enemy_x = (obs.observation.feature_minimap.player_relative == _PLAYER_HOSTILE).nonzero()
            for i in range(0, len(enemy_y)):
                y = int(math.ceil((enemy_y[i] + 1) / 8))
                x = int(math.ceil((enemy_x[i] + 1) / 8))

                hot_squares[((y - 1) * 8) + (x - 1)] = 1

            if not self.base_top_left:
                hot_squares = hot_squares[::-1]

            for i in range(0, 64):
                current_state[i + 8] = hot_squares[i]

            green_squares = np.zeros(64)
            friendly_y, friendly_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF)\
                .nonzero()
            for i in range(0, len(friendly_y)):
                y = int(math.ceil((friendly_y[i] + 1) / 8))
                x = int(math.ceil((friendly_x[i] + 1) / 8))

                green_squares[((y - 1) * 8) + (x - 1)] = 1

            if not self.base_top_left:
                green_squares = green_squares[::-1]

            for i in range(0, 64):
                current_state[i + 8] = green_squares[i]
            if self.previous_action is not None:
                self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))

            excluded_actions = []

            if spawning_pools_count == 0:  # cant make Zerglings so dont try
                excluded_actions.append(5)

            if spawning_pools_count == 1:  # we have a Spawning pool, dont build another
                excluded_actions.append(7)

            if free_supply > 8:  # got enough supply dont bother with overlord
                excluded_actions.append(3)
            if worker_supply ==0 : # no point building with no drones
                for action in smart_actions:
                    if 'build' in action or 'drone' in action:
                        excluded_actions.append(smart_actions.index(action))

            if extractor_count >= (htc_count * 2): # if all extractors are made no point building new ones
                excluded_actions.append(9)
            if roach_warren_count == 1:
                excluded_actions.append(8)  # we have a Roach warren, dont build another
            if roach_warren_count == 0:  # cant make Roaches so dont try
                excluded_actions.append(6)
            if army_supply == 0:  # no supply so no attacking
                for action in smart_actions:
                    if ACTION_ATTACK in action:
                        excluded_actions.append(smart_actions.index(action))

            rl_action = self.qlearn.choose_action(str(current_state), excluded_actions)

            self.previous_state = current_state
            self.previous_action = rl_action

            smart_action, x, y = self.splitAction(self.previous_action)
            if smart_action == ACTION_DO_NOTHING:
                return actions.FUNCTIONS.no_op()

            elif smart_action == ACTION_BUILD_SPAWNINGPOOL or smart_action == ACTION_BUILD_EXTRACTOR or \
                    smart_action == ACTION_BUILD_ROACH_WARREN or smart_action == ACTION_BUILD_HATCHERY\
                    or smart_action == ACTION_DRONE_HARVEST or smart_action == ACTION_DRONE_HGAS:
                    # builds spawn pool stage 1 : go to base
                    self.select = [self.player_x, self.player_y]
                    self.should_select = True
                    return actions.FUNCTIONS.move_camera(self.select)

            elif smart_action == ACTION_TRAIN_OVERLORD or smart_action == ACTION_TRAIN_ZERGLING or \
                    smart_action == ACTION_TRAIN_DRONE or smart_action == ACTION_TRAIN_ROACH:
                # train troop/drone stage 1 : go to base
                if not self.unit_type_is_selected(obs, units.Zerg.Larva):
                        self.should_select = True
                        self.select = [self.player_x, self.player_y]
                        return actions.FUNCTIONS.move_camera(self.select)

            elif smart_action == ACTION_ATTACK:
                if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                    return actions.FUNCTIONS.select_army("select")

        elif self.move_number == 1:
            self.move_number += 1

            smart_action, x, y = self.splitAction(self.previous_action)

            if smart_action == ACTION_BUILD_SPAWNINGPOOL or smart_action == ACTION_BUILD_EXTRACTOR or \
                    smart_action == ACTION_BUILD_ROACH_WARREN or smart_action == ACTION_BUILD_HATCHERY \
                    or smart_action == ACTION_DRONE_HARVEST or smart_action == ACTION_DRONE_HGAS:
                if self.should_select:
                    idleness = actions.FUNCTIONS.select_idle_worker()
                    if len(idleness)>0:
                        print('yup')
                        drone = idleness[0]
                    else:
                        drones = self.get_units_by_type(obs, units.Zerg.Drone)
                        if len(drones)>0:
                            drone = random.choice(drones)
                    drone_x = drone.x
                    drone_y = drone.y
                    self.select = [drone_x, drone_y]
                    return actions.FUNCTIONS.select_point("select", self.select)

            elif smart_action == ACTION_TRAIN_OVERLORD or smart_action == ACTION_TRAIN_ZERGLING or \
                    smart_action == ACTION_TRAIN_DRONE or smart_action == ACTION_TRAIN_ROACH:
                # train troop/drone stage 1 : get larvae
                if self.should_select:
                    larvae = self.get_units_by_type(obs, units.Zerg.Larva)
                    if len(larvae) > 0:
                        larva = random.choice(larvae)
                        self.select = [larva.x, larva.y]
                    return actions.FUNCTIONS.select_point("select_all_type", self.select)

            elif smart_action == ACTION_ATTACK:
                do_it = True
                if self.unit_type_is_selected(obs, units.Zerg.Drone):
                    do_it = False
                if do_it and self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                    target_location = self.transformLocation(int(x), int(y))
                    return actions.FUNCTIONS.Attack_minimap('now', target_location)

        elif self.move_number == 2:
            self.move_number += 1

            smart_action, x, y = self.splitAction(self.previous_action)

            if smart_action == ACTION_BUILD_HATCHERY:
                if self.can_do(obs, actions.FUNCTIONS.Build_Hatchery_screen.id):
                    self.build_here = [int(x), int(y)]
                    return actions.FUNCTIONS.move_camera(self.build_here)

            elif smart_action == ACTION_BUILD_SPAWNINGPOOL or smart_action == ACTION_BUILD_ROACH_WARREN:
                if self.get_units_by_type(obs, units.Zerg.Hatchery):
                        self.build_here = self.transformDistance(self.htc_x, 15, self.htc_y, -9)
                        return actions.FUNCTIONS.move_camera(self.build_here)

            elif smart_action == ACTION_BUILD_EXTRACTOR:
                if self.can_do(obs, actions.FUNCTIONS.Build_Extractor_screen.id):
                    geysers = self.get_geysers(obs)
                    if geysers:
                        for geyser in geysers:  # for each geyser
                            gey_x = geyser.x
                            gey_y = geyser.y
                            self.geyser_taken = False
                            for hatch in hatcheries:  # for each hatch
                                htc_x = hatch.x
                                htc_y = hatch.y
                                if self.calculateDistance(gey_x, gey_y, htc_x, htc_y) <= 35:
                                    # is idea anywhere near a hatch?
                                    extractors = self.get_units_by_type(obs, units.Zerg.Extractor)
                                    if extractors:
                                        for extractor in extractors:
                                            extc_x = extractor.x
                                            extc_y = extractor.y
                                            if gey_x == extc_x and gey_y == extc_y:
                                                self.geyser_taken = True
                                            # does location not have an extractor already?
                                            # no? YAY build it!.
                                    if not self.geyser_taken:
                                        self.build_here = [gey_x, gey_y]
                                        return actions.FUNCTIONS.move_camera(self.build_here)

            elif smart_action == ACTION_DRONE_HARVEST:
                if self.can_do(obs, actions.FUNCTIONS.Harvest_Gather_screen.id):
                    minerals = self.get_units_by_type(obs, units.Neutral.MineralField)
                    for mineral in minerals:
                        hatcheries = self.get_units_by_type(obs, units.Zerg.Hatchery)
                        for hatch in hatcheries:  # for each hatch
                            self.should_harvest = False
                            if self.calculateDistance(mineral.x, mineral.y, hatch.x, hatch.y) <= 35:
                                self.should_harvest = True
                                self.harvest_here = [mineral.x, mineral.y]
                                return actions.FUNCTIONS.move_camera(self.harvest_here)

            elif smart_action == ACTION_DRONE_HGAS:
                if self.can_do(obs, actions.FUNCTIONS.Harvest_Gather_screen.id):
                    extractors = self.get_units_by_type(obs, units.Zerg.Extractor)
                    for extractor in extractors:
                        hatcheries = self.get_units_by_type(obs, units.Zerg.Hatchery)
                        for hatch in hatcheries:  # for each hatch
                            self.should_harvest = False
                            if self.calculateDistance(extractor.x, extractor.y, hatch.x, hatch.y) <= 35:
                                self.should_harvest = True
                                self.harvest_here = [extractor.x, extractor.y]
                                return actions.FUNCTIONS.move_camera(self.harvest_here)

            elif smart_action == ACTION_TRAIN_OVERLORD:
                if self.can_do(obs, actions.FUNCTIONS.Train_Overlord_quick.id):
                    return actions.FUNCTIONS.Train_Overlord_quick('now')

            elif smart_action == ACTION_TRAIN_ROACH:
                if self.can_do(obs, actions.FUNCTIONS.Train_Roach_quick.id):
                    return actions.FUNCTIONS.Train_Roach_quick("now")

            elif smart_action == ACTION_TRAIN_ZERGLING:
                if self.can_do(obs, actions.FUNCTIONS.Train_Zergling_quick.id):
                    return actions.FUNCTIONS.Train_Zergling_quick('now')

            elif smart_action == ACTION_TRAIN_DRONE:
                if self.can_do(obs, actions.FUNCTIONS.Train_Drone_quick.id):
                    return actions.FUNCTIONS.Train_Drone_quick('now')

        elif self.move_number == 3:
            self.move_number = 0  # empty action state atm

            smart_action, x, y = self.splitAction(self.previous_action)

            if smart_action == ACTION_BUILD_ROACH_WARREN:
                if roach_warren_count < 1 and self.can_do(obs, actions.FUNCTIONS.Build_RoachWarren_screen.id):
                        return actions.FUNCTIONS.Build_RoachWarren_screen("now", (42, 42))
                        # build right in the middle of screen
            elif smart_action == ACTION_BUILD_HATCHERY:
                if self.can_do(obs, actions.FUNCTIONS.Build_Hatchery_screen.id):
                    return actions.FUNCTIONS.Build_Hatchery_screen('now', (42,42))
                    # build right in the middle of screen
            elif smart_action == ACTION_BUILD_SPAWNINGPOOL or smart_action == ACTION_BUILD_ROACH_WARREN:
                if spawning_pools_count < 1 and self.can_do(obs, actions.FUNCTIONS.Build_SpawningPool_screen.id):
                        return actions.FUNCTIONS.Build_SpawningPool_screen('now', (42, 42))
                        # build right in the middle of screen
            elif smart_action == ACTION_BUILD_EXTRACTOR:
                if self.can_do(obs, actions.FUNCTIONS.Build_Extractor_screen.id):
                    if self.geyser_taken:
                        return actions.FUNCTIONS.Build_Extractor_screen("now", (42, 42))
                        # build right in the middle of screen
            elif smart_action == ACTION_DRONE_HARVEST or smart_action == ACTION_DRONE_HGAS:
                if self.can_do(obs, actions.FUNCTIONS.Harvest_Gather_screen.id):
                    if self.should_harvest:
                        return actions.FUNCTIONS.Harvest_Gather_screen("now", (42, 42))
                        # build right in the middle of screen

        return actions.FUNCTIONS.no_op()


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.disallowed_actions = {}

    def choose_action(self, observation, excluded_actions=[]):
        self.check_state_exist(observation)
        self.disallowed_actions[observation] = excluded_actions
        state_action = self.q_table.ix[observation, :]

        for excluded_action in excluded_actions:
            del state_action[excluded_action]

        if np.random.uniform() < self.epsilon:
            # choose best action
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        if s == s_:
            return
        self.check_state_exist(s_)
        self.check_state_exist(s)

        q_predict = self.q_table.ix[s, a]
        s_rewards = self.q_table.ix[s_, :]

        if s_ in self.disallowed_actions:
            for excluded_action in self.disallowed_actions[s_]:
                del s_rewards[excluded_action]

        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        else:
            q_target = r  # next state is terminal
        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


def main(unused_argv):
    agent = ZergAgentAttack()
    try:
        while True:
            with sc2_env.SC2Env(map_name="AcidPlant",
                                players=[sc2_env.Agent(sc2_env.Race.zerg),
                                         sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.easy)],
                                agent_interface_format=features.AgentInterfaceFormat(
                                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                                    use_feature_units=True),
                                step_mul=16,
                                game_steps_per_episode=0, visualize=False)as env:
                agent.setup(env.observation_spec(), env.action_spec())

                timesteps = env.reset()
                agent.reset()
                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
