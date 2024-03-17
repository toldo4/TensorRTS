import sys
import os
tensor_path = os.path.abspath(os.path.join(os.path.basename(__file__), os.pardir, os.pardir, os.pardir))
sys.path.append(tensor_path)

import random 
from typing import Dict, List, Mapping, Tuple, Set
from entity_gym.env import Observation
from entity_gym.runner import CliRunner
from entity_gym.env import *
from TensorRTS import Agent
from enn_trainer import load_checkpoint, RogueNetAgent

class toldo_bot(Agent):
    def __init__(self, init_observation : Observation, action_space : Dict[ActionName, ActionSpace]) -> None: 
        super().__init__(init_observation, action_space)
        checkpoint = load_checkpoint("bots/toldo_bot/checkpoint")
        self.current = RogueNetAgent(checkpoint.state.agent)

    def take_turn(self, current_game_state: Observation) -> Mapping[ActionName, Action]:
        if self.is_player_one:
            action, predicted_return = self.agent.act(current_game_state)
            return action
        elif self.is_player_two:
            entities = current_game_state.features
            actions = current_game_state.actions
            done = current_game_state.done
            reward = current_game_state.reward

            clusters = entities["Cluster"]
            tensors = entities["Tensor"]

            opp_clusters = [[32-i-1, j] for i, j in clusters]
            opp_tensors = [[32-i-1, j, k, l] for i, j, k, l in tensors]

            opp_obs = Observation(
                entities={
                "Cluster": (
                    opp_clusters,
                    [("Cluster", i) for i in range(len(opp_clusters))]
                ),
                "Tensor": (
                    opp_tensors,
                    [("Tensor", i) for i in range(len(opp_tensors))]
                )
                },
                actions=actions,
                done=done,
                reward=reward
            )

            action, predicted_return = self.agent.act(opp_obs)
            return action


    def on_game_start(self, is_player_one : bool, is_player_two : bool) -> None:
        return super().on_game_start(is_player_one, is_player_two)
    
    def on_game_over(self, did_i_win : bool, did_i_tie : bool) -> None:
        return super().on_game_over(did_i_win, did_i_tie)
    
def agent_hook(init_observation : Observation, action_space : Dict[ActionName, ActionSpace]) -> Agent: 
    """Creates an agent of this type

    Returns:
        Agent: toldo bot
    """
    return toldo_bot(init_observation, action_space)

def student_name_hook() -> str: 
    """Provide the name of the student as a string

    Returns:
        str: Name of student
    """
    return 'Mansour Rezaei'
