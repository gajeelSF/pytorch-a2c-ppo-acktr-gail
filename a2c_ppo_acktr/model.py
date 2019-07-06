import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space,num_actors=3, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = FCNBase(obs_shape[0], num_actors,**base_kwargs)
        self.dists = []

        if action_space.__class__.__name__ == "Discrete":
            self.num_outputs = action_space.n
            for i in range(num_actors):
                self.dists.append(Categorical(self.base.output_size, self.num_outputs))
        elif action_space.__class__.__name__ == "Box":
            self.num_outputs = action_space.shape[0]
            for i in range(num_actors):
                self.dists.append(DiagGaussian(self.base.output_size, self.num_outputs))
        elif action_space.__class__.__name__ == "MultiBinary":
            self.num_outputs = action_space.shape[0]
            for i in range(num_actors):
                self.dists.append(Bernoulli(self.base.output_size, self.num_outputs))
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs,rnn_hxs, masks, deterministic=False):
        value, actors, cdist, choice, choice_log_prob,rnn_hxs  = self.base(inputs, rnn_hxs,masks)

        hidden_actor = torch.empty(1, self.base.output_size)
        actions = torch.empty(choice.shape[0],self.num_outputs)
        action_log_probs = torch.empty(choice.shape[0])
        for i in range(0, inputs.shape[0]):
            hidden_actor[0] = actors[choice[i]](inputs[i])
            dist = self.dists[choice[i]](hidden_actor)
            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()
            actions[i] = action

            action_log_prob = dist.log_probs(action)
            action_log_probs[i] = action_log_prob

        return value, actions, choice, action_log_probs, choice_log_prob, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _, _, _ ,_ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs,rnn_hxs, masks, action, choice):
        value, actors, cdist, _, _ ,_= self.base(inputs,rnn_hxs, masks)

        hidden_actor = torch.empty(1, self.base.output_size)
        action_log_probs = torch.empty(choice.shape[0],1)
        dist_entropys = torch.empty(choice.shape[0],1)

        for i in range(0, inputs.shape[0]):
            hidden_actor = actors[choice[i]](inputs[i])

            dist = self.dists[choice[i]](hidden_actor.view(1,hidden_actor.shape[0]))
            action_log_prob = dist.log_probs(action[i])
            action_log_probs[i] = action_log_prob

            dist_entropys[i] = dist.entropy()

        choice_log_probs = cdist.log_probs(choice)
        dist_entropy = dist_entropys.mean() + cdist.entropy().mean()

        return value, action_log_probs, choice_log_probs, dist_entropy


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs

class FCNBase(NNBase):
    def __init__(self, num_inputs, num_actors,recurrent=False, hidden_size=64):
        super(FCNBase, self).__init__(recurrent, num_inputs, hidden_size)
        self.num_actors = num_actors

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.cdist = Categorical(hidden_size, self.num_actors)

        self.decider = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.actors = []
        '''
        for i in range(self.num_actors):
            self.actors.append(
            nn.Sequential(
                init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
            )
        '''
        for i in range(self.num_actors):
            self.actors.append(
           nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
           )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs,rnn_hxs, masks):
        hidden_critic = self.critic(inputs)
        hidden_decision = self.decider(inputs)
        cdist = self.cdist(hidden_decision)
        choice = cdist.sample()
        choice.data.fill_(2)
        choice_log_probs = cdist.log_probs(choice)

        return self.critic_linear(hidden_critic), self.actors, cdist, choice, choice_log_probs, rnn_hxs
