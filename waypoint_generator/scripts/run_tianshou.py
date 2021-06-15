import os
import gym
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Independent, Normal
from asv_env import AsvEnv
from net import PolicyNet, ValueNet

from tianshou.policy import PPOPolicy
from tianshou.utils import BasicLogger
# from tianshou.utils.net.common import Net
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from tianshou.utils.net.continuous import Actor, ActorProb, Critic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=3000)
    parser.add_argument('--episode-per-collect', type=int, default=5)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--task', type=str, default='WptGen')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')

    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.25)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=1)
    parser.add_argument('--norm-adv', type=int, default=1)
    parser.add_argument('--recompute-adv', type=int, default=0)
    args = parser.parse_known_args()[0]
    return args


def run(args=get_args()):
    # torch.set_num_threads(1)
    env = AsvEnv()
    args.scan_frames = env.scan_frames
    args.action_shape = env.action_space.shape
    args.max_action = env.action_space.high[0]
    print(args.action_shape)
    print(args.max_action)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    pi_net = PolicyNet(args.scan_frames, device=args.device)
    # actor = ActorProb(pi_net, (2,), max_action=args.max_action, preprocess_net_output_dim=64, 
    actor = ActorProb(pi_net, (1,), preprocess_net_output_dim=1, 
                      device=args.device).to(args.device)

    v_net = ValueNet(args.scan_frames, device=args.device)
    critic = Critic(v_net, preprocess_net_output_dim=64, 
                    device=args.device).to(args.device)

    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    optim = torch.optim.Adam(set(
        actor.parameters()).union(critic.parameters()), lr=args.lr)

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = PPOPolicy(
        actor, critic, optim, dist,
        discount_factor=args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.eps_clip,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
        value_clip=args.value_clip,
        gae_lambda=args.gae_lambda,
        action_space=env.action_space)

    train_collector = Collector(
        policy, env, ReplayBuffer(args.buffer_size), exploration_noise=True)

    test_collector = Collector(policy, env)

    log_path = os.path.join(args.logdir, "AsvEnv", 'ppo')
    writer = SummaryWriter(log_path)
    logger = BasicLogger(writer)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return mean_rewards >= env.reward_threshold

    result = onpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.repeat_per_collect, 10,
        args.batch_size, episode_per_collect=args.episode_per_collect,
        stop_fn=stop_fn, save_fn=save_fn, logger=logger)

    assert stop_fn(result['best_reward'])

    if __name__ == "__main__":
        pprint.pprint(result)


if __name__ == "__main__":
    run()
