#!/usr/bin/env python

import rospy
import numpy as np
from gym import spaces

from asv_env import AsvEnv
from ppo import tf_util as U
from ppo import agent, ppo_clipped

def main():
    # ob_dim = rospy.get_param("~ob_dim")
    ac_type = rospy.get_param("~ac_type")
    save_csv = rospy.get_param("~save_csv")
    num_timesteps = rospy.get_param("~num_timesteps")
    train_flag = rospy.get_param("~train")
    restore_flag = rospy.get_param("~restore")
    save_path = rospy.get_param("~save_path")
    print("{}, {}, {}, {}, {}, {}".format(ac_type, num_timesteps,
        save_csv, train_flag, restore_flag, save_path))

    U.make_session(num_cpu=12).__enter__()

    # ob_shape = spaces.Box(-10., 10, shape=(1,), dtype=np.float32)
    ob_space = spaces.Dict(
        {"scan": spaces.Box(
            low=10.0, high=250.0, shape=(3, 180), dtype=np.float32),
         "pose": spaces.Box(
            low=np.array([-1000.0, -1000., -np.pi]), high=np.array([1000., 1000., np.pi]), dtype=np.float32),
            # low=-1000, high=1000, shape=(2,), dtype=np.float32),
         "goal": spaces.Box(
            low=np.array([0.0, -np.pi]), high=np.array([1000., np.pi]), dtype=np.float32),
            # low=-1000, high=1000, shape=(2,), dtype=np.float32),
         "lac": spaces.Box(  # last action
            # low=np.array([0.0, -np.pi/2]), high=np.array([10., np.pi/2]), dtype=np.float32),
            low=-10, high=10, shape=(1,), dtype=np.float32),
            # low=-10., high=10., shape=(2,), dtype=np.float32),
        })
    
    if ac_type == "continuous":
        ac_space = spaces.Box(
            # low=np.array([0.0, -np.pi/2]), high=np.array([10., np.pi/2]), dtype=np.float32)
            # low=-np.pi/2, high=np.pi/2, shape=(1,), dtype=np.float32)
            low=-10., high=10., shape=(1,), dtype=np.float32)
    elif ac_type == "discrete":
        ac_space = spaces.Discrete(3)  # -10, 0, 10
    else:
        rospy.logerr("WRONG ACTION SHAPE!!!")
        raise RuntimeError

    def pol_fn(name, sess, ob_space, ac_space):
        return agent.Agent(
            name=name, sess=sess,
            ob_space=ob_space, ac_space=ac_space)

    env = AsvEnv()
    ppo_clipped.learn(env, pol_fn, 
                      ob_space=ob_space, 
                      ac_space=ac_space,
                      max_timesteps=num_timesteps,
                      timesteps_per_actorbatch=256,
                      clip_param=0.2, 
                      entcoeff=0.01,
                      optim_epochs=4,
                      optim_stepsize=1e-3,
                      optim_batchsize=64,
                      gamma=0.99, 
                      lamda=0.95,
                      schedule='linear',
                      train=train_flag,
                      restore=restore_flag, 
                      save_path=save_path)


if __name__ == '__main__':
    rospy.init_node("drl_wpt")
    try:
        main()
    except rospy.ROSInterruptException: 
        pass
    # rospy.spin()
