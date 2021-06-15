import os

import gym
import joblib
import tensorflow as tf
import tensorlayer as tl

import ppo.tf_util as U
from ppo.distributions import CategoricalPdType, make_pdtype
from ppo.mpi_running_mean_std import RunningMeanStd


# agent: policy net and value net
class Agent(object):
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, sess, ob_space, ac_space, gaussian_fixed_var=True):
        self.sess = sess
        self.ac_space = ac_space

        # observation
        assert isinstance(ob_space, gym.spaces.Dict)
        self.scan_ph = U.get_placeholder(name='scan_ph', dtype=tf.float32, 
            shape=[None, ob_space['scan'].shape[0], ob_space['scan'].shape[1]])
        self.pose_ph = U.get_placeholder(name='pose_ph', dtype=tf.float32, 
            shape=[None, ob_space['pose'].shape[0]])
        self.goal_ph = U.get_placeholder(name='goal_ph', dtype=tf.float32, 
            shape=[None, ob_space['goal'].shape[0]])
        self.lac_ph = U.get_placeholder(name='lac_ph', dtype=tf.float32, 
            shape=[None, ob_space['lac'].shape[0]])

        # observation filters
        with tf.variable_scope("pose_filter"):
            self.ob_pose_rms = RunningMeanStd(shape=ob_space['pose'].shape)
        with tf.variable_scope("goal_filter"):
            self.ob_goal_rms = RunningMeanStd(shape=ob_space['goal'].shape)
        with tf.variable_scope("lac_filter"):
            self.ob_lac_rms = RunningMeanStd(shape=ob_space['lac'].shape)

        # action
        self.pdtype = pdtype = make_pdtype(ac_space)

        # network
        ob_scan = self.scan_ph / 250.
        ob_pose = tf.clip_by_value((self.pose_ph - self.ob_pose_rms.mean) / self.ob_pose_rms.std,
                                    -5.0, 5.0)
        ob_goal = tf.clip_by_value((self.goal_ph - self.ob_goal_rms.mean) / self.ob_goal_rms.std,
                                    -5.0, 5.0)
        ob_lac = tf.clip_by_value((self.lac_ph - self.ob_lac_rms.mean) / self.ob_lac_rms.std,
                                   -5.0, 5.0)

        net = tl.layers.InputLayer(ob_scan, name='input')
        net = tl.layers.Conv1d(
            net, 16, 3, 2, act=tf.nn.relu, name="cnn1")
        net = tl.layers.Conv1d(
            net, 16, 8, 2, act=tf.nn.relu, name="cnn2")
        net = tl.layers.FlattenLayer(net, name='flat')
        net = tl.layers.DenseLayer(net, 128, act=tf.nn.tanh, name='cnn_fc')
        cnn_output = net.outputs

        concat_net = tl.layers.InputLayer(
            tf.concat([ob_goal, ob_pose, ob_lac, cnn_output], axis=1), 
            name='concat')
        fc_net = tl.layers.DenseLayer(concat_net, 64, act=tf.nn.tanh, name='fc')

        # value net
        vf_net = tl.layers.DenseLayer(fc_net, 1, name='vf')
        self.vpred = vf_net.outputs[:, 0]

        # policy net
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = tl.layers.DenseLayer(fc_net, pdtype.param_shape()[0]//2, 
                name='mean')
            logstd = tf.get_variable(name="logstd", shape=[1, 
                pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        else:
            ac_net = tl.layers.DenseLayer(
                fc_net, pdtype.param_shape()[0], name='logits')
            pdparam = ac_net.outputs

        self.pd = pdtype.pdfromflat(pdparam)
        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, self.scan_ph, self.pose_ph, self.goal_ph, self.lac_ph], 
                               [ac, self.vpred])


    def load(self, path):
        loaded_params = joblib.load(path)
        restores = []
        params = tf.trainable_variables()
        for p, loaded_p in zip(params, loaded_params):
            restores.append(p.assign(loaded_p))
        self.sess.run(restores)
        print("loaded weights...")

    def save(self, path):
        params = tf.trainable_variables()
        ps = self.sess.run(params)
        joblib.dump(ps, path)

    def act(self, stochastic, scan, pose, goal, lac):
        ac1, vpred1 = self._act(stochastic, scan[None], pose[None], goal[None], lac[None])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []
