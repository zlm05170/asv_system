import time
from collections import OrderedDict, deque

import os
import rospy
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from tabulate import tabulate
from tensorflow.python.tools import inspect_checkpoint as chkp

from . import tf_util as U
from .dataset import Dataset
from .mpi_adam import MpiAdam
from .mpi_moments import mpi_moments


def traj_segment_generator(pol, env, horizon, stochastic):
    t = 0
    ac = pol.ac_space.sample()
    new = True
    scan, pose, goal, lac = env.reset()

    curr_ep_ret = 0
    curr_ep_len = 0
    ep_rets = []
    ep_lens = []

    scans = np.array([scan for _ in range(horizon)])
    poses = np.array([pose for _ in range(horizon)])
    goals = np.array([goal for _ in range(horizon)])
    lacs = np.array([lac for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    # prev_acs = acs.copy()

    while not rospy.is_shutdown():
        env.render()
        # prev_ac = ac
        ac, vpred = pol.act(stochastic, scan, pose, goal, lac)
        if t > 0 and t % horizon == 0:
            yield {"scan": scans, "pose": poses, "goal": goals, "lac": lacs,
                   "rew": rews, "vpred": vpreds,
                   "new": news, "ac": acs, 
                #    "prev_ac": prev_acs,
                   "next_vpred": vpred * (1-new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens}
            ep_rets = []
            ep_lens = []
        i = t % horizon
        scans[i] = scan
        poses[i] = pose
        goals[i] = goal
        lacs[i] = lac
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        # prev_acs[i] = prev_ac

        # observation, reward, done
        # print("action [label]: ", ac)
        scan, pose, goal, lac, rew, new, _ = env.step(ac)
        rews[i] = rew
        curr_ep_ret += rew
        curr_ep_len += 1

        if new:  # if done is True:
            ep_rets.append(curr_ep_ret)
            ep_lens.append(curr_ep_len)
            curr_ep_ret = 0
            curr_ep_len = 0
            scan, pose, goal, lac = env.reset()
        t += 1
        print(("t: {}, ac: {}".format(t, ac)))


def add_vtarg_and_adv(seg, gamma, lamda):
    new = np.append(seg["new"], 0)
    vpred = np.append(seg["vpred"], seg["next_vpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, "float32")
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(list(range(T))):
        nonterminal = 1 - new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lamda * nonterminal * \
            lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


def print_statistics(stats):
    print((tabulate(
        [k_v for k_v in list(stats.items()) if np.asarray(k_v[1]).size == 1],
        tablefmt="grid")))


def learn(env, pol_fn,
          ob_space, ac_space,
          timesteps_per_actorbatch,
          clip_param, entcoeff,
          optim_epochs, optim_stepsize, optim_batchsize,
          gamma, lamda,
          max_timesteps=0, max_episodes=0,
          max_iters=0, max_seconds=0,
          callback=None, adam_epsilon=1e-5,
          schedule='constant',
          train=True,
          restore=False, 
          save_path="~/"):
    sess = tf.get_default_session()

    model_save_path = save_path + "model/"
    summary_save_path = save_path + "summary/"
    if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

    if not os.path.exists(summary_save_path):
            os.makedirs(summary_save_path)

    # for tensorboard
    tb_rew_ph = tf.placeholder(dtype=tf.float32, name="tb_rew_ph")
    tb_len_ph = tf.placeholder(dtype=tf.float32, name="tb_len_ph")
    tb_ent_ph = tf.placeholder(dtype=tf.float32, name="tb_ent_ph")
    tb_kl_ph = tf.placeholder(dtype=tf.float32, name="tb_kl_ph")
    tb_vf_loss_ph = tf.placeholder(dtype=tf.float32, name="tb_vf_loss_ph")
    tb_pol_surr_ph = tf.placeholder(dtype=tf.float32, name="tb_pol_surr_ph")
    with tf.name_scope('results'):
        tf.summary.scalar('reward', tb_rew_ph)
        tf.summary.scalar('ep_len', tb_len_ph)
        tf.summary.scalar('entropy', tb_ent_ph)
        tf.summary.scalar('kl', tb_kl_ph)
        tf.summary.scalar('vf_loss', tb_vf_loss_ph)
        tf.summary.scalar('pol_surr', tb_pol_surr_ph)

    tb_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    tb_data = tf.summary.merge_all()
    tb_writer = tf.summary.FileWriter(
        summary_save_path + "{}".format(tb_time), sess.graph)
    # end tensorboard

    pi = pol_fn("pi", sess, ob_space, ac_space)
    oldpi = pol_fn("oldpi", sess, ob_space, ac_space)

    atarg = tf.placeholder(dtype=tf.float32, shape=[None])
    ret = tf.placeholder(dtype=tf.float32, shape=[None])
    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])
    clip_param = clip_param * lrmult

    scan = U.get_placeholder_cached(name="scan_ph")
    pose = U.get_placeholder_cached(name="pose_ph")
    goal = U.get_placeholder_cached(name="goal_ph")
    lac = U.get_placeholder_cached(name="lac_ph")
    ac = pi.pdtype.sample_placeholder([None])

    kl_oldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kl_oldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))
    surr1 = ratio * atarg
    surr2 = tf.clip_by_value(ratio, 1.-clip_param, 1.+clip_param) * atarg
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    loss_and_grad = U.function([scan, pose, goal, lac, ac, atarg, ret, lrmult],
                               losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function(
        [], [], updates=[tf.assign(oldv, newv)
                         for (oldv, newv) in U.zipsame(
                                 oldpi.get_variables(),
                                 pi.get_variables())])
    compute_losses = U.function([scan, pose, goal, lac, ac, atarg, ret, lrmult], losses)

    U.initialize()
    adam.sync()

    # rollout
    seg_gen = traj_segment_generator(
        pi, env, timesteps_per_actorbatch, stochastic=train)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100)
    rewbuffer = deque(maxlen=100)
    best_score = -10000.

    assert sum(
        [max_iters > 0, max_timesteps > 0,
         max_episodes > 0, max_seconds > 0]) == 1, \
        "Only one time constraint permitted"

    saver = tf.train.Saver()
    if not train or restore:
        saver.restore(sess, model_save_path + "last.ckpt")
        # chkp.print_tensors_in_checkpoint_file(
        #    "./results/model/last.ckpt", tensor_name="",
        #    all_tensors=True, all_tensor_names=True)

    while not rospy.is_shutdown():
        if callback:
            callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.
        elif schedule == 'linear':
            cur_lrmult = max(1. - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        print(("************** Iteration %i ************" % iters_so_far))

        print("Rollouting...")
        seg = next(seg_gen)
        add_vtarg_and_adv(seg, gamma, lamda)

        scan, pose, goal, lac, ac, atarg, tdlamret = \
            seg["scan"], seg["pose"], seg["goal"], seg["lac"], \
            seg["ac"], seg["adv"], seg["tdlamret"]
        vpred_before = seg["vpred"]
        atarg = (atarg - atarg.mean()) / atarg.std()
        d = Dataset(dict(scan=scan, pose=pose, goal=goal, lac=lac,
                         ac=ac, atarg=atarg, vtarg=tdlamret),
                    shuffle=True)
        optim_batchsize = optim_batchsize or pose.shape[0]

        if hasattr(pi, "ob_pose_rms"):
            pi.ob_pose_rms.update(pose)
        if hasattr(pi, "ob_goal_rms"):
            pi.ob_goal_rms.update(goal)
        if hasattr(pi, "ob_lac_rms"):
            pi.ob_lac_rms.update(lac)

        assign_old_eq_new()
        if train:
            print("Optimizing...")
            for _ in range(optim_epochs):
                losses = []
                for batch in d.iterate_once(optim_batchsize):
                    landg = loss_and_grad(
                        batch["scan"], batch["pose"], batch["goal"], batch["lac"],
                        batch["ac"], batch["atarg"],
                        batch["vtarg"], cur_lrmult)
                    # print("length of lossandgrad: ", len(landg))
                    adam.update(landg[-1], optim_stepsize * cur_lrmult)
                    losses.append(landg[:-1])
                # print("losses_mean: ", np.mean(losses, axis=0))

        print("Evaluating losses...")
        stats = OrderedDict()
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(
                batch["scan"], batch["pose"], batch["goal"], batch["lac"],
                batch["ac"], batch["atarg"],
                batch["vtarg"], cur_lrmult)
            losses.append(newlosses)
        meanlosses, _, _ = mpi_moments(losses, axis=0)
        stats["meanlosses"] = meanlosses
        for (lossval, name) in U.zipsame(meanlosses, loss_names):
            stats["loss_"+name] = lossval
        stats["ev_tdlm_before"] = U.explained_variance(
            vpred_before, tdlamret)
        lrlocal = (seg["ep_lens"], seg["ep_rets"])
        listoflfpairs = MPI.COMM_WORLD.allgather(lrlocal)
        lens, rews = list(map(flatten_lists, list(zip(*listoflfpairs))))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        stats["EpLenMean"] = np.mean(lenbuffer)
        stats["EpRewMean"] = np.mean(rewbuffer)
        stats["EpthisIter"] = len(lens)
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        stats["EpisodesSoFar"] = episodes_so_far
        stats["TimestepsSoFar"] = timesteps_so_far
        stats["TimeElapsed"] = time.time() - tstart
        print_statistics(stats)

        if train:
            saver.save(sess, model_save_path + "last.ckpt")
            if stats["EpRewMean"] > best_score:
                saver.save(sess, model_save_path + "best.ckpt")
                best_score = stats["EpRewMean"]
            print("saved the model...")

        tb_feed_dict = {
            tb_rew_ph: stats["EpRewMean"],
            tb_len_ph: stats["EpLenMean"],
            tb_ent_ph: stats["loss_ent"],
            tb_kl_ph: stats["loss_kl"],
            tb_vf_loss_ph: stats["loss_vf_loss"],
            tb_pol_surr_ph: stats["loss_pol_surr"]
        }
        summary = sess.run(tb_data, tb_feed_dict)
        tb_writer.add_summary(summary, iters_so_far)
