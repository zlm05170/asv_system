import numpy as np
# import scipy.signal
# import tensorflow as tf

class RunningStat(object):
    def __init__(self, shape):
        self._size = 0
        self._mean, self._std = np.zeros(shape), np.zeros(shape)

    def add(self, x):
        x = np.asarray(x)
        assert x.shape == self._mean.shape

        self._size += 1
        if self._size == 1:
            self._mean = x
        else:
            self._mean_old = self._mean.copy()
            self._mean = self._mean_old + (x - self._mean_old) / self._size
            self._std = self._std + (x - self._mean_old) * (x - self._mean)

    @property
    def size(self):
        return self._size

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._std / (self._size - 1) if self._size > 1 else self._std

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._mean.shape

class RunningAverageFilter(object):
    """
    do a running average filter of the incoming observations and rewards
    """

    def __init__(self, shape, obstype=None,
                 demean=False, destd=False, update=False,
                 clip=None):
        self.demean = demean
        self.destd = destd
        self.clip = clip
        self.obstype = obstype
        self.update = update

        self.rs = RunningStat(shape)

    def __call__(self, x):
        # x is scanObsBatch, len(x) == num_agents
        if len(x) > 1:
            filtered_x = []
            for i in range(len(x)):  # for each agent
                if self.obstype == "scan":
                    data = np.stack(
                        (x[i].scan_pprev.ranges,
                         x[i].scan_prev.ranges,
                         x[i].scan_now.ranges),
                        axis=1)
                elif self.obstype == "goal":
                    data = [x[i].goal_now.goal_dist, x[i].goal_now.goal_theta]
                elif self.obstype == "action":
                    data = [
                        [x[i].ac_pprev.vx, x[i].ac_pprev.vz],
                        [x[i].ac_prev.vx, x[i].ac_prev.vz]]
                elif self.obstype == "vel":
                    data = [x[i].vel_now.vx, x[i].vel_now.vz]
                else:
                    data = x
                data = np.array(data)

                # print("data shape before filter: {}".format(data.shape))
                if self.update:
                    self.rs.add(data[-1])
                if self.demean:
                    data -= self.rs.mean
                if self.destd:
                    data /= (self.rs.std + 1e-8)
                if self.clip is not None:
                    data = np.clip(data, -self.clip, self.clip)

                filtered_x.append(data)
            return np.array(filtered_x)
        else:
            x = np.array(x)
            if self.update:
                self.rs.add(x)
            if self.demean:
                x -= self.rs.mean
            if self.destd:
                x /= (self.rs.std + 1e-8)
            if self.clip is not None:
                x = np.clip(x, -self.clip, self.clip)
            return x
