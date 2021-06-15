import rosbag
import copy
import numpy as np
from geometry_msgs.msg import PoseStamped

offset = 100
pose = []
pose_new = []

for topic, msg, t in rosbag.Bag('crossing.bag').read_messages(topics='/asv/pose'):
    pose.append(msg)

max_id = len(pose)
wp_num = max_id / offset
j = 0
for i in range(max_id):
    p = PoseStamped()
    p.header = pose[i].header
    if i == j * offset:
        real_id = (j+1) * offset
        # print real_id, i, j
        if real_id > max_id:
            real_id = max_id-1
        p.pose = pose[real_id].pose
        p.pose.position.x = pose[real_id].pose.position.x + \
            np.random.normal(0, 0.1, 1)[0]
        p.pose.position.y = pose[real_id].pose.position.y + \
            np.random.normal(0, 0.05, 1)[0]
        j += 1
    else:
        p.pose = pose_new[i-1].pose
        p.pose.position.x = pose_new[i-1].pose.position.x + \
            np.random.normal(0, 0.1, 1)[0]
        p.pose.position.y = pose_new[i-1].pose.position.y + \
            np.random.normal(0, 0.05, 1)[0]

    pose_new.append(p)

# print pose_new[0]
# print pose[5]
print len(pose_new)

i = 0
with rosbag.Bag('crossing_new.bag', 'w') as outbag:
    for topic, msg, t in rosbag.Bag('crossing.bag').read_messages():
        if topic == '/asv/pose':
            if i < len(pose_new):
                outbag.write(topic, pose_new[i], t)
                i += 1
        else:
            outbag.write(topic, msg, t)


# for i in range(max_id):
#    if i+offset < max_id:
#        p = PoseStamped()
#        p.header = pose[i].header
#        p.pose = pose[i+offset].pose
#        p.pose.position.x = pose[i+offset].pose.position.x + \
#            np.random.normal(0, 0.15, 1)[0]
#        p.pose.position.y = pose[i+offset].pose.position.y + \
#            np.random.normal(0, 0.05, 1)[0]
#
#        pose_new.append(p)
#    else:
#        p = PoseStamped()
#        p.header = pose[i].header
#        p.pose = pose_new[i-1].pose
#        pose_new.append(p)
