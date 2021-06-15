#!/usr/bin/env python
# Reads material definitions and publishes visualization message of segment list

from geometry_msgs.msg import Vector3
from visualization_msgs.msg import MarkerArray, Marker
from light_scan_sim.msg import SegmentList, MaterialList, CircleList
import rospy
import roslib
roslib.load_manifest("light_scan_sim")


seg_markers = MarkerArray()
circle_markers = MarkerArray()
segments = None
circles = None
materials = None


def material_callback(data):
    global materials
    materials = data.materials
    print "Got material list with %d segments" % len(data.materials)


def segment_callback(data):
    global segments
    segments = data
    print "Got segment list with %d segments" % len(data.segments)
    generate_seg_markers()


def circle_callback(data):
    global circles
    circles = data
    print "Got circle list with %d circles" % len(data.circles)
    generate_circle_markers()


def generate_seg_markers():
    global seg_markers, materials, segments

    if materials is None:
        print "Waiting for materials"
        return

    if segments is None:
        print "Waiting on segments"
        return

    seg_markers.markers = []

    for i in range(len(segments.segments)):
        segment = segments.segments[i]

        # Generate marker for each segment
        m = Marker()
        m.header.frame_id = segments.frame_id
        m.header.stamp = rospy.get_rostime()
        m.id = i
        m.type = 4  # Line list
        m.pose.orientation.x = 0
        m.pose.orientation.y = 0
        m.pose.orientation.z = 0
        m.pose.orientation.w = 1.0
        m.scale.x = 0.05
        m.scale.y = 0.05
        m.scale.z = 0.05
        m.action = 0
        m.points.append(Vector3(segment.start[0], segment.start[1], 0))
        m.points.append(Vector3(segment.end[0], segment.end[1], 0))
        if (segment.type < len(materials)):
            m.color.r = materials[segment.type].color[0]
            m.color.g = materials[segment.type].color[1]
            m.color.b = materials[segment.type].color[2]
        m.color.a = 0.8
        seg_markers.markers.append(m)

    print "Generated segment markers"


def generate_circle_markers():
    global circle_markers, materials, circles

    if materials is None:
        print "Waiting for materials"
        return

    if circles is None:
        print "Waiting on circles"
        return

    circle_markers.markers = []

    for i in range(len(circles.circles)):
        circle = circles.circles[i]

        # Generate marker for each circle
        m = Marker()
        m.header.frame_id = circles.frame_id
        m.header.stamp = rospy.get_rostime()
        m.id = i
        m.type = m.CYLINDER
        m.pose.orientation.x = 0
        m.pose.orientation.y = 0
        m.pose.orientation.z = 0
        m.pose.orientation.w = 1.0
        m.scale.x = circle.radius*2
        m.scale.y = circle.radius*2
        m.scale.z = 0.1
        m.action = m.ADD
        m.points.append(Vector3(circle.center[0], circle.center[1], 0))
        if (circle.type < len(materials)):
            m.color.r = materials[circle.type].color[0]
            m.color.g = materials[circle.type].color[1]
            m.color.b = materials[circle.type].color[2]
        m.color.a = 1.
        circle_markers.markers.append(m)

    print "Generated circle markers"


# Publish the
if __name__ == '__main__':
    rospy.init_node('visualize_circle_list')

    # Todo: Load the material list

    rospy.Subscriber(rospy.get_param(
        '~input_topic', '/segments'), SegmentList, segment_callback)
    rospy.Subscriber(rospy.get_param(
        '~input_topic', '/circles'), CircleList, circle_callback)
    rospy.Subscriber(rospy.get_param(
        '~materials_topic', '/materials'), MaterialList, material_callback)
    pub = rospy.Publisher(rospy.get_param(
        '~output_topic', '/circle_vis'), MarkerArray, queue_size=1)

    rate = rospy.Rate(1.0)
    while not rospy.is_shutdown():
            # Publish the MarkerArray
        # pub.publish(seg_markers)
        pub.publish(circle_markers)
        rate.sleep()
