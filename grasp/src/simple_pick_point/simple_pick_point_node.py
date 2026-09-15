#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from simple_pick_point import SimplePickPoint

NODE_NAME = 'simple_pick_point'

from threading import Lock

def main():
    rospy.init_node(NODE_NAME, anonymous=True)
    simple_pick_point = SimplePickPoint(NODE_NAME)
    lock = Lock()
    while True:
        rospy.sleep(1)
        with lock:
            if simple_pick_point.draw:
                simple_pick_point.draw_pick_point(simple_pick_point.preprocessed_image)
                simple_pick_point.draw = False

if __name__ == '__main__':
    main()
