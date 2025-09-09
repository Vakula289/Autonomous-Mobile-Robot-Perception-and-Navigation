#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

class PeriodicPrint (Node):
    def __init__(self):
        super().__init__("PeriodicPrint")

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.time = self.create_timer(0.2, self.timer_callback)
        # self.subs = self.create_subscription(Twist, '/cmd_vel', self.callback, 10)
        self.kill = self.create_subscription(Bool, '/kill', self.killcb, 10)

    def timer_callback(self):
        
        msg = Twist()
        msg.linear.x = 5.0
        msg.angular.z = 0.0

        self.pub.publish(msg)

    
    def killcb(self, msg):
        if (msg.data == True):
            self.time.cancel()
            msgTw = Twist()
            msgTw.linear.x = 0.0
            msgTw.angular.z = 0.0
            self.pub.publish(msgTw)
            
    
if __name__ == "__main__":
    rclpy.init()
    node = PeriodicPrint()
    rclpy.spin(node)
    rclpy.shutdown()

    