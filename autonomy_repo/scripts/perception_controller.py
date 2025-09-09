#!/usr/bin/env python3

import numpy as np
import rclpy

from asl_tb3_lib.control import BaseController
# from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from std_msgs.msg import Bool

class PerceptionController(BaseController):

    def __init__(self):
        super().__init__(node_name="PerceptionController")
        
        # Proportional Control 
        self.start_time = None
        self.flag = -1
        self.callback_time = None
        # self.ignore_flag = False
        self.declare_parameter("active", True)
        self.sup = self.create_subscription(Bool, '/detector_bool', self.callback, 10)


    @property
    def active(self) -> bool:
        return self.get_parameter("active").value

    
    def compute_control(self) -> TurtleBotControl:
        """_summary_

        Args:
            current_state (TurtleBotState): _description_
            desired_state (TurtleBotState): _description_

        Returns:
            TurtleBotControl: _description_
        """
        msg = TurtleBotControl()

        
        if self.active:
            msg.omega = 0.5
        else:
            if not self.start_time:
                self.start_time = self.get_clock().now().nanoseconds / 1e9
                self.end_time = self.start_time
            msg.omega = 0.0
            if((self.get_clock().now().nanoseconds / 1e9) - self.start_time > 5):
                msg.omega = 0.5
                self.set_parameters([rclpy.Parameter("active", value=True)])
                self.start_time = None
        # Update angular velocity in Turtlebot
        return msg
    
    def callback (self, msg):
        if msg.data and self.active:
            if not self.callback_time:
                self.set_parameters([rclpy.Parameter("active", value=False)])
                self.callback_time = self.get_clock().now().nanoseconds / 1e9
            if((self.get_clock().now().nanoseconds / 1e9) - self.callback_time > 8):
                self.callback_time = None
                
            
            
            
            
        
if __name__ == "__main__":
    rclpy.init()
    perception_controller = PerceptionController()
    rclpy.spin(perception_controller)
    rclpy.shutdown()