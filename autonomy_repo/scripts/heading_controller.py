#!/usr/bin/env python3

import numpy as np
import rclpy

from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

class HeadingController(BaseHeadingController):

    def __init__(self):
        super().__init__()
        
        # Proportional Control Gain
        self.declare_parameter("kp", 2.0)

    @property
    def kp(self):
        return self.get_parameter("kp").value

    
    def compute_control_with_goal(
            self,
            current_state:TurtleBotState,
            desired_state: TurtleBotState
            ) -> TurtleBotControl:
        """_summary_

        Args:
            current_state (TurtleBotState): _description_
            desired_state (TurtleBotState): _description_

        Returns:
            TurtleBotControl: _description_
        """
        
        # Calculate angular velocity needed to correct heading error
        heading_error = wrap_angle(desired_state.theta - current_state.theta)
        angular_velocity = self.get_parameter("kp").value * heading_error
        
        # Update angular velocity in Turtlebot
        msg = TurtleBotControl()
        msg.omega = angular_velocity
        return msg
        
if __name__ == "__main__":
    rclpy.init()
    heading_controller = HeadingController()
    rclpy.spin(heading_controller)
    rclpy.shutdown()