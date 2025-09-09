#!/usr/bin/env python3


import numpy as np
import rclpy
import typing as T
from scipy.signal import convolve2d
from enum import Enum

from asl_tb3_lib.control import Node
from asl_tb3_lib.navigation import TrajectoryPlan
from asl_tb3_lib.navigation import StochOccupancyGrid2D, Path
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid


class FrontierExploration(Node):
    UNKNOWN = -1 
    THRESHOLD = -0.01
    
    @property
    def active(self) -> bool:
        return self.get_parameter("active").value
    
    def __init__(self):
        super().__init__("frontier_exploration")  # Instantiate Node class
        
        # State subscriber – listen to state publisher to get state updates
        self.state: T.Optional[TurtleBotState] = None
        self.state_sub = self.create_subscription(TurtleBotState, "/state", self.state_callback, 10)
        
        # Map subscriber – listen to map publisher to get map updates.
        self.occupancy: T.Optional[StochOccupancyGrid2D] = None
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)
        
        # Navigation subscriber – listens to /nav_success to know when to plan for the next path
        self.nav_sub = self.create_subscription(Bool, "/nav_success", self.plan, 10)
        self.nav_sucess_true = self.create_publisher(Bool, "/nav_success", 10)
        
        # Command navigator publisher – publish to /cmd_nav to tell navigator to plan
        self.cmd_nav_pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)
        
        self.detector = self.create_subscription(Bool, "/detector_bool", self.detect_cb, 10)
        
        self.flag = False
        self.callback_time = None
        self.start_time = None

        
    def plan(self,
        msg: Bool
    ) -> None:
        """When receiving a call for a successful navigation, begin to plan the
        next path, or rather, examine the probs map, determine what to explore,
        then explore the nearest point.

        Args:
            msg (Bool): true or false boolean that says whether or not to plan
        """
        self.get_logger().info("In plan")
        # Ensure we have an occupancy map:
        if self.occupancy is None:
            self.get_logger().warn("Unable to plan: occupancy map is not yet availale")
        
        # Plan new path by finding explorable regions
        """
        Steps:
            1.) Compute explorables (states we can explore)
            2.) Pick a state in explorables, maybe just the closest one
            3.) Call compute_trajectory_plan using our current state and chosen state from 2
        """
        
            
        
        explorables = self.explore()
        if not explorables.size:
            self.get_logger().info("Finished Exploring")
            return
        
        # Pick a state (the closest one)
        current_state = np.array([self.state.x, self.state.y])
        distances = np.linalg.norm(explorables - current_state, axis=1) # > self.THRESHOLD
        goal = explorables[np.argmin(distances, axis=0)]
        
        # Make a goal state
        goal_state = TurtleBotState()
        goal_state.x = goal[0]
        goal_state.y = goal[1]
        goal_state.theta = 0.

        # self.get_logger().info(f"going to state x: {goal_state.x}, y: {goal_state.y}")
        
        # If planning succeeded
        if self.flag:
            self.get_logger().info("Stopping rn")
            self.cmd_nav_pub.publish(self.state)
        else:
            self.get_logger().info("Going to goal")
            self.cmd_nav_pub.publish(goal_state)

        
    
    def explore(self) -> np.ndarray:
                
        window_size = 13
        grid = self.occupancy.probs

        kernel = np.ones((window_size, window_size))

        unknown_mask = (grid<0)
        known_mask = (grid >= 0) & (grid < 0.5)
        occupied_mask = (grid >= 0.5)

        unknown = convolve2d(unknown_mask, kernel, mode='same', boundary='fill', fillvalue=0)
        known = convolve2d(known_mask, kernel, mode='same', boundary='fill', fillvalue=0)
        occupied = convolve2d(occupied_mask, kernel, mode='same', boundary='fill', fillvalue=0)

        total = window_size*window_size

        h1 = (unknown >= (0.2*total))
        h2 = (known >= (0.3*total))
        h3 = (occupied == 0)

        valid_cells = h1 & h2 & h3

        frontier_states = np.argwhere(valid_cells.T)
        
        if frontier_states.size == 0:
            return None, None
        
        frontier_states = self.occupancy.grid2state(frontier_states)

        return frontier_states
        
    def state_callback(self,
        msg: TurtleBotState
    ) -> None:
        """ callback triggered when receiving latest turtlebot state

        Args:
            msg (TurtleBotState): latest turtlebot state
        """
        self.state = msg
        
    def map_callback(self,
        msg: StochOccupancyGrid2D
    ) -> None:
        """ Callback triggered when the map is updated

        Args:
            msg (StochOccupancyGrid2D): updated map message
        """
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=7,
            probs=msg.data,
        )    
    
    def detect_cb(self, msg:Bool):
        if msg.data:
            if not self.callback_time:
                self.callback_time = self.get_clock().now().nanoseconds / 1e9
            if((self.get_clock().now().nanoseconds / 1e9) - self.callback_time > 15):
                self.callback_time = None
                self.start_time = None
            else:
                if not self.start_time:
                    self.start_time = self.get_clock().now().nanoseconds / 1e9
                if((self.get_clock().now().nanoseconds / 1e9) - self.start_time < 5):
                    self.get_logger().info("Flag is true")
                    self.flag = True
                else:
                    self.get_logger().info("Flag is False")
                    self.flag = False
                true_msg = Bool()
                true_msg.data = True
                self.nav_sucess_true.publish(true_msg)
            


    # def detect_cb(self, msg: Bool):
    #     if not msg.data:
    #         return
        
    #     if self.active:
    #         if not self.callback_time:
    #             self.set_parameters([rclpy.Parameter("active", value=False)])
    #             self.callback_time = self.get_clock().now().nanoseconds / 1e9
    #         if((self.get_clock().now().nanoseconds / 1e9) - self.callback_time > 15):
    #             self.callback_time = None
    #     else:
    #         if not self.start_time:
    #             self.get_logger().info("Setting start time")
    #             self.start_time = self.get_clock().now().nanoseconds / 1e9
    #         if((self.get_clock().now().nanoseconds / 1e9) - self.start_time > 5):
    #             self.get_logger().info("Setting active true")
    #             self.set_parameters([rclpy.Parameter("active", value=True)])
    #             self.start_time = None
    #             true_msg = Bool()
    #             true_msg.data = True
    #             self.get_logger().info("Planning again")
    #             self.plan(true_msg)

if __name__ == "__main__":
    rclpy.init()
    frontier_exploration = FrontierExploration()
    rclpy.spin(frontier_exploration)
    rclpy.shutdown()