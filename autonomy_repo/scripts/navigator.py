#!/usr/bin/env python3

import numpy as np
import rclpy
import typing as T

from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from asl_tb3_lib.grids import StochOccupancyGrid2D
from scipy.interpolate import splev, splrep



class CustomNavigator(BaseNavigator):
    
    def __init__(self):
        super().__init__()
        
        # Constants
        self.kpx = 2.0
        self.kpy = 2.0
        self.kdx = 2.0
        self.kdy = 2.0
        self.kp = 2.0
        
        # Minimum velocity threshold
        self.V_PREV_THRES = 0.0001
        
    def reset(self) -> None:
        self.V_prev = 0.
        self.om_prev = 0.
        self.t_prev = 0.
        
    def compute_heading_control(
            self,
            state: TurtleBotState,
            goal: TurtleBotState
            ) -> TurtleBotControl:
        """Given a state and goal, returns a control to get to that heading.

        Args:
            state (TurtleBotState): current turtle bot state
            goal (TurtleBotState): desired turtle bot state

        Returns:
            TurtleBotControl: control to get towards desired state.
        """
        # Calculate angular velocity needed to correct heading error
        heading_error = wrap_angle(goal.theta - state.theta)
        angular_velocity = self.kp * heading_error
        
        # Update angular velocity in Turtlebot
        msg = TurtleBotControl()
        msg.omega = angular_velocity
        return msg
    
    def compute_trajectory_tracking_control(
            self, 
            state: TurtleBotState, 
            plan: TrajectoryPlan, 
            t: float
            ) -> TurtleBotControl:
        """ Compute control target using a trajectory tracking controller

        Args:
            state (TurtleBotState): current robot state
            plan (TrajectoryPlan): planned trajectory
            t (float): current timestep

        Returns:
            TurtleBotControl: control command
        """
        
        # Get dt
        dt = t - self.t_prev

        # Extract current state
        x, y, th = state.x, state.y, state.theta
        
        # Get the desired positions, velocities, and accelerations
        desired_state = plan.desired_state(t)
        x_d, y_d = desired_state.x, desired_state.y
        xd_d, yd_d = splev(t, plan.path_x_spline, der=1), splev(t, plan.path_y_spline, der=1)
        xdd_d, ydd_d = splev(t, plan.path_x_spline, der=2), splev(t, plan.path_y_spline, der=2)
        
        # Calculate actual xd and yd, ensuring that V > 0
        self.V_prev = self.V_PREV_THRES if np.abs(self.V_prev) <= self.V_PREV_THRES else self.V_prev
        xd = self.V_prev * np.cos(th)
        yd = self.V_prev * np.sin(th)
        
        # Calculate virtual inputs
        u = np.array([xdd_d + self.kpx * (x_d - x) + self.kdx * (xd_d - xd),
                      ydd_d + self.kpy * (y_d - y) + self.kdy * (yd_d - yd)])
        
        # Compute control inputs using linalg.solve.
        J = np.array([[np.cos(th), -self.V_prev * np.sin(th)],
                      [np.sin(th), self.V_prev * np.cos(th)]])
        inputs = np.linalg.solve(J, u)
        
        # Use inputs to set V and om
        V = self.V_prev + dt * inputs[0]
        om = inputs[1]

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om
        
        # Construct control output
        control = TurtleBotControl()
        control.omega = om
        control.v = V    
        
        return control    
        
    def compute_trajectory_plan(self,
            state: TurtleBotState,
            goal: TurtleBotState,
            occupancy: StochOccupancyGrid2D,
            resolution: float,
            horizon: float,
            ) -> T.Optional[TrajectoryPlan]:
        """ Compute a trajectory plan using A* and cubic spline fitting

        Args:
            state (TurtleBotState): state
            goal (TurtleBotState): goal
            occupancy (StochOccupancyGrid2D): occupancy
            resolution (float): resolution
            horizon (float): horizon

        Returns:
            T.Optional[TrajectoryPlan]:
        """
        def compute_smooth_plan(path, v_desired=0.15, spline_alpha=0.05) -> TrajectoryPlan:
            
            # Compute time stamps assuming constant velocity
            # distances = np.linalg.norm(np.diff(path), axis=1)
            # delta_ts = distances / v_desired
            # ts = np.zeros(len(path))
            # ts[1:] = np.cumsum(delta_ts)
            ts = [0]
            prev_point = path[0]
            for i in range(1, len(path)):
                dist = np.linalg.norm(path[i] - prev_point)
                ts.append(ts[-1] + dist/v_desired)
                prev_point = path[i]

            ts = np.asarray(ts)
            
            # Fit splines using time stamps, path, and spline params
            path_x_spline = splrep(ts, path[:, 0], s=spline_alpha)
            path_y_spline = splrep(ts, path[:, 1], s=spline_alpha)
            
            return TrajectoryPlan(
                path=path,
                path_x_spline=path_x_spline,
                path_y_spline=path_y_spline,
                duration=ts[-1],
                )
        
        astar = AStar((-horizon + state.x, -horizon + state.y), 
                      (horizon + state.x, horizon + state.y),
                      (state.x, state.y),
                      (goal.x, goal.y), 
                      occupancy, 
                      resolution=resolution)
        if not astar.solve() or len(astar.path) < 4:
            return None
        
        # Reset tracking control history 
        self.reset()
        smoothed_path = np.asarray(astar.path)
        return compute_smooth_plan(smoothed_path)
        


class AStar(object):
    """Represents a motion planning problem to be solved using A*"""
    
    DIRECTIONS = np.array([(-1, -1), (0, -1), (1, -1),
                           (-1,  0),          (1,  0),
                           (-1,  1), (0,  1), (1,  1)])
    
    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init                     
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        ########## Code starts here ##########
        x = np.asarray(x)
        return np.all(x >= self.statespace_lo) and np.all(x <= self.statespace_hi) and self.occupancy.is_free(x)
        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        ########## Code starts here ##########
        return np.linalg.norm(np.array(x1) - np.array(x2))
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        neighbors = []
        ########## Code starts here ##########
            
        for direction in self.DIRECTIONS:
            neighbor = self.snap_to_grid((x[0] + direction[0] * self.resolution, 
                                        x[1] + direction[1] * self.resolution))
            if self.is_free(neighbor):
                neighbors.append(neighbor)

        ########## Code ends here ##########
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        ########## Code starts here ##########

        # Go through potential nodes
        while self.open_set:
            curr_state = self.find_best_est_cost_through()
            if curr_state == self.x_goal:
                self.path = self.reconstruct_path()
                return True

            # Make sure that we don't explore the current state
            self.open_set.remove(curr_state)
            self.closed_set.add(curr_state)
            
            for neighbor in self.get_neighbors(curr_state):
                # If neighbor already explored, don't explore.
                if neighbor in self.closed_set:
                    continue

                tentative_cost_to_arrive = self.cost_to_arrive[curr_state] + self.distance(curr_state, neighbor)
                # Add neighbor as explorable, or skip if we already have a better path
                if neighbor not in self.open_set:
                    self.open_set.add(neighbor)
                elif tentative_cost_to_arrive > self.cost_to_arrive[neighbor]:
                    continue

                self.came_from[neighbor] = curr_state
                self.cost_to_arrive[neighbor] = tentative_cost_to_arrive
                self.est_cost_through[neighbor] = tentative_cost_to_arrive + self.distance(neighbor, self.x_goal)

        return False
        ########## Code ends here ##########
        
if __name__ == "__main__":
    rclpy.init()
    custom_navigator = CustomNavigator()
    rclpy.spin(custom_navigator)
    rclpy.shutdown()