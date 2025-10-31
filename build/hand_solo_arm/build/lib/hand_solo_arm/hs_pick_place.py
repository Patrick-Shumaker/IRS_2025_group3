#!/usr/bin/env python3
import time
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, Constraints, JointConstraint

class HSWaypointRunner(Node):
    """Minimal: define a waypoint, execute, wait; then define the next one."""

    JOINT_NAMES: List[str] = ['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6']
    PLANNING_GROUP: str = 'tmr_arm'
    ACTION_NAME: str = '/move_action'  # keep identical to your setup

    def __init__(self):
        super().__init__('hs_waypoint_runner')
        self.client = ActionClient(self, MoveGroup, self.ACTION_NAME)
        self.get_logger().info('Waiting for MoveGroup action server...')
        self.client.wait_for_server()
        self.get_logger().info('MoveGroup action server ready.')

    # --- helpers --------------------------------------------------------------
    def _goal_from_joints(self, joints: List[float]) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = self.PLANNING_GROUP
        req.num_planning_attempts = 10
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = 0.5
        req.max_acceleration_scaling_factor = 0.5

        cs = Constraints()
        for name, val in zip(self.JOINT_NAMES, joints):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = val
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            cs.joint_constraints.append(jc)

        req.goal_constraints.append(cs)
        goal.request = req
        goal.planning_options.plan_only = False  # plan + execute
        return goal

    def _send_and_wait(self, joints: List[float]) -> bool:
        goal = self._goal_from_joints(joints)
        send_fut = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut)
        handle = send_fut.result()
        if not handle or not handle.accepted:
            self.get_logger().error('Goal rejected.')
            return False
        res_fut = handle.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut)
        res = res_fut.result()
        return bool(res and res.status == 4)  # 4 = STATUS_SUCCEEDED

    # --- script-style sequence (no arrays, no names) --------------------------
    def run(self):
    
        # Above box
        if not self._send_and_wait([0.0, 0.0, 1.57, 0.0, 1.57, 0.0]): return
        time.sleep(5.0)

        # Pick pose
        if not self._send_and_wait([0.0, 0.42, 1.57, 0.0, 1.57, 0.0]): return
        time.sleep(5)

        # Carry pose
        if not self._send_and_wait([0.0, 0.35, 0.44, 0.0, 1.57, 0.0]): return
        time.sleep(1.5)

        # Place pose
        if not self._send_and_wait([0.0, 0.0, 1.34, 0.0, 1.57, 0.0]): return
        time.sleep(0.5)

        self.get_logger().info('Sequence complete ✅')


def main(args=None):
    rclpy.init(args=args)
    node = HSWaypointRunner()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
