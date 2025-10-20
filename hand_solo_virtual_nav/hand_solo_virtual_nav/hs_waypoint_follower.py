#!/usr/bin/env python3
import math
import json
from enum import Enum, auto
from collections import deque
from typing import Dict, Optional, List

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

# MoveIt (arm) action + msgs
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, Constraints, JointConstraint


# --- helper: build PoseStamped in 'map' ---
def make_pose(x: float, y: float, yaw: float) -> PoseStamped:
    ps = PoseStamped()
    ps.header.frame_id = 'map'
    ps.pose.position.x = float(x)
    ps.pose.position.y = float(y)
    half = 0.5 * yaw
    ps.pose.orientation.z = math.sin(half)
    ps.pose.orientation.w = math.cos(half)
    return ps

# States for continuous cycling
class Step(Enum):
    IDLE = auto()
    MOVING_TO_PICK = auto()
    PICKING = auto()            # run arm sequence [above->pick->carry]
    MOVING_TO_SHELF = auto()
    DROPPING = auto()           # run arm sequence [place->carry]
    RETURNING = auto()


class WarehouseNavigator(Node):
    """TODO node summary"""

    # Arm config 
    JOINT_NAMES: List[str] = ['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6']
    PLANNING_GROUP: str = 'tmr_arm'
    MOVE_ACTION: str = '/move_action'

    # Arm joint poses (radians)
    ARM_ABOVE_BOX = [0.0, 0.0, 1.57, 0.0, 1.57, 0.0]
    ARM_PICK      = [0.0, 0.40, 1.57, 0.0, 1.4, 0.0]
    ARM_CARRY     = [0.0, -0.70, 1.2, 0.0, 1.57, 0.0]
    ARM_PLACE     = [0.0, -0.17, 1.66, 0.0, 1.57, 0.0]

    def __init__(self):
        super().__init__('warehouse_nav')

        # Box picking poses, A = Big, B = Medium, C = Small 
        self.pick_map: Dict[str, PoseStamped] = {
            'A': make_pose(2.5,  -0.7, 0.0),
            'B': make_pose(2.5,  0.0, 0.0),
            'C': make_pose(2.5,  0.7, 0.0),
        }

        # 
        self.shelf_pose: PoseStamped = make_pose(30.9, -5.0, 3*math.pi / 2)   
        self.staging_pose: PoseStamped = make_pose(0.00, 0.00, 0.0)

        # Quality of service profile to recieve all messages reliably,
        # keep last 10 messages in memory and discard older messages
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Create subscriber for box information ('hmi/unified_status' topic)
        self.subscription = self.create_subscription(String, 'hmi/unified_status', self._hmi_cb, qos)

        # Create an ActionClient for the NavigateToPose action
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().info('Waiting for Nav2 action server...')
        self.nav_client.wait_for_server()
        self.get_logger().info('Nav2 ready.')

        # Create an ActionClient for the MoveGroup action
        self.arm_client = ActionClient(self, MoveGroup, self.MOVE_ACTION)
        self.get_logger().info('Waiting for MoveIt MoveGroup action server...')
        self.arm_client.wait_for_server()
        self.get_logger().info('MoveGroup ready.')

        # Initialise state machine to idle
        self.step: Step = Step.IDLE
        # Double ended que to store next job, discard finished jobs 
        self.jobs: deque[str] = deque()
        # Current job
        self.current_key: Optional[str] = None

        # Async bookkeeping
        self._nav_result_future = None
        self._nav_handle = None

        self._arm_seq: Optional[List[List[float]]] = None   # list of joint targets
        self._arm_index: int = 0
        self._arm_result_future = None
        self._arm_handle = None

        # Tick the state machine 20 times per second (20 Hz)
        self.create_timer(0.05, self._tick)

        # Drive to staging at startup
        self._start_nav(self.staging_pose)
        self.step = Step.RETURNING

    # -------------------- HMI --------------------
    def _hmi_cb(self, msg: String):
        """Expect: {"box":{"location":"A"|"B"|"C"}}."""
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"JSON parse error: {e}; raw={msg.data}")
            return

        box = data.get('box')
        if not isinstance(box, dict):
            self.get_logger().warn(f"Missing 'box': {data}")
            return

        loc_raw = box.get('location', '')
        if not isinstance(loc_raw, str):
            self.get_logger().warn(f"Bad 'location' type: {type(loc_raw)}")
            return

        key = loc_raw.lstrip(':').strip().upper()
        if key not in self.pick_map:
            self.get_logger().warn(f"Unknown location '{key}'. Use one of {list(self.pick_map.keys())}")
            return

        # Latest-wins queue, most recent box position 
        self.jobs.clear()
        self.jobs.append(key)
        self.get_logger().info(f"Queued: {key}")

    # -------------------- TICK --------------------
    def _tick(self):
        # --- check nav completion ---
        if self._nav_result_future is not None and self._nav_result_future.done():
            nav_res = self._nav_result_future.result()
            self._nav_result_future = None
            status = getattr(nav_res, 'status', None)
            if status == 4:  # SUCCEEDED
                self._on_nav_success()
            else:
                self.get_logger().error(f"Nav2 finished with status {status}; aborting job.")
                self._reset_to_idle()
            return

        # --- check arm completion / progress ---
        if self._arm_result_future is not None and self._arm_result_future.done():
            arm_res = self._arm_result_future.result()
            self._arm_result_future = None
            astatus = getattr(arm_res, 'status', None)
            if astatus == 4:  # SUCCEEDED
                self._arm_index += 1
                if self._arm_seq and self._arm_index < len(self._arm_seq):
                    # send next joint target
                    self._start_arm_goal(self._arm_seq[self._arm_index])
                else:
                    # arm sequence complete -> advance main SM
                    self._arm_seq = None
                    self._on_arm_sequence_complete()
            else:
                self.get_logger().error(f"Arm move status {astatus}; aborting job.")
                self._reset_to_idle()
            return

        # --- drive idle -> start job ---
        if self.step == Step.IDLE and self.jobs:
            self.current_key = self.jobs.popleft()
            self.get_logger().info(f"Starting job {self.current_key}")
            self._start_nav(self.pick_map[self.current_key])
            self.step = Step.MOVING_TO_PICK

        # PICKING / DROPPING phases progress via arm futures above

    # -------------------- SM helpers --------------------
    def _on_nav_success(self):
        if self.step == Step.MOVING_TO_PICK:
            # start arm sequence: above -> pick -> carry
            self.get_logger().info(f"At pick {self.current_key}; running arm sequence [above, pick, carry].")
            self._start_arm_sequence([
                self.ARM_ABOVE_BOX,
                self.ARM_PICK,
                self.ARM_CARRY,
            ])
            self.step = Step.PICKING

        elif self.step == Step.MOVING_TO_SHELF:
            # start arm sequence at shelf: place -> carry
            self.get_logger().info("At shelf; running arm sequence [place, carry].")
            self._start_arm_sequence([
                self.ARM_PLACE,
                self.ARM_CARRY,
            ])
            self.step = Step.DROPPING

        elif self.step == Step.RETURNING:
            self.get_logger().info("Back at staging; cycle complete.")
            self.step = Step.IDLE
            self.current_key = None

    def _on_arm_sequence_complete(self):
        if self.step == Step.PICKING:
            # now drive to shelf
            self._start_nav(self.shelf_pose)
            self.step = Step.MOVING_TO_SHELF
        elif self.step == Step.DROPPING:
            # now return to staging
            self._start_nav(self.staging_pose)
            self.step = Step.RETURNING

    def _reset_to_idle(self):
        self._nav_handle = None
        self._nav_result_future = None
        self._arm_seq = None
        self._arm_index = 0
        self._arm_handle = None
        self._arm_result_future = None
        self.current_key = None
        self.step = Step.IDLE
        self.jobs.clear() 

    # -------------------- NAV wrappers (async) --------------------
    def _start_nav(self, pose: PoseStamped):
        pose.header.stamp = self.get_clock().now().to_msg()
        goal = NavigateToPose.Goal(); goal.pose = pose

        def fb(fbmsg):
            try:
                d = fbmsg.feedback.distance_remaining
                self.get_logger().info(f"[Nav] dist {d:.2f} m")
            except Exception:
                pass

        send_future = self.nav_client.send_goal_async(goal, feedback_callback=fb)

        def _on_goal_ready(fut):
            handle = fut.result()
            if not handle or not handle.accepted:
                self.get_logger().error("Nav goal rejected.")
                self._reset_to_idle()
                return
            self._nav_handle = handle
            self._nav_result_future = handle.get_result_async()

        send_future.add_done_callback(_on_goal_ready)

    # -------------------- ARM wrappers (async) --------------------
    def _arm_goal_from_joints(self, joints: List[float]) -> MoveGroup.Goal:
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
            jc.position = float(val)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            cs.joint_constraints.append(jc)

        req.goal_constraints.append(cs)
        goal.request = req
        goal.planning_options.plan_only = False  # plan + execute
        return goal

    def _start_arm_sequence(self, joint_list: List[List[float]]):
        """Begin an async sequence of arm joint targets."""
        self._arm_seq = list(joint_list)
        self._arm_index = 0
        self._start_arm_goal(self._arm_seq[self._arm_index])

    def _start_arm_goal(self, joints: List[float]):
        goal = self._arm_goal_from_joints(joints)
        send_fut = self.arm_client.send_goal_async(goal)

        def _on_arm_goal_ready(fut):
            handle = fut.result()
            if not handle or not handle.accepted:
                self.get_logger().error("Arm goal rejected.")
                self._reset_to_idle()
                return
            self._arm_handle = handle
            self._arm_result_future = handle.get_result_async()

        send_fut.add_done_callback(_on_arm_goal_ready)


def main():
    rclpy.init()
    node = WarehouseNavigator()
    try:
        node.get_logger().info("Ready. Send jobs on 'hmi/unified_status' (A/B/C).")
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
