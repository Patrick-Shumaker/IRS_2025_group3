#!/usr/bin/env python3
import math
import json
from enum import Enum, auto
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
    DROPPED = auto()           # run arm sequence [place->carry]
    RETURNING = auto()


class WarehouseNavigator(Node):
    """State machine for autonmous operation of an omron mobile robot to pick boxes off conveyor
    at specific positions and place on shelf while avoiding dynamic/static obstacles continuously.

    Built for the simulation environment by Collabrotiive Robotics Lab"""

    # Arm config 
    JOINT_NAMES: List[str] = ['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6']
    PLANNING_GROUP: str = 'tmr_arm'
    MOVE_ACTION: str = '/move_action'

    # Arm joint poses (radians)
    ARM_ABOVE_BOX = [0.0, 0.0, 1.57, 0.0, 1.57, 0.0]
    ARM_PICK      = [0.0, 0.35, 1.57, 0.0, 1.57, 0.0]
    ARM_CARRY     = [0.0, -0.70, 1.7, 0.0, 1.57, 0.0]
    ARM_PLACE     = [0.0, -0.2, 1.57, 0.0, 1.57, 0.0]

    def __init__(self):
        super().__init__('warehouse_nav')

        # Box picking positions, A = Big, B = Medium, C = Small 
        self.pick_map: Dict[str, PoseStamped] = {
            'A': make_pose(2.5,  -0.7, 0.0),
            'B': make_pose(2.5,  0.0, 0.0),
            'C': make_pose(2.5,  0.7, 0.0),
        }

        # 
        self.shelf_pose: PoseStamped = make_pose(30.9, -5.3, 3*math.pi / 2)   
        self.staging_pose: PoseStamped = make_pose(0.00, 0.00, 0.0)

        # Quality of service profile to recieve all messages reliably,
        # keep last 10 messages in memory and discard older messages
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Create subscriber for box information ('hmi/unified_status' topic)
        self.subscription = self.create_subscription(String, 'hmi/unified_status', self.hmi_cb, qos)

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

       
        # Initialise current job
        self.current_job = ''

        # Async bookkeeping
        self._nav_result_future = None
        self._nav_handle = None

        self._arm_seq: Optional[List[List[float]]] = None   # list of joint targets
        self._arm_index: int = 0
        self._arm_result_future = None
        self._arm_handle = None

        # Tick the state machine 20 times per second (20 Hz)
        self.create_timer(0.05, self.tick)

        # Drive to staging at startup
        self.start_nav(self.staging_pose)
        self.step = Step.RETURNING

    # -------------------- HMI --------------------
    def hmi_cb(self, msg: String):
        """Expect: {"box":{"location":"A"|"B"|"C"}}."""
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"JSON parse error: {e}; raw={msg.data}")
            return

        box = data.get('box')
        
        loc_raw = box.get('location', '')
        if not isinstance(loc_raw, str):
            self.get_logger().warn(f"Bad 'location' type: {type(loc_raw)}")
            return

        job = loc_raw.lstrip(':').strip().upper()
        if job not in self.pick_map:
            self.get_logger().warn(f"Unknown location or no box present")
            return

        # If no job, add box location to queue
        if self.current_job == '':
            self.current_job = job
            self.get_logger().info(f"Queued: {job}")

    # tick func
    def tick(self):
        # check for nav completion 
        if self._nav_result_future is not None and self._nav_result_future.done():
            nav_res = self._nav_result_future.result()
            self._nav_result_future = None
            status = getattr(nav_res, 'status', None)
            if status == 4:  # SUCCEEDED
                # Progress cycle
                self.on_nav_success()
            else:
                self.get_logger().error(f"Nav2 finished with status {status}; aborting job.")
                self.reset_to_idle()
            return

        # check arm sequence completion/progress 
        if self._arm_result_future is not None and self._arm_result_future.done():
            arm_res = self._arm_result_future.result()
            self._arm_result_future = None
            astatus = getattr(arm_res, 'status', None)
            if astatus == 4:  # SUCCEEDED
                # Increment arm index to progress through sequence
                self._arm_index += 1
                if self._arm_seq and self._arm_index < len(self._arm_seq):
                    # send next joint target
                    self.start_arm_goal(self._arm_seq[self._arm_index])
                else:
                    # arm sequence complete -> advance main SM
                    self._arm_seq = None
                    self.arm_seq_finished()
            else:
                # on failure reset to idle
                self.get_logger().error(f"Arm move status {astatus}; aborting job.")
                self.reset_to_idle()
            return

        # start job from idle state
        if self.step == Step.IDLE and self.current_job != '':
            self.get_logger().info(f"Starting job {self.current_job}")
            self.start_nav(self.pick_map[self.current_job])
            self.step = Step.MOVING_TO_PICK



    # SM helpers  

    # Progress from successful nav based on current state
    def on_nav_success(self):
        if self.step == Step.MOVING_TO_PICK:
            # start arm sequence above -> pick -> carry
            self.get_logger().info(f"At pick position {self.current_job}; running arm sequence [above, pick, carry].")
            self.start_arm_seq([
                self.ARM_ABOVE_BOX,
                self.ARM_PICK,
                self.ARM_CARRY,
            ])
            self.step = Step.PICKING

        elif self.step == Step.MOVING_TO_SHELF:
            # start arm sequence at shelf: place -> carry
            self.get_logger().info("At shelf; running arm sequence [place, carry].")
            self.start_arm_seq([
                self.ARM_PLACE,
                self.ARM_CARRY,
            ])
            self.step = Step.DROPPED

        elif self.step == Step.RETURNING:
            self.get_logger().info("Back at staging; cycle complete.")
            self.step = Step.IDLE
            # Clear job on completion to enable next job queue
            self.current_job = ''

    def arm_seq_finished(self):
        if self.step == Step.PICKING:
            # Drive to shelf
            self.start_nav(self.shelf_pose)
            self.step = Step.MOVING_TO_SHELF
        elif self.step == Step.DROPPED:
            # Return to staging
            self.start_nav(self.staging_pose)
            self.step = Step.RETURNING

    # When failure occurs in nav/arm reset to idle and try again
    def reset_to_idle(self):
        self._nav_handle = None
        self._nav_result_future = None
        self._arm_seq = None
        self._arm_index = 0
        self._arm_handle = None
        self._arm_result_future = None
        self.current_job = ''
        self.step = Step.IDLE


    # NAV wrappers (async) 
    def start_nav(self, pose: PoseStamped):
        pose.header.stamp = self.get_clock().now().to_msg()
        goal = NavigateToPose.Goal(); goal.pose = pose

        def fb(fbmsg):
            try:
                d = fbmsg.feedback.distance_remaining
                self.get_logger().info(f"[Nav] dist {d:.2f} m")
            except Exception:
                pass

        send_future = self.nav_client.send_goal_async(goal, feedback_callback=fb)

        def goal_ready(fut):
            handle = fut.result()
            if not handle or not handle.accepted:
                self.get_logger().error("Nav goal rejected.")
                self.reset_to_idle()
                return
            self._nav_handle = handle
            self._nav_result_future = handle.get_result_async()

        send_future.add_done_callback(goal_ready)

    # ARM wrappers (async) 
    def arm_goal(self, joints: List[float]) -> MoveGroup.Goal:
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

    def start_arm_seq(self, joint_list: List[List[float]]):
        #Begin a sequence of arm joint targets.
        self._arm_seq = list(joint_list)
        # Initialise arm sequence index
        self._arm_index = 0
        self.start_arm_goal(self._arm_seq[self._arm_index])

    def start_arm_goal(self, joints: List[float]):
        goal = self.arm_goal(joints)
        send_fut = self.arm_client.send_goal_async(goal)

        def arm_goal_ready(fut):
            handle = fut.result()
            if not handle or not handle.accepted:
                self.get_logger().error("Arm goal rejected.")
                self.reset_to_idle()
                return
            self._arm_handle = handle
            self._arm_result_future = handle.get_result_async()

        send_fut.add_done_callback(arm_goal_ready)


def main():
    rclpy.init()
    node = WarehouseNavigator()
    try:
        node.get_logger().info("Ready. Send jobs on 'hmi/unified_status' (A/B/C).")
        rclpy.spin(node)
    except jobboardInterrupt:
        node.get_logger().info("Shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
