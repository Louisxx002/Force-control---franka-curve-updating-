"""FR3 planning services only: no hardware driver or execution capability."""
from pathlib import Path
import xacro
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    description = Path(get_package_share_directory('franka_description'))
    config = Path(get_package_share_directory('franka_fr3_moveit_config')) / 'config'
    def model(suffix):
        return xacro.process_file(str(description / 'robots/fr3' / ('fr3.' + suffix + '.xacro')),
                                  mappings={'hand': 'false', 'ee_id': 'none',
                                            'ros2_control': 'false'}).toxml()
    def settings(name):
        return yaml.safe_load((config / name).read_text())
    ompl = settings('ompl_planning.yaml')
    ompl.update(planning_plugins=['ompl_interface/OMPLPlanner'],
                request_adapters=['default_planning_request_adapters/ResolveConstraintFrames',
                                  'default_planning_request_adapters/ValidateWorkspaceBounds',
                                  'default_planning_request_adapters/CheckStartStateBounds',
                                  'default_planning_request_adapters/CheckStartStateCollision'],
                response_adapters=['default_planning_response_adapters/AddTimeOptimalParameterization',
                                   'default_planning_response_adapters/ValidateSolution'])
    urdf = model('urdf')
    return LaunchDescription([Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        namespace='autowipe', parameters=[{'robot_description': urdf}], output='log'), Node(
        package='moveit_ros_move_group', executable='move_group',
        namespace='autowipe', output='screen', parameters=[{
            'robot_description': urdf,
            'robot_description_semantic': model('srdf'),
            'robot_description_kinematics': settings('kinematics.yaml'),
            'robot_description_planning': settings('fr3_joint_limits.yaml'),
            'planning_pipelines': ['ompl'], 'default_planning_pipeline': 'ompl', 'ompl': ompl,
            'allow_trajectory_execution': False,
            'disable_capabilities': 'move_group/MoveGroupExecuteTrajectoryAction move_group/MoveGroupMoveAction',
            'publish_planning_scene': True,
        }])])
