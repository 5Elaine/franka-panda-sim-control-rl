# Third-party notices

This repository depends on, but does not vendor, the following projects:

- ROS 2, MoveIt 2, Gazebo, and ros2_control.
- Franka ROS 2 / `franka_description` robot description resources.
- MuJoCo and the MuJoCo Menagerie Franka Emika Panda model.
- Gymnasium, Stable-Baselines3, CVXPY, OSQP, NumPy, and Matplotlib.

The generated Panda URDF/Xacro snapshots under `ros2_ws/src/panda_gazebo_bringup/urdf/` identify their `franka_description` source in the file header and reference external mesh resources at runtime. Third-party components remain subject to their own licenses and notices.

No repository-wide open-source license is asserted here. Contact the repository owner before reusing project-authored code.
