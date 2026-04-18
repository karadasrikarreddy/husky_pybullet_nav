"""
Generates a Clearpath Husky-like URDF for use with PyBullet.
Husky specs (approximate):
  - Body: 0.87m L x 0.58m W x 0.26m H
  - Wheel radius: 0.1651m, width: 0.1143m
  - Track width (centre-to-centre): 0.5708m
  - Wheelbase (front-to-rear axle): 0.5120m
"""


def get_husky_urdf() -> str:
    return """<?xml version="1.0"?>
<robot name="husky">

  <!-- ======== MATERIALS ======== -->
  <material name="dark_grey"><color rgba="0.2 0.2 0.2 1"/></material>
  <material name="grey"><color rgba="0.5 0.5 0.5 1"/></material>
  <material name="black"><color rgba="0.08 0.08 0.08 1"/></material>
  <material name="yellow"><color rgba="0.8 0.7 0.1 1"/></material>

  <!-- ======== BASE LINK ======== -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0.18" rpy="0 0 0"/>
      <geometry><box size="0.87 0.58 0.26"/></geometry>
      <material name="grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.18" rpy="0 0 0"/>
      <geometry><box size="0.87 0.58 0.26"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.18" rpy="0 0 0"/>
      <mass value="46.034"/>
      <inertia ixx="4.4684" ixy="0.0" ixz="0.0"
               iyy="5.5789" iyz="0.0" izz="3.0657"/>
    </inertial>
  </link>

  <!-- ======== TOP PLATE ======== -->
  <joint name="top_plate_joint" type="fixed">
    <parent link="base_link"/>
    <child link="top_plate_link"/>
    <origin xyz="-0.02 0 0.321" rpy="0 0 0"/>
  </joint>
  <link name="top_plate_link">
    <visual>
      <geometry><box size="0.78 0.46 0.008"/></geometry>
      <material name="dark_grey"/>
    </visual>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- ======== FRONT BUMPER ======== -->
  <joint name="front_bumper_joint" type="fixed">
    <parent link="base_link"/>
    <child link="front_bumper_link"/>
    <origin xyz="0.48 0 0.12" rpy="0 0 0"/>
  </joint>
  <link name="front_bumper_link">
    <visual>
      <geometry><box size="0.06 0.62 0.12"/></geometry>
      <material name="dark_grey"/>
    </visual>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- ======== REAR BUMPER ======== -->
  <joint name="rear_bumper_joint" type="fixed">
    <parent link="base_link"/>
    <child link="rear_bumper_link"/>
    <origin xyz="-0.48 0 0.12" rpy="0 0 0"/>
  </joint>
  <link name="rear_bumper_link">
    <visual>
      <geometry><box size="0.06 0.62 0.12"/></geometry>
      <material name="dark_grey"/>
    </visual>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- ======== FRONT LEFT WHEEL ======== -->
  <joint name="front_left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="front_left_wheel_link"/>
    <origin xyz="0.256 0.2854 0.0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics damping="0.1" friction="0.2"/>
  </joint>
  <link name="front_left_wheel_link">
    <visual>
      <origin xyz="0 0.0285 0" rpy="1.5707963 0 0"/>
      <geometry><cylinder radius="0.1651" length="0.1143"/></geometry>
      <material name="black"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="1.5707963 0 0"/>
      <geometry><cylinder radius="0.1651" length="0.1143"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="2.637"/>
      <inertia ixx="0.02467" ixy="0" ixz="0"
               iyy="0.04411" iyz="0" izz="0.02467"/>
    </inertial>
  </link>

  <!-- ======== FRONT RIGHT WHEEL ======== -->
  <joint name="front_right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="front_right_wheel_link"/>
    <origin xyz="0.256 -0.2854 0.0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics damping="0.1" friction="0.2"/>
  </joint>
  <link name="front_right_wheel_link">
    <visual>
      <origin xyz="0 -0.0285 0" rpy="1.5707963 0 0"/>
      <geometry><cylinder radius="0.1651" length="0.1143"/></geometry>
      <material name="black"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="1.5707963 0 0"/>
      <geometry><cylinder radius="0.1651" length="0.1143"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="2.637"/>
      <inertia ixx="0.02467" ixy="0" ixz="0"
               iyy="0.04411" iyz="0" izz="0.02467"/>
    </inertial>
  </link>

  <!-- ======== REAR LEFT WHEEL ======== -->
  <joint name="rear_left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="rear_left_wheel_link"/>
    <origin xyz="-0.256 0.2854 0.0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics damping="0.1" friction="0.2"/>
  </joint>
  <link name="rear_left_wheel_link">
    <visual>
      <origin xyz="0 0.0285 0" rpy="1.5707963 0 0"/>
      <geometry><cylinder radius="0.1651" length="0.1143"/></geometry>
      <material name="black"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="1.5707963 0 0"/>
      <geometry><cylinder radius="0.1651" length="0.1143"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="2.637"/>
      <inertia ixx="0.02467" ixy="0" ixz="0"
               iyy="0.04411" iyz="0" izz="0.02467"/>
    </inertial>
  </link>

  <!-- ======== REAR RIGHT WHEEL ======== -->
  <joint name="rear_right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="rear_right_wheel_link"/>
    <origin xyz="-0.256 -0.2854 0.0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics damping="0.1" friction="0.2"/>
  </joint>
  <link name="rear_right_wheel_link">
    <visual>
      <origin xyz="0 -0.0285 0" rpy="1.5707963 0 0"/>
      <geometry><cylinder radius="0.1651" length="0.1143"/></geometry>
      <material name="black"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="1.5707963 0 0"/>
      <geometry><cylinder radius="0.1651" length="0.1143"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="2.637"/>
      <inertia ixx="0.02467" ixy="0" ixz="0"
               iyy="0.04411" iyz="0" izz="0.02467"/>
    </inertial>
  </link>

</robot>
"""
