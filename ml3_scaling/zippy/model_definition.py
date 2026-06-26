
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pydrake.all import (
    AbstractValue, AddMultibodyPlantSceneGraph, AngleAxis, ContactResults,
    DiagramBuilder, DiscreteContactApproximation, LeafSystem, List, Parser,
    Quaternion, RollPitchYaw, ScopedName,
)




# --- portable paths (auto-cleaned) ---
import os as _os
_PKG_DIR = _os.path.dirname(_os.path.abspath(__file__))
_REPO_DIR = _os.path.normpath(_os.path.join(_PKG_DIR, "..", ".."))
_MESH_DIR = _os.path.join(_REPO_DIR, "meshes", _os.path.basename(_PKG_DIR))
_DATA_DIR = _os.path.join(_REPO_DIR, "data")
# --- end portable paths ---

'''---------------------------------------Set up walker model--------------------------------------'''



def create_walker_urdf(scale=1, ground_friction=0.5, feet_friction=0.3): 

    def s(v):  # scale vector
        return ' '.join(str(scale * float(x)) for x in v.split())

    def s_mass(m):  # scale mass
        return str((scale ** 3) * float(m))

    def s_inertia(i):  # scale inertia
        return str((scale ** 5) * float(i))

    def mesh_res_hint(val):  # scale mesh resolution
        return str(scale * float(val))

    return f"""
       <?xml version="1.0" ?>
<!-- Generated using onshape-to-robot -->
<!-- Onshape https://cad.onshape.com/documents/d4162f5f6a620a91d67bd0f0/w/d866279445751daabbe0c95e/e/13afda9ab73add5250d40401 -->
<robot name="walker">

  <link name="ground">
      <visual>
      <origin xyz="{s('0 0 -0.25')}" rpy="0 0 0" />
      <geometry>
          <box size="{s('8 8 0.5')}" />
      </geometry>
      <material>
          <color rgba="0.3 0.38 0.46 1.0" />
          <!--color rgba="0.93 .74 .4 1" -->
      </material>
      </visual>
      <collision>
      <origin xyz="{s('0 0 -0.25')}" rpy="0 0 0" />
      <geometry>
          <box size="{s('8 8 0.5')}" />
      </geometry>
          <drake:proximity_properties>
          <drake:compliant_hydroelastic/>
          <drake:hydroelastic_modulus value="{s('5e7')}"/>
          <drake:mu_dynamic value="{ground_friction}"/>
          <drake:mu_static value="{ground_friction}"/>
          <drake:mesh_resolution_hint value="{0.00157}"/>
          </drake:proximity_properties>
      </collision>
  </link>

  <joint name="ground_weld" type="fixed">
      <parent link="world" />
      <child link="ground" />
  </joint>

    <!-- Link left_leg -->
  <link name="left_leg">
    <inertial>
      <origin xyz="{s('-0.0138911 0.00178604 -0.0194523')}" rpy="0 0 0"/>
      <mass value="{s_mass('0.00955')}"/>
      <inertia ixx="{s_inertia('1.03336e-06')}" 
      ixy="{s_inertia('-3.91222e-07')}" 
      ixz="{s_inertia('-3.59315e-07')}" 
      iyy="{s_inertia('2.45205e-06')}" 
      iyz="{s_inertia('-8.4632e-09')}" 
      izz="{s_inertia('2.17626e-06')}"/>
    </inertial>

    <!-- Part left_leg -->
    <visual>
      <origin xyz="{s('-1.73472e-18 0.0139 0')}" rpy="0 -0 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/left_leg.obj" scale = "{scale} {scale} {scale}"/>
      </geometry>
      <material name="left_leg_material">
        <color rgba="0.8 0.8 0.8 1.0"/>
      </material>
    </visual>
    <!-- 
    <collision>
      <origin xyz="{s('-1.73472e-18 0.0139 0')}" rpy="0 -0 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/left_leg.obj" scale = "{scale} {scale} {scale}"/>
      </geometry>
    </collision>
    -->
  
    <!-- Part r_hand_weight -->
    <visual>
      <origin xyz="{s('-1.73472e-18 0.0139 0')}" rpy="0 -0 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/r_hand_weight.obj" scale = "{scale} {scale} {scale}"/>
      </geometry>
      <material name="r_hand_weight_material">
        <color rgba="0.8 0.8 0.8 1.0"/>
      </material>
    </visual>
    <!-- 
    <collision>
      <origin xyz="{s('-1.73472e-18 0.0139 0')}" rpy="0 -0 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/r_hand_weight.obj" scale = "{scale} {scale} {scale}"/>
      </geometry>
    </collision>
    -->

    <!-- Part left_foot -->
    <visual>
      <origin xyz="{s('-0.0304044 -0.012563 0')}" rpy="0 -0 3.14159"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/left_foot.obj" scale = "{scale} {scale} {scale}"/>
      </geometry>
      <material name="left_foot_material">
        <color rgba="0.6 0.68 0.75 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="{s('-0.0304044 -0.012563 0')}" rpy="0 -0 3.14159"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/left_foot.obj" scale = "{scale} {scale} {scale}"/>
      </geometry>
      <drake:proximity_properties>
        <drake:rigid_hydroelastic/>
        <drake:mu_dynamic value="{feet_friction}"/>
        <drake:mu_static value="{feet_friction}"/>
        <!--drake:hydroelastic_modulus value="{s('5.0e7')}"-->
        <drake:mesh_resolution_hint value="{0.00157}"/>
      </drake:proximity_properties>
    </collision>

    <!-- Part r_arm -->
    <visual>
      <origin xyz="{s('-1.73472e-18 0.0139 0')}" rpy="0 -0 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/r_arm.obj" scale = "{scale} {scale} {scale}"/>
      </geometry>
      <material name="r_arm_material">
        <color rgba="0.8 0.8 0.8 1.0"/>
      </material>
    </visual>
    <!-- 
    <collision>
      <origin xyz="{s('-1.73472e-18 0.0139 0')}" rpy="0 -0 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/r_arm.obj" scale = "{scale} {scale} {scale}"/>
      </geometry>
    </collision>
    -->
  </link>

  <!-- Link left_hardstop_a -->
  <link name="left_hardstop_a">
    <inertial>
      <origin xyz="{s('0.00213371 -0.0014257 -0.0015')}" rpy="0 0 0"/>
      <mass value="{s_mass('0.0016')}"/>
      <inertia ixx="{s_inertia('4.72279e-09')}" 
      ixy="{s_inertia('-1.27531e-09')}" 
      ixz="{s_inertia('3.87688e-25')}" 
      iyy="{s_inertia('3.66629e-09')}" 
      iyz="{s_inertia('3.90452e-25')}" 
      izz="{s_inertia('5.98908e-09')}"/>
    </inertial>
    <!-- Part left_hardstop_a -->
    <visual>
      <origin xyz="{s('0.015691 -0.00858911 -0.0173272')}" rpy="1.5708 1.5708 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/left_hardstop_a.obj" scale = "{scale} {scale} {scale}"/>
      </geometry>
      <material name="left_hardstop_a_material">
        <color rgba="0.8 0.8 0.8 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="{s('0.015691 -0.00858911 -0.0173272')}" rpy="1.5708 1.5708 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/left_hardstop_a.obj" scale = "{scale} {scale} {scale}"/>
      </geometry>
      <drake:proximity_properties>
        <drake:rigid_hydroelastic/>
        <drake:mu_dynamic value="0.5"/>
        <drake:mu_static value="0.5"/>
        <!--drake:hydroelastic_modulus value="{s('5.0e7')}"-->
        <drake:mesh_resolution_hint value="{0.00157}"/>
      </drake:proximity_properties>
    </collision>
  </link>


  <!-- Joint from left_leg to left_hardstop_a -->
  <joint name="leftHSa" type="fixed">
    <origin xyz="{s('-0.0173272 -0.00179095 -0.00858911')}" rpy="-1.5708 -0 1.5708"/>
    <parent link="left_leg"/>
    <child link="left_hardstop_a"/>
    <axis xyz="0 0 1"/>
    <limit effort="10" velocity="10" lower="-1" upper="1"/>
  </joint>


  <!-- Link left_hardstop_b -->
  <link name="left_hardstop_b">
    <inertial>
      <origin xyz="{s('-0.00213371 -0.0014257 -0.0015')}" rpy="0 0 0"/>
      <mass value="{s_mass('0.0016')}"/>
      <inertia ixx="{s_inertia('4.72279e-09')}" 
      ixy="{s_inertia('1.27531e-09')}" 
      ixz="{s_inertia('3.87688e-25')}" 
      iyy="{s_inertia('3.66629e-09')}" 
      iyz="{s_inertia('-3.90452e-25')}" 
      izz="{s_inertia('5.98908e-09')}"/>
    </inertial>
    <!-- Part left_hardstop_b -->
    <visual>
      <origin xyz="{s('0.015691 -0.00858911 -0.0173272')}" rpy="1.5708 1.5708 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/left_hardstop_b.obj" scale = "{scale} {scale} {scale}"/>
      </geometry>
      <material name="left_hardstop_b_material">
        <color rgba="0.8 0.8 0.8 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="{s('0.015691 -0.00858911 -0.0173272')}" rpy="1.5708 1.5708 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/left_hardstop_b.obj" scale = "{scale} {scale} {scale}"/>
      </geometry>
      <drake:proximity_properties>
        <drake:rigid_hydroelastic/>
        <drake:mu_dynamic value="0.5"/>
        <drake:mu_static value="0.5"/>
        <!--drake:hydroelastic_modulus value="{s('5.0e7')}"-->
        <drake:mesh_resolution_hint value="{0.00157}"/>
      </drake:proximity_properties>
    </collision>
  </link>


  <!-- Joint from left_leg to left_hardstop_b -->
  <joint name="leftHSb" type="fixed">
    <origin xyz="{s('-0.0173272 -0.00179095 -0.00858911')}" rpy="-1.5708 -0 1.5708"/>
    <parent link="left_leg"/>
    <child link="left_hardstop_b"/>
    <axis xyz="0 0 1"/>
    <limit effort="10" velocity="10" lower="-1" upper="1"/>
  </joint>


  <!-- Link right_leg -->
  <link name="right_leg">
    <inertial>
      <origin xyz="{s('0.00333367 -0.0107781 0.00171679')}" rpy="0 0 0"/>
      <mass value="{s_mass('0.01113')}"/>
      <inertia ixx="{s_inertia('2.61349e-06')}" 
      ixy="{s_inertia('-4.90837e-11')}" 
      ixz="{s_inertia('4.11837e-07')}" 
      iyy="{s_inertia('2.26772e-06')}" 
      iyz="{s_inertia('3.43082e-07')}" 
      izz="{s_inertia('1.1141e-06')}"/>
    </inertial>

    <!-- Part right_foot -->
    <visual>
      <origin xyz="{s('0.015691 0.00858911 0.0172522')}" rpy="-1.5708 -1.5708 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/right_foot.obj" scale = "{scale} {scale} {scale}"/>
      </geometry>
      <material name="right_foot_material">
        <color rgba="0.6 0.68 0.75 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="{s('0.015691 0.00858911 0.0172522')}" rpy="-1.5708 -1.5708 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/right_foot.obj" scale = "{scale} {scale} {scale}"/>
      </geometry>
      <drake:proximity_properties>
        <drake:rigid_hydroelastic/>
        <drake:mu_dynamic value="{feet_friction}"/>
        <drake:mu_static value="{feet_friction}"/>
        <!--drake:hydroelastic_modulus value="{s('5.0e7')}"-->
        <drake:mesh_resolution_hint value="{0.00157}"/>
      </drake:proximity_properties>
    </collision>

    <!-- Part right_hardstop_cyl -->
    <visual>
      <origin xyz="{s('0.015691 0.00858911 0.0173272')}" rpy="-1.5708 -1.5708 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/right_hardstop_cyl.obj" scale = "{scale} {scale} {scale}"/>
      </geometry>
      <material name="right_hardstop_cyl_material">
        <color rgba="0.8 0.8 0.8 1.0"/>
      </material>
    </visual>
    <!-- 
    <collision>
      <origin xyz="{s('0.015691 0.00858911 0.0173272')}" rpy="-1.5708 -1.5708 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/right_hardstop_cyl.obj" scale = "{scale} {scale} {scale}"/>
      </geometry>
    </collision>
    -->

    <!-- Part l_hand_weight -->
    <visual>
      <origin xyz="{s('0.0160186 0.00825631 0.0300522')}" rpy="-1.59189 -1.5708 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/l_hand_weight.obj" scale = "{scale} {scale} {scale}"/>
      </geometry>
      <material name="l_hand_weight_material">
        <color rgba="0.8 0.8 0.8 1.0"/>
      </material>
    </visual>
    <!-- 
    <collision>
      <origin xyz="{s('0.0160186 0.00825631 0.0300522')}" rpy="-1.59189 -1.5708 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/l_hand_weight.obj" scale = "{scale} {scale} {scale}"/>
      </geometry>
    </collision>
    -->

    <!-- Part right_leg -->
    <visual>
      <origin xyz="{s('0.015691 0.00858911 0.0172522')}" rpy="-1.5708 -1.5708 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/right_leg.obj" scale = "{scale} {scale} {scale}"/>
      </geometry>
      <material name="right_leg_material">
        <color rgba="0.8 0.8 0.8 1.0"/>
      </material>
    </visual>
    <!-- 
    <collision>
      <origin xyz="{s('0.015691 0.00858911 0.0172522')}" rpy="-1.5708 -1.5708 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/right_leg.obj" scale = "{scale} {scale} {scale}"/>
      </geometry>
    </collision>
    -->

    <!-- Part l_arm -->
    <visual>
      <origin xyz="{s('0.015691 0.00858911 0.0172522')}" rpy="-1.5708 -1.5708 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/l_arm.obj" scale = "{scale} {scale} {scale}"/>
      </geometry>
      <material name="l_arm_material">
        <color rgba="0.8 0.8 0.8 1.0"/>
      </material>
    </visual>
    <!-- 
    <collision>
      <origin xyz="{s('0.015691 0.00858911 0.0172522')}" rpy="-1.5708 -1.5708 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/l_arm.obj" scale = "{scale} {scale} {scale}"/>
      </geometry>
    </collision>
    -->
  </link>


  <!-- Link right_hardstop -->
  <link name="right_hardstop">
    <inertial>
      <origin xyz="{s('-2.3611e-18 0.00108382 -0.001625')}" rpy="0 0 0"/>
      <mass value="{s_mass('0.0016')}"/>
      <inertia ixx="{s_inertia('1.83367e-09')}" 
      ixy="{s_inertia('0')}" 
      ixz="{s_inertia('-9.00702e-25')}" 
      iyy="{s_inertia('5.75858e-09')}" 
      iyz="{s_inertia('0')}" 
      izz="{s_inertia('4.77559e-09')}"/>
    </inertial>
    <!-- Part right_hardstop -->
    <visual>
      <origin xyz="{s('0.015691 0.0131341 0.0133272')}" rpy="-1.5708 -1.5708 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/right_hardstop.obj" scale = "{scale} {scale} {scale}"/>
      </geometry>
      <material name="right_hardstop_material">
        <color rgba="0.8 0.8 0.8 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="{s('0.015691 0.0131341 0.0133272')}" rpy="-1.5708 -1.5708 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/right_hardstop.obj" scale = "{scale} {scale} {scale}"/>
      </geometry>
      <drake:proximity_properties>
        <drake:rigid_hydroelastic/>
        <drake:mu_dynamic value="0.5"/>
        <drake:mu_static value="0.5"/>
        <!--drake:hydroelastic_modulus value="{s('5.0e7')}"-->
        <drake:mesh_resolution_hint value="{0.00157}"/>
      </drake:proximity_properties>
    </collision>
  </link>


  <!-- Joint from right_leg to right_hardstop -->
  <joint name="rightHS" type="fixed">
    <origin xyz="{s('-4.27176e-17 -0.004545 0.004')}" rpy="0 -0 0"/>
    <parent link="right_leg"/>
    <child link="right_hardstop"/>
    <axis xyz="0 0 1"/>
    <limit effort="10" velocity="10" lower="-1" upper="1"/>
  </joint>


  <!-- Joint from left_leg to right_leg -->
  <joint name="hip" type="revolute">
    <origin xyz="{s('-0.0173272 -0.00179095 -0.00858911')}" rpy="1.5708 -0 1.5708"/>
    <parent link="left_leg"/>
    <child link="right_leg"/>
    <axis xyz="0 0 1"/>
    <limit effort="10" velocity="10" lower="-3.141592653589793" upper="3.141592653589793"/>
  </joint>


<transmission name="hip_joint_transmission">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="hip">
        <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
        </joint>
    
    <actuator name="hip_joint_motor">
        <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
    </actuator>
</transmission>


</robot>
    """


    # <!-- Rotate robot so it's upright -->
    # <link name="dummy">
    # </link>

    # <joint name="dummy_joint" type="floating">
    #     <parent link="dummy"/>
    #     <child link="left_leg"/>
    # </joint>


    # <joint name="world_dummy_joint" type="fixed">
    #     <parent link="dummy"/>
    #     <child link="left_leg"/>
    #     <origin xyz="0 0 0" rpy="{-np.pi/2} -0 0"/>
    # </joint>


def setup_walker_plant(scale, ground_friction, feet_friction, timestep=0.0001):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=timestep)
    parser = Parser(plant)
    parser.AddModelsFromString(create_walker_urdf(scale = scale, ground_friction = ground_friction, feet_friction = feet_friction),"urdf")

    with open(_os.path.join(_PKG_DIR, "generated_urdf.txt"),'w') as f:
      f.write(create_walker_urdf(scale = scale, ground_friction = ground_friction, feet_friction = feet_friction))

    # plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)
    plant.Finalize()
    instance = plant.GetModelInstanceByName("walker")
    return plant, scene_graph, builder, instance

def setup_walker_controller_plant(scale, ground_friction, feet_friction, timestep=0.0001):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=timestep)
    parser = Parser(plant)
    parser.AddModelsFromString(create_walker_urdf(scale = scale, ground_friction = ground_friction, feet_friction = feet_friction), "urdf")
    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)
    plant.Finalize()
    diagram = builder.Build()
    instance = plant.GetModelInstanceByName("walker")
    return plant, scene_graph, diagram, instance


def get_home_state(scale):
    """spider for now."""
    home_state = np.zeros(15)
    ## state space definition
    # [0] = qw (relative to left leg)
    # [1] = qx (relative to left leg)
    # [2] = qy (relative to left leg)
    # [3] = qz (relative to left leg)
    # [4] = x (relative to left leg)
    # [5] = y (relative to left leg)
    # [6] = z (relative to left leg)
    # [7] = hip joint angle
    # [8] = q _dot (relative to left leg)
    # [9] = q _dot (relative to left leg)
    # [10] = q _dot (relative to left leg)
    # [11] = x_dot (relative to left leg)
    # [12] = y_dot (relative to left leg)
    # [13] = z_dot (relative to left leg)
    # [14] = hip joint angular velocity

    
    # Regular sim home_state values, 
    home_state[0:4] = RollPitchYaw(roll=0,pitch=0,yaw=-np.pi/2).ToQuaternion().wxyz()
    home_state[6] = 0.04 # starting height

    # # use this for checking the natural frequency. This drops the robot in at an angle
    # home_state[0:4] = RollPitchYaw(roll=np.pi/2,pitch=0,yaw=0).ToQuaternion().wxyz()
    # home_state[6] = 0.3
    # home_state[2] = -0.2
    
    return home_state * scale

'''---------------------------------------Set up foot contact--------------------------------------'''
class ContactResultsToArray(LeafSystem):
    def __init__(
            self,
            plant,
            scene_graph,
            collision_pairs = [
                [ScopedName("walker", "left_leg"), ScopedName("walker", "ground")],
                [ScopedName("walker", "right_leg"), ScopedName("walker", "ground")],
            ]
            # collision_pairs: list[list[ScopedName]] # Requires scoped names
            ):

        LeafSystem.__init__(self)
        self.geometryid2name={}
        scene_graph_context = scene_graph.CreateDefaultContext()
        query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)
        inspector = query_object.inspector()
        for geometry_id in inspector.GetAllGeometryIds():
            body = plant.GetBodyFromFrameId(inspector.GetFrameId(geometry_id))
            if hasattr(body,'name'):
                # Scoped name adds the name of the object and the body
                scoped_name = body.scoped_name()
                self.geometryid2name[geometry_id.get_value()]=scoped_name.to_string()
            else:
                self.geometryid2name[geometry_id.get_value()]='NONAME'
        self.collision_pair_map = {}
        start_idx = 0
        for collision_pair in collision_pairs:
            name1 = collision_pair[0].to_string()
            name2 = collision_pair[1].to_string()
            if name1 not in self.collision_pair_map:
                self.collision_pair_map[name1] = {}
            if name2 not in self.collision_pair_map:
                self.collision_pair_map[name2] = {}
            idx_range = [start_idx,start_idx + 3]
            # collect both directions for efficiency later.
            self.collision_pair_map[name1][name2] = idx_range
            self.collision_pair_map[name2][name1] = idx_range
            start_idx += 3
        self.num_forces = start_idx
        self.force_output = np.zeros(self.num_forces)
        self.contact_points = np.zeros(self.num_forces)  # List to store contact points

        self.force_output_dict: dict[str, np.array] = dict()
        self.contact_points_dict: dict[str, np.array] = dict()

        self.DeclareAbstractInputPort(
            "contact_results", AbstractValue.Make(ContactResults())
        )
        self.DeclareVectorOutputPort(
            "contact_results_array", self.num_forces, self.Publish
        )        
        # Add periodic update event
        self.DeclarePeriodicDiscreteUpdateEvent(0.001, 0, self.Publish) # used to be 0.0001

    def GetCollisionPairMap(self):
        return self.collision_pair_map
    def Publish(self, context, output):
        
        formatter = {"float": lambda x: "{:5.2f}".format(x)}
        results = self.get_input_port().Eval(context)

        # Reset forces and contact points for this loop
        self.force_output[:] = 0.0
        self.contact_points[:] = 0.0

        # Arrays for left and right foot forces and contact points
        left_foot_force = np.zeros(3)
        right_foot_force = np.zeros(3)
        left_foot_point = np.zeros(3)
        right_foot_point = np.zeros(3)

        # Loop over all hydroelastic contacts
        for i in range(results.num_hydroelastic_contacts()):
            info = results.hydroelastic_contact_info(i)
            cs = info.contact_surface()
            id1 = cs.id_M().get_value()
            id2 = cs.id_N().get_value()

            name1 = self.geometryid2name[id1]
            name2 = self.geometryid2name[id2]
            spatialforce = info.F_Ac_W()
            fxfyfz = spatialforce.translational()
            contact_point = cs.centroid()

            # Check if the contact is with the left foot
            if (ScopedName("walker", "left_leg").to_string() in name1 or 
                ScopedName("walker", "left_leg").to_string() in name2):
                left_foot_force += fxfyfz
                left_foot_point = contact_point  # Store contact point for the left foot

            # Check if the contact is with the right foot
            elif (ScopedName("walker", "right_leg").to_string() in name1 or 
                ScopedName("walker", "right_leg").to_string() in name2):
                right_foot_force += fxfyfz
                right_foot_point = contact_point  # Store contact point for the right foot

        # Store both the forces and contact points for the current timestep
        self.force_output_dict[str(context.get_time())] = {
            'left_foot_force': left_foot_force.copy(),
            'right_foot_force': right_foot_force.copy()
        }
        self.contact_points_dict[str(context.get_time())] = {
            'left_foot_point': left_foot_point.copy(),
            'right_foot_point': right_foot_point.copy()
        }


    def get_forces_and_points(self):
        return self.force_output_dict, self.contact_points_dict

'''---------------------------------------Set up controller--------------------------------------'''
def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2  
    return np.array([w, x, y, z])

def normalize_quat(q):
    norm = np.linalg.norm(q)
    # return q / norm if norm != 0 else q
    return q / norm 

def quat_inverse(q):
    # Assuming q is a numpy array [w, x, y, z]
    return np.array([q[0], -q[1], -q[2], -q[3]])

# walking data
sample_int = 0.2 * 1000 #in milliseconds
file_name = f"{_DATA_DIR}/zippy_power_data.csv"
sample = np.genfromtxt(file_name, delimiter=',', usecols=0, dtype=float, skip_header=1)
pol_voltage = np.genfromtxt(file_name, delimiter=',', usecols=1, dtype=float, skip_header=1)
pol_current = np.genfromtxt(file_name, delimiter=',', usecols=2, dtype=float, skip_header=1)
pol_power = np.genfromtxt(file_name, delimiter=',', usecols=3, dtype=float, skip_header=1)
pol_time = sample_int * sample

def ozin_to_Nm(torque_in_ozin): 
    torque_in_Nm = torque_in_ozin * 0.0070615518333333
    return torque_in_Nm

def rpm_to_radpersec(angvel_rpm):
    angvel_radpersec = angvel_rpm * 2*np.pi/60
    return angvel_radpersec

def get_hip_torque(scale, safety_factor=1.0):
    SCALE_DATA = np.array([0.9,         1.0,         1.02960932,  1.17788372,  1.34751117,  1.54156673, 
                           1.76356831,  2.01754040,  2.30808711,  2.64047555,  3.02073137, 
                           3.45574796,  3.95341145,  4,           4.52274363,  5.17406555, 
                           5.91918456,  6,           6.12,        6.194,       6.77160842,  7.74679015, 8, 
                           8.86240814,  10.13868667, 11,          11.59876253, 13.26910443, 
                           15,          15.17999286, 17.36606900, 19.86696275, 22.72801109, 
                           26.00108001, 29.74550475, 30,          34.02916543, 38.92971759, 
                           40,          40.486,      41,          44.536],     dtype=float)
    
    TORQUE_DATA = np.array([0.003,      0.005,       0.007,       0.010,       0.02,        0.03,       
                            0.05,       0.08,        0.11,        0.2,         0.3, 
                            0.6,        1.0,         1.3,         2,           3,  
                            4.8,        5.8,         6,           7,           8,           13,        15, 
                            25,         40,          50,          75,          140, 
                            195,        200,         390,         770,         985, 
                            1800,       2800,        3100,        5000,        7600, 
                            8500,       9000,       11000,       14000],       dtype=float)
    
    POWER_ALPHA = 3.9034 # found from getting a bunch of torque values that lead to walking
    POWER_C = 0.00507   # torque ≈ C * scale^alpha
    if SCALE_DATA[0] <= scale <= SCALE_DATA[-1]:
        min_torque = np.interp(scale, SCALE_DATA, TORQUE_DATA)
    else:
        min_torque = POWER_C * scale**POWER_ALPHA

    exact_idx = np.where(np.abs(SCALE_DATA - scale) < 1e-5)[0]
    if len(exact_idx) > 0:
        torque = TORQUE_DATA[exact_idx[0]]
    else:
        torque = min_torque * safety_factor
    torque = float(np.clip(torque, min_torque, 1e5))

    return torque

    
def get_torque_input(volt_val,plot_walking_data = False,plot_motor_specs = False, pol_current = pol_current):
    # motor specs: https://www.pololu.com/product/2357/specs
    # the script is based on https://www.mathworks.com/matlabcentral/fileexchange/54695-polulu-motor-plot-generator 
    # (the parts of the script that plot the real motor values were mostly generated by chatgpt)
    
    # Constants
    discrete_bins = 500  # Number of bins for plotting and calculating functions like torque, speed, etc.

    # Predefined values (as provided by pololu)
    StallTorque = 1.5  # in oz-inch
    StallCurrent = 400  # in mA
    RatedVoltage = 6  # in Volts
    NoLoadCurrent = 35  # in mA
    NoLoadSpeed = 2500  # in RPM

    # Calculations
    Resistance = RatedVoltage / (StallCurrent / 1000)

    # Torque line
    TorqueLine = np.linspace(0, StallTorque, discrete_bins)

    # Current Line
    CurrentLine = np.linspace(NoLoadCurrent, StallCurrent, discrete_bins)

    # Speed Line
    SpeedLine = np.linspace(NoLoadSpeed, 0, discrete_bins)

    # Torque Constant (Torque per Current)
    SlopeOfTorqueVsCurrent = (StallCurrent - NoLoadCurrent) / StallTorque

    # Output Mechanical Power in watts
    OutputPower = 0.00074 * TorqueLine * SpeedLine

    # Input Electrical Power to the motor in watts
    InputPower = CurrentLine * RatedVoltage / 1000  # Convert mA to A for power in watts

    # Power Efficiency
    PowerEff = np.divide(OutputPower, InputPower, where=InputPower!=0)  # Avoid division by zero

    # Output information
    max_power = np.max(OutputPower)
    max_power_idx = np.argmax(OutputPower)

    # Get the average peak current of motor 
    # I'm assuming that the average motor current is the max current reading - min current reading. 
    # I'm getting the average of the max values (when the motor is on) and the min values (when the motor is off)
    threshold = np.mean(pol_current[30:])
    peaks, _ = find_peaks(pol_current)
    peak_pol_current = pol_current[peaks][(pol_current[peaks] > threshold) & (pol_current[peaks] < 0.17)]
    avg_max_current = np.mean(peak_pol_current)
    valleys, _ = find_peaks(-pol_current)
    val_pol_current = pol_current[peaks][(pol_current[peaks] > -threshold)]
    avg_min_current = np.mean(val_pol_current)
    avg_current_amp = avg_max_current - avg_min_current
    avg_current = np.mean(pol_current[30:])

    # Get target torque
    power_val = 0.5
    idx = np.abs(OutputPower - power_val).argmin() # Find the index of the closest power value
    torque_input =  TorqueLine[idx] * 0.0070615518333333 #oz-in to Nm
    ang_vel_input = SpeedLine[idx] * 0.104719755 #RPM to rad/s
    target_input = np.array([torque_input, ang_vel_input])
    # # Get angular velocity for target_state

    # plotting real data values
    # plot data from walking trial
    if plot_walking_data:
        # finding the real angular velocity of the hip joint
        hip_omega_input = []
        for i in range(len(pol_power)):
            idx = np.abs(OutputPower - pol_power[i]).argmin() # Find the index of the closest power value
            hip_omega_input.append(pol_power[i]/TorqueLine[idx])

        # Create a figure and subplots with shared x-axis
        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 12))

        # Plot hip input angular velocity
        axs[0].plot(pol_time[30:], hip_omega_input[30:])
        axs[0].set_title("Hip Angular Velocity")
        axs[0].set_ylabel("Angular Velocity")

        # Plot voltage input
        axs[1].plot(pol_time[30:], pol_voltage[30:])
        axs[1].set_title("Voltage Input")
        axs[1].set_ylabel("Voltage [V]")

        # Plot current input
        axs[2].plot(pol_time[30:], pol_current[30:])
        axs[2].set_title("Current Input")
        axs[2].set_ylabel("Current [A]")

        # Plot power input
        axs[3].plot(pol_time[30:], pol_power[30:])
        axs[3].set_title("Power Input")
        axs[3].set_ylabel("Power [W]")
        axs[3].set_xlabel("Time [1e-4 s]")

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Display the plots
        plt.show()
    else:
        pass

    # plot motor torque specs
    if plot_motor_specs:
        # Plot Torque vs Speed & Torque vs Current
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        ax1, ax2 = axs[0, 0].twinx(), axs[0, 0]
        ax1.plot(TorqueLine, SpeedLine, 'b-', label='Speed (RPM)')
        ax2.plot(TorqueLine, CurrentLine, 'r-', label='Current (mA)')
        
        ax1.set_xlabel('Torque (oz-in)')
        ax1.set_ylabel('Speed (RPM)', color='b')
        ax2.set_ylabel('Current (mA)', color='r')
        axs[0, 0].set_title('Torque vs. Speed & Torque vs. Current')

        # Plot Torque vs Output Power & Torque vs Input Power
        ax3, ax4 = axs[0, 1].twinx(), axs[0, 1]
        ax3.plot(TorqueLine, OutputPower, 'b-', label='Output Power (W)')
        ax4.plot(TorqueLine, InputPower, 'r-', label='Input Power (W)')
        
        ax3.set_xlabel('Torque (oz-in)')
        ax3.set_ylabel('Output Power (W)', color='b')
        ax4.set_ylabel('Input Power (W)', color='r')
        axs[0, 1].set_title('Torque vs. Output Power & Torque vs. Input Power')

        # Plot Power Efficiency
        axs[1, 0].plot(TorqueLine, PowerEff, 'g-')
        
        axs[1, 0].set_xlabel('Torque (oz-in)')
        axs[1, 0].set_ylabel('Power Efficiency')
        axs[1, 0].set_title('Torque vs. Power Efficiency')

        # Output information
        print(f'\n\nSlope of TorqueVsCurrent is {SlopeOfTorqueVsCurrent:.6f}. The reciprocal is {1/SlopeOfTorqueVsCurrent:.6f}') 
        print(f'Maximum output mechanical power is {max_power:.6f} watts.')
        print(f'This happens at a Torque load of {TorqueLine[max_power_idx]:.6f} oz-in, with Current {CurrentLine[max_power_idx]:.6f} mA')
        print(f'Resistance of the motor is {Resistance:.6f} ohms')

        plt.tight_layout()
        plt.show()
    else:
        pass

    return torque_input, ang_vel_input

class Controller(LeafSystem):
    def __init__(
        self,
        scale,
        ground_friction,
        feet_friction,
        # target_state,
        control_period=0.005,
        hip_kp=0,
        hip_ki=0,
        hip_kd=0.001,
        threshold_force = 0.0001, # in N, To set contact mode
        ):

        # Assign the parameters to the instance variables
        self.hip_kp = hip_kp
        self.hip_ki = hip_ki
        self.hip_kd = hip_kd
        self.scale = scale
        self.control_period=control_period
        self.integral_error = 0
        self.time = 0
        self.prev_error = 0
        self.wait_time = 0.135 # 0.135s is correct setting
        self.frequency = 1 / (2 * self.wait_time)  # Square wave frequency (Hz)
        self.frequency = self.frequency/np.sqrt(self.scale)
        self.ang_vel_input = 0
        self.counter = 0
    

        """ For now, just a pd control tracking 1 state."""
        LeafSystem.__init__(self)
        self.controller_plant, self.scene_graph, self.controller_diagram, self.instance = setup_walker_controller_plant(scale = scale, ground_friction = ground_friction,
        feet_friction = feet_friction, timestep=self.control_period)

        # Init context
        self.controller_diagram_context = self.controller_diagram.CreateDefaultContext()
        self.controller_plant_context = self.controller_plant.GetMyContextFromRoot(self.controller_diagram_context)
        
        self.n = self.controller_diagram_context.get_discrete_state_vector().size()
        self.n_pos = self.controller_plant.num_positions()
        self.m = self.controller_plant.num_actuators()

        # Init control signal
        self.control_signal = np.zeros(self.m )

        # Init state measurement
        self.current_state = np.zeros(self.n)

        # Init FT measurements
        orientation_cartesian_dim = 6
        self.threshold_force = threshold_force

        # Init gain matrix
        n_unactuated = self.controller_plant.num_positions() - self.controller_plant.num_actuators()
        """ quaternion difference is size 3 and quaternion is size 4"""
        self.gain_matrix = np.zeros((self.m, self.n - 1)) 
        self.gain_matrix[:,n_unactuated:self.n_pos] = np.eye(self.m)*self.hip_kp #gains for elements corresponding to positions
        self.gain_matrix[:,self.n_pos+n_unactuated:] = np.eye(self.m)*self.hip_kd #gains for elements corresponding to velocities

        #Init target state
        self.target_state = get_home_state(scale)

        # Specify inputs and outputs
        self.state_input_index = self.DeclareVectorInputPort(
            "state", self.n
        ).get_index()
        
        self.DeclareVectorOutputPort(
            "control", self.m , self.SetOutput
        )


        # Add periodic update event
        self.DeclarePeriodicDiscreteUpdateEvent(self.control_period, 0, self.Update)
    
    @staticmethod
    def ComputeStateDifference(desired_state, current_state):
        """ Fill out this """
        difference_state = np.zeros(len(current_state)-1)

        # troubleshooting
        current_quat = current_state[0:4]
        desired_quat = desired_state[0:4]
        current_norm = np.linalg.norm(current_quat)
        desired_norm = np.linalg.norm(desired_quat)
        if current_norm == 0:
            print("Current quaternion is a zero vector.")
        if desired_norm == 0:
            print("Desired quaternion is a zero vector.")

        current_quaternion = Quaternion(current_quat/current_norm)
        desired_quaternion = Quaternion(desired_quat/desired_norm)
        difference_quaternion = desired_quaternion.multiply(current_quaternion.inverse())
        difference_angle_axis = AngleAxis(difference_quaternion)
        difference_rotation = difference_angle_axis.axis() * difference_angle_axis.angle()
        difference_state[0:3] = difference_rotation
        difference_state[3:] = desired_state[4:] - current_state[4:]
        return difference_state
    @staticmethod
    
    def ComputeControl(self, current_state, desired_state, gain_matrix,feedforward=None):
        error = desired_state[-1] - current_state[-1]
        self.integral_error += (error * self.control_period)
        self.derivative_error = (error - self.prev_error)/self.control_period
        feedback_input = 0
        self.prev_error = error
        self.time += self.control_period
        if feedforward is None:
            return feedback_input
        else:
            return feedback_input + feedforward

    def LegStateMachine(self):
        pass
  


    def Update(self, context, events):
        # # get time if needed
        # get the current state
        self.current_state = self.get_input_port(int(self.state_input_index)).Eval(context)

        # set the state in our internal robot model
        self.controller_plant.SetPositionsAndVelocities(self.controller_plant_context,self.current_state)
        """ 
        We can compute any rigid body dynamics quantities here now with controller plant. 
        List is here https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_multibody_plant.html
        For example: plant.CalcMassMatrix, plant.CalcBiasTerm (coriollis*v)
        Save a class variable for these quantities and then you can just grab it after each timestep.
        """
        
        self.mass_matrix = self.controller_plant.CalcMassMatrix(self.controller_plant_context)
        
        elapsed_time = context.get_time()
        act_start_time = 2
        adjusted_time = elapsed_time - act_start_time
        
        # compute control (torque input for hip motor)
        if elapsed_time > act_start_time:   
            max_value = 2.9
            min_value = 2.7

            # Generate square wave signal
            square_wave = np.sign(np.sin(2 * np.pi * self.frequency * elapsed_time))

            # Determine torque input based on square wave
            volt_val = max_value if square_wave > 0 else min_value
            torque_input, ang_vel_input = get_torque_input(volt_val)
            
            torque_input = get_hip_torque(self.scale)

            
            ang_vel_input = ang_vel_input *2
            self.ang_vel_input = ang_vel_input

            # if self.counter == 0:


            # Set target state and feedforward input
            self.target_state[-1] = ang_vel_input if square_wave > 0 else -ang_vel_input
            feedforward_input = torque_input if square_wave > 0 else -torque_input

            # compute control
            self.control_signal[:] = self.ComputeControl(self,
                current_state=self.current_state,
                desired_state=self.target_state,
                gain_matrix=self.gain_matrix,
                feedforward=feedforward_input
                )
        else:
            pass

    def SetOutput(self, context, output):
        output.SetFromVector(self.control_signal)