
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
        return str((scale ** 2) * float(m))

    def s_inertia(i):  # scale inertia
        return str((scale ** 4) * float(i))

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
          <box size="{s('40 40 0.5')}" />
      </geometry>
      <material>
          <!-- <color rgba="0.93 .74 .4 1" /> -->
          <color rgba="0.3 0.38 0.46 1.0" />
      </material>
      </visual>
      <collision>
      <origin xyz="{s('0 0 -0.25')}" rpy="0 0 0" />
      <geometry>
          <box size="{s('40 40 0.5')}" />
      </geometry>
          <drake:proximity_properties>
          <drake:compliant_hydroelastic/>
          <drake:hydroelastic_modulus value="{s('1e8')}"/>
          <drake:mu_dynamic value="{ground_friction}"/>
          <drake:mu_static value="{ground_friction}"/>
          </drake:proximity_properties>
      </collision>
  </link>

  <joint name="ground_weld" type="fixed">
      <parent link="world" />
      <child link="ground" />
  </joint>

  <!-- Link right_leg -->
  <link name="right_leg">
    <inertial>
      <origin xyz="{s('0.00339271 -0.0683175 0.066068')}" rpy="0 0 0"/>
      <mass value="{s_mass('0.265796')}"/>
      <inertia ixx="{s_inertia('0.000867908')}" 
      ixy="{s_inertia('-9.90116e-05')}" 
      ixz="{s_inertia('-1.96539e-05')}" 
      iyy="{s_inertia('0.000418449')}" 
      iyz="{s_inertia('-0.000161766')}" 
      izz="{s_inertia('0.000602868')}"/>
    </inertial>

    <!-- Part left_arm -->
    <visual>
      <origin xyz="{s('0.0781384 0.0652662 -0.0232028')}" rpy="1.5708 1.5708 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/left_arm.obj" scale="{scale} {scale} {scale}"/>
      </geometry>
      <material name="left_arm_material">
        <color rgba="0.8 0.8 0.8 1.0"/>
      </material>
    </visual>
    <!-- 
    <collision>
      <origin xyz="{s('0.0781384 0.0652662 -0.0232028')}" rpy="1.5708 1.5708 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/left_arm.obj" scale="{scale} {scale} {scale}"/>
      </geometry>
    </collision>
    -->

    <!-- Part right_leg -->
    <visual>
      <origin xyz="{s('-0.0255 0.02 -0.005')}" rpy="1.5708 3.40689e-23 1.27453e-26"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/right_leg.obj" scale="{scale} {scale} {scale}"/>
      </geometry>
      <material name="right_leg_material">
        <color rgba="0.8 0.8 0.8 1.0"/>
      </material>
    </visual>
    <!-- 
    <collision>
      <origin xyz="{s('-0.0255 0.02 -0.005')}" rpy="1.5708 3.40689e-23 1.27453e-26"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/right_leg.obj" scale="{scale} {scale} {scale}"/>
      </geometry>
    </collision>
    -->

    <!-- Part right_battery -->
    <visual>
      <origin xyz="{s('0.0347476 -0.0595251 0.0792817')}" rpy="1.5708 -1.79157e-17 2.96706"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/right_battery.obj" scale="{scale} {scale} {scale}"/>
      </geometry>
      <material name="right_battery_material">
        <color rgba="0.8 0.8 0.8 1.0"/>
      </material>
    </visual>
    <!-- 
    <collision>
      <origin xyz="{s('0.0347476 -0.0595251 0.0792817')}" rpy="1.5708 -1.79157e-17 2.96706"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/right_battery.obj" scale="{scale} {scale} {scale}"/>
      </geometry>
    </collision>
    -->
  </link>


  <!-- Link left_leg -->
  <link name="left_leg">
    <inertial>
      <origin xyz="{s('0.0165799 0.0617476 -0.0445552')}" rpy="0 0 0"/>
      <mass value="{s_mass('0.28377')}"/>
      <inertia ixx="{s_inertia('0.000997428')}" 
      ixy="{s_inertia('-0.000118618')}" 
      ixz="{s_inertia('3.35196e-05')}" 
      iyy="{s_inertia('0.000490668')}" 
      iyz="{s_inertia('0.000210604')}" 
      izz="{s_inertia('0.000678868')}"/>
    </inertial>

    <!-- Part right_arm -->
    <visual>
      <origin xyz="{s('0.0924094 0.198961 -0.017434')}" rpy="1.5708 1.5708 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/right_arm.obj" scale="{scale} {scale} {scale}"/>
      </geometry>
      <material name="right_arm_material">
        <color rgba="0.8 0.8 0.8 1.0"/>
      </material>
    </visual>
    <!-- 
    <collision>
      <origin xyz="{s('0.0924094 0.198961 -0.017434')}" rpy="1.5708 1.5708 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/right_arm.obj" scale="{scale} {scale} {scale}"/>
      </geometry>
    </collision>
    -->

    <!-- Part right_battery_2 -->
    <visual>
      <origin xyz="{s('-0.00788849 0.0842042 -0.0613817')}" rpy="-1.5708 3.37492e-21 -0.174533"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/right_battery.obj" scale="{scale} {scale} {scale}"/>
      </geometry>
      <material name="right_battery_2_material">
        <color rgba="0.8 0.8 0.8 1.0"/>
      </material>
    </visual>
    <!-- 
    <collision>
      <origin xyz="{s('-0.00788849 0.0842042 -0.0613817')}" rpy="-1.5708 3.37492e-21 -0.174533"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/right_battery.obj" scale="{scale} {scale} {scale}"/>
      </geometry>
    </collision>
    -->

    <!-- Part left_leg -->
    <visual>
      <origin xyz="{s('-0.0112911 0.153343 0.023')}" rpy="1.5708 -1.07671e-23 1.46076e-20"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/left_leg.obj" scale="{scale} {scale} {scale}"/>
      </geometry>
      <material name="left_leg_material">
        <color rgba="0.8 0.8 0.8 1.0"/>
      </material>
    </visual>
    <!-- 
    <collision>
      <origin xyz="{s('-0.0112911 0.153343 0.023')}" rpy="1.5708 -1.07671e-23 1.46076e-20"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/left_leg.obj" scale="{scale} {scale} {scale}"/>
      </geometry>
    </collision>
    -->
  </link>

    <!-- Link right_foot -->
  <link name="right_foot">
    <inertial>
      <origin xyz="{s('0.088555 0.0226985 -0.140596')}" rpy="0 0 0"/>
      <mass value="{s_mass('0.130181')}"/>
      <inertia ixx="{s_inertia('8.91682e-05')}" 
      ixy="{s_inertia('-3.04732e-08')}" 
      ixz="{s_inertia('-3.66091e-06')}" 
      iyy="{s_inertia('4.06624e-05')}" 
      iyz="{s_inertia('9.09291e-09')}" 
      izz="{s_inertia('0.000119968')}"/>
    </inertial>
    <!-- Part right_foot -->
    <visual>
      <origin xyz="{s('0.0658465 0.0227089 -0.133251')}" rpy="-1.41849e-10 -0.00139484 2.03391e-07"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/right_foot.obj" scale="{scale} {scale} {scale}"/>
      </geometry>
      <material name="right_foot_material">
        <color rgba="0.6 0.68 0.75 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="{s('0.0658465 0.0227089 -0.133251')}" rpy="-1.41849e-10 -0.00139484 2.03391e-07"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/right_foot.obj" scale="{scale} {scale} {scale}"/>
      </geometry>
      <drake:proximity_properties>
          <drake:rigid_hydroelastic/>
          <drake:mu_dynamic value="{feet_friction}"/>
          <drake:mu_static value="{feet_friction}"/>
          <!--drake:hydroelastic_modulus value="{s('5.0e7')}"-->
          <drake:mesh_resolution_hint value="{s('0.1')}"/>
      </drake:proximity_properties>
    </collision>
  </link>


  <!-- Link left_foot -->
  <link name="left_foot">
    <inertial>
      <origin xyz="{s('-1.0409e-05 0.00737654 0.0176982')}" rpy="0 0 0"/>
      <mass value="{s_mass('0.130181')}"/>
      <inertia ixx="{s_inertia('4.06624e-05')}" 
      ixy="{s_inertia('-9.09829e-09')}" 
      ixz="{s_inertia('-3.05961e-08')}" 
      iyy="{s_inertia('0.000119978')}" 
      iyz="{s_inertia('3.61793e-06')}" 
      izz="{s_inertia('8.9158e-05')}"/>
    </inertial>
    <!-- Part left_foot -->
    <visual>
      <origin xyz="{s('-3.46945e-18 2.77556e-17 -0.0576688')}" rpy="1.5708 1.5708 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/left_foot.obj" scale="{scale} {scale} {scale}"/>
      </geometry>
      <material name="left_foot_material">
        <color rgba="0.6 0.68 0.75 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="{s('-3.46945e-18 2.77556e-17 -0.0576688')}" rpy="1.5708 1.5708 0"/>
      <geometry>
        <mesh filename="file://{_MESH_DIR}/left_foot.obj" scale="{scale} {scale} {scale}"/>
      </geometry>
      <drake:proximity_properties>
          <drake:rigid_hydroelastic/>
          <drake:mu_dynamic value="{feet_friction}"/>
          <drake:mu_static value="{feet_friction}"/>
          <!--drake:hydroelastic_modulus value="{s('5.0e7')}"-->
          <drake:mesh_resolution_hint value="{s('0.1')}"/>
      </drake:proximity_properties>
    </collision>
  </link>


  <!-- Joint from left_leg to left_foot -->
  <joint name="left_foot_joint" type="fixed">
    <origin xyz="{s('0.0142089 0.133343 0.018')}" rpy="-2.46197e-19 -1.07671e-23 1.46076e-20"/>
    <parent link="left_leg"/>
    <child link="left_foot"/>
    <axis xyz="0 0 1"/>
  </joint>


  <!-- Joint from right_leg to left_leg -->
  <joint name="hip" type="revolute">
    <origin xyz="{s('-0.0142089 -0.133343 0.0301')}" rpy="-6.79877e-19 2.60008e-23 0.000145816"/>
    <parent link="right_leg"/>
    <child link="left_leg"/>
    <axis xyz="0 0 1"/>
    <limit effort="{s('10')}" velocity="{s('10')}" lower="-1.57094" upper="1.57065"/>
  </joint>

  <!-- Joint from right_foot to right_leg -->
  <joint name="right_foot_joint" type="fixed">
    <origin xyz="{s('0.0708465 0.0227089 -0.133244')}" rpy="-1.57219 1.41849e-10 1.5708"/>
    <parent link="right_foot"/>
    <child link="right_leg"/>
    <axis xyz="0 0 1"/>
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


  #   <link name="leftfoot_mass">
  #     <inertial>
  #         <origin rpy="0 0 0" xyz="{s('0 0 0')}"/>
  #         <mass value="{s_mass('0.18')}"/>
  #         <inertia ixx="{s_inertia('0.01333333')}" 
  #         izz="{s_inertia('0.01333333')}"/>
  #     </inertial>
  #     <visual>
  #         <origin rpy="0 0 0" xyz="{s('0 0 0')}"/>
  #         <geometry>
  #             <box size="{s('0.02 0.02 0.02')}"/>
  #         </geometry>
  #         <material name="apricot">
  #             <color rgba="1.0 0.6941176470588235 0.42745098039215684 1.0"/>
  #         </material>
  #     </visual>
  # </link>

  # <joint name="leftfootmass_joint" type="fixed">
  #     <parent link="left_foot"/>
  #     <child link="leftfoot_mass"/>
  # </joint>

  # <link name="rightfoot_mass">
  #     <inertial>
  #         <origin rpy="0 0 0" xyz="{s('0.08 0.023 -0.133')}"/>
  #         <mass value="{s_mass('0.18')}"/>
  #         <inertia ixx="{s_inertia('0.01333333')}" 
  #         izz="{s_inertia('0.01333333')}"/>
  #     </inertial>
  #     <visual>
  #         <origin rpy="0 0 0" xyz="{s('0.08 0.023 -0.133')}"/>
  #         <geometry>
  #             <box size="{s('0.02 0.02 0.02')}"/>
  #         </geometry>
  #         <material name="aqua">
  #             <color rgba="0.07450980392156863 0.9176470588235294 0.788235294117647 1.0"/>
  #         </material>
  #     </visual>
  # </link>
      

  #   <joint name="rightfootmass_joint" type="fixed">
  #       <parent link="right_foot"/>
  #       <child link="rightfoot_mass"/>
  #   </joint>



def setup_walker_plant(scale, ground_friction, feet_friction, timestep=0.001):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=timestep)
    parser = Parser(plant)
    parser.AddModelsFromString(create_walker_urdf(scale = scale, ground_friction = ground_friction, feet_friction = feet_friction),"urdf")
    # plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)
    plant.Finalize()
    instance = plant.GetModelInstanceByName("walker")
    return plant, scene_graph, builder, instance

def setup_walker_controller_plant(scale, ground_friction, feet_friction, timestep=0.001):
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
    home_state[6] = 0.16 # starting height

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
                [ScopedName("walker", "left_foot"), ScopedName("walker", "ground")],
                [ScopedName("walker", "right_foot"), ScopedName("walker", "ground")],
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
        self.DeclarePeriodicDiscreteUpdateEvent(0.001, 0, self.Publish)

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
            if (ScopedName("walker", "left_foot").to_string() in name1 or 
                ScopedName("walker", "left_foot").to_string() in name2):
                left_foot_force += fxfyfz
                left_foot_point = contact_point  # Store contact point for the left foot

            # Check if the contact is with the right foot
            elif (ScopedName("walker", "right_foot").to_string() in name1 or 
                ScopedName("walker", "right_foot").to_string() in name2):
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

class Controller(LeafSystem):
    def __init__(
        self,
        scale,
        ground_friction,
        feet_friction,
        # target_state,
        control_period=0.0005,
        hip_kp=8, 
        hip_ki=0,
        hip_kd=0,
        threshold_force = 0.0001, # in N, To set contact mode
        ):

        # Assign the parameters to the instance variables
        self.scale = scale
        if self.scale >= 6.5:
            hip_kp = hip_kp * 3500 
            hip_kd = 1800
        elif 6.5 > self.scale >= 5.8:
            hip_kp = hip_kp * 2700 
            hip_kd = 1200
        elif 5.8 > self.scale >= 4.5:
            hip_kp = hip_kp * 2000 
            hip_kd = 810
        elif 4.5 > self.scale >= 3.3:
            hip_kp = hip_kp * 2000
            hip_kd = 400
        elif 3.3 > self.scale >= 3:
            hip_kp = hip_kp * 2000
            hip_kd = 300
        elif 3 > self.scale >= 2:
            hip_kp = hip_kp * 1000
            hip_kd = 50
        elif 2 > self.scale >= 1:
            hip_kp = hip_kp * 1000
        elif 1 > self.scale >= 0.5:
            hip_kd = 0.08
        elif 0.5 > self.scale >= 0.3: # works up to 0.163 so far
            hip_kp = 0.8
            hip_kd = 0.02
        elif 0.3 > self.scale >= 0.15: # works up to 0.163 so far
            hip_kp = 0.8
        elif 0.15 > self.scale:
            hip_kp = 0.5
        else:
            hip_kp = 1
            hip_ki = 0

        self.hip_kp = hip_kp
        self.hip_ki = hip_ki
        self.hip_kd = hip_kd
        self.control_period=control_period
        self.integral_error = 0
        self.time = 0
        self.prev_pos_error = 0
        self.prev_vel_error = 0
        # real frequency is calculted by f = (1/2pi) * (g/l)^(1/2) where l is *debatable, likely distance from ground to body COM height
        # based on 1.4Hz, Mugatu's COM should be 12.67 cm below the hip
        # based on the COM height reported in the paper, Mugatu's walking frequency should be 1.93 (assuming l is 6.6cm)
        self.frequency = 1.93 # works from 1.6 to 1.93, have been using 1.7
        self.frequency = self.frequency/np.sqrt(self.scale)
        self.wait_time = 1 / (2 * self.frequency)  
        self.counter = 0
        self.amplitude = 0.5
    

        """ For now, just a pd control tracking 1 state."""
        LeafSystem.__init__(self)
        self.controller_plant, self.scene_graph, self.controller_diagram, self.instance = setup_walker_controller_plant(scale = scale, ground_friction = ground_friction,
        feet_friction = feet_friction)

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
        self.prev_state = self.current_state

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
        pos_error = desired_state[7] - current_state[7]
        vel_error = desired_state[14] - current_state[14]
        self.integral_error += (pos_error * self.control_period)
        self.derivative_error = vel_error

        feedback_input = self.hip_kp*pos_error + self.hip_ki*self.integral_error + self.hip_kd*self.derivative_error
        self.prev_pos_error = pos_error
        self.prev_vel_error = vel_error
        self.time += self.control_period
        self.prev_state = current_state
        if feedforward is None:
            return feedback_input
        else:
            return feedback_input + feedforward

    def LegStateMachine(self):
        pass

    def Update(self, context, events):
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
        act_start_time = 3
        adjusted_time = elapsed_time - act_start_time

        ang_freq = 2 * np.pi * self.frequency
        servo_input = self.amplitude * np.sin(ang_freq * adjusted_time)
        servo_vel = self.amplitude * ang_freq * np.cos(ang_freq * adjusted_time)
        if elapsed_time > act_start_time: # give time for sim to sit and settle
            self.target_state[7] = servo_input # comment this out when you're getting sim natural frequency
            self.target_state[14] = servo_vel # comment this out when you're getting sim natural frequency
        else:
           pass
    
        # compute control
        self.control_signal[:] = self.ComputeControl(self,
            current_state=self.current_state,
            desired_state=self.target_state,
            gain_matrix=self.gain_matrix
            )

    def SetOutput(self, context, output):
        output.SetFromVector(self.control_signal)
