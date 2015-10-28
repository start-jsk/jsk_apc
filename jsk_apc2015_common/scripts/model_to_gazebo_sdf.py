#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('name', help='model name')
args = parser.parse_args()
name = args.name

here = os.path.dirname(os.path.abspath(__file__))

mesh_dir = os.path.join(here, '../meshes', name)
mesh_file = os.path.join(mesh_dir, name + '.dae')
texture_file = os.path.join(mesh_dir, 'optimized_tsdf_texture_mapped_mesh.png')

new_mesh_dir = os.path.join('../models', name, 'meshes')
material_dir = os.path.join('../models', name, 'materials')
scripts_dir = os.path.join(material_dir, 'scripts')
textures_dir = os.path.join(material_dir, 'textures')

os.mkdir(os.path.join('../models', name))
os.mkdir(new_mesh_dir)
os.mkdir(material_dir)
os.mkdir(scripts_dir)
os.mkdir(textures_dir)

shutil.copy(mesh_file, new_mesh_dir)
shutil.copy(texture_file, textures_dir)


with open(os.path.join(new_mesh_dir, name + '.dae'), 'r') as f:
    content = f.read()
    content = content.replace('optimized_poisson_texture_mapped_mesh.png', '../materials/textures/optimized_tsdf_texture_mapped_mesh.png')
with open(os.path.join(new_mesh_dir, name + '.dae'), 'w') as f:
    f.write(content)


with open(os.path.join(scripts_dir, name + '.material'), 'w') as f:
    camel_name = ''.join(map(lambda x: x.capitalize(), name.split('_')))
    content = '''\
material @HERE@
{
  technique
  {
    pass
    {
      texture_unit
      {
        texture optimized_tsdf_texture_mapped_mesh.png
      }
    }
  }
}
'''
    content.replace('@HERE@', camel_name)
    f.write(content)



with open(os.path.join('../models', name, 'model.config'), 'w') as f:
    f.write('''\
<?xml version="1.0"?>

<model>
  <name>{0}</name>
  <version>1.0</version>
  <sdf version="1.0">model.sdf</sdf>

  <description>
    A model of a {0}
  </description>

  <depend>
  </depend>
</model>
'''.format(name))



with open(os.path.join('../models', name, 'model.sdf'), 'w') as f:
    f.write('''
<?xml version="1.0" ?>
<sdf version="1.4">
  <model name="{0}">
    <link name="{0}_link">
      <inertial>
        <pose>0 0 0 0 0 0</pose>
        <mass>3</mass>
        <inertia>
          <ixx>1e-03</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <izz>1e-03</izz>
          <iyz>0</iyz>
          <iyy>1e-03</iyy>
        </inertia>
      </inertial>

      <visual name="{0}_visual">
        <pose>0 0 0 0 0 0</pose>
        <material>
          <script>
            <uri>model://{0}/materials/scripts</uri>
            <uri>model://{0}/materials/textures</uri>
            <name>{0}</name>
          </script>
        </material>
        <geometry>
          <mesh>
            <uri>model://{0}/meshes/{0}.dae</uri>
          </mesh>
        </geometry>
      </visual>


      <collision name="{0}_collision">
        <pose>0 0 0 0 0 0</pose>
        <geometry>
          <mesh>
            <uri>model://{0}/meshes/{0}.dae</uri>
          </mesh>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>0.8</mu>
              <mu2>0.8</mu2>
              <fdir1>0.0 0.0 0.0</fdir1>
              <slip1>1.0</slip1>
              <slip2>1.0</slip2>
            </ode>
          </friction>
        </surface>
      </collision>

    </link>
  </model>
</sdf>
'''.format(name))