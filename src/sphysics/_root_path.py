#pylint: disable=invalid-name
from os.path import join, exists
import sys

root_paths = [ join(p, 'sphysics') for p in sys.path if exists(join(p, 'sphysics', 'libSphysicsPwa.so')) ]

if not root_paths:
	raise FileNotFoundError("Cannot find installation folder of sphysics")

root_path = root_paths[0]
