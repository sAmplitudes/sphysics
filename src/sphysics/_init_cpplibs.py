#pylint: disable=invalid-name,raise-missing-from
from os.path import join, exists

# runtime dependencies
from ctypes import cdll

from . import tf # pylint: disable=unused-import
from ._root_path import root_path


try:
	cdll.LoadLibrary(join(root_path, "libSphysicsPwa.so"))
except OSError:
	if not exists(join(root_path, "libSphysicsPwa.so")):
		raise FileNotFoundError(f"libSphysicsPwa.so was not found on your system at '{join(root_path, 'libSphysicsPwa.so')}'. Is this package correctly installed?")
	raise FileNotFoundError("libSphysicsPwa.so could not be loaded. There is a missing dependency.")

try:
	cdll.LoadLibrary(join(root_path, "libSphysicsConstants.so"))
except OSError:
	if not exists(join(root_path, "libSphysicsConstants.so")):
		raise FileNotFoundError("libSphysicsConstants.so was not found on your system. Is this package correctly installed?")
	raise FileNotFoundError("libSphysicsConstants.so could not be loaded. There is a missing dependency.")

try:
	cdll.LoadLibrary(join(root_path, "libSphysicsParticleData.so"))
except OSError:
	if not exists(join(root_path, "libSphysicsParticleData.so")):
		raise FileNotFoundError("libSphysicsParticleData.so was not found on your system. Is this package correctly installed?")
	raise FileNotFoundError("libSphysicsParticleData.so could not be loaded. There is a missing dependency.")

# try:
# 	cdll.LoadLibrary(join(root_path, "libSphysicsTfLib.so"))
# except OSError:
# 	if not exists(join(root_path, "libSphysicsTfLib.so")):
# 		raise FileNotFoundError("libSphysicsTfLib.so was not found on your system. Is this package correctly installed?")
# 	raise FileNotFoundError("libSphysicsTfLib.so could not be loaded. There is a missing dependency.")

try:
	cdll.LoadLibrary(join(root_path, "libSphysicsUtilities.so"))
except OSError:
	if not exists(join(root_path, "libSphysicsUtilities.so")):
		raise FileNotFoundError("libSphysicsUtilities.so was not found on your system. Is this package correctly installed?")
	raise FileNotFoundError("libSphysicsUtilities.so could not be loaded. There is a missing dependency.")
