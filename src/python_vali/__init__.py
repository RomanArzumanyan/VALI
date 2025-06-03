"""

"""

__author__ = "Roman Arzumanyan"
__copyright__ = "Copyright 2022, NVIDIA; Copyright 2023, Vision Labs LLC"
__credits__ = []
__license__ = "Apache 2.0"
__version__ = "4.7.1.3"
__maintainer__ = "Roman Arzumanyan"
__email__ = "TODO"
__status__ = "Production"

try:
    # Import native module
    from ._python_vali import *  # noqa
except ImportError:
    import distutils.sysconfig
    from os.path import join, dirname
    raise RuntimeError("Failed to import native module _python_vali! "
                           f"Please check whether \"{join(dirname(__file__), '_python_vali' + distutils.sysconfig.get_config_var('EXT_SUFFIX'))}\""  # noqa
                           " exists and can find all library dependencies (CUDA, ffmpeg).\n"
                           "On Unix systems, you can use `ldd` on the file to see whether it can find all dependencies.\n"
                           "On Windows, you can use \"dumpbin /dependents\" in a Visual Studio command prompt or\n"
                           "https://github.com/lucasg/Dependencies/releases.")
