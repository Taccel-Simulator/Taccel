cmake_minimum_required(VERSION 2.8.3)
project(allegro_hand_description)

find_package(catkin REQUIRED)

# Declare a package.
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES handover_viz
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS python-scipy python-matplotlib
)

#############
## Install ##
#############

## Install all the xacro & mesh files
install(DIRECTORY .
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/
  FILES_MATCHING
  PATTERN "*.xacro"
  PATTERN "*.world"
  PATTERN "*.STL"
)

catkin_install_python(PROGRAMS scripts/detect_pcan.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
