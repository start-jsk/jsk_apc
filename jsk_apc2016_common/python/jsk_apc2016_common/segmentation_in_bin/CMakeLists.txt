cython_add_module(segmentation_in_bin_util segmentation_in_bin_util.pyx)
set_target_properties(
  segmentation_in_bin_util PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_PYTHON_DESTINATION}/${PROJECT_NAME})
install(TARGETS segmentation_in_bin_util
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
