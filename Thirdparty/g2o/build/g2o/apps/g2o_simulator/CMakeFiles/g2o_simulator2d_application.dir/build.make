# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/build

# Include any dependencies generated for this target.
include g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/depend.make

# Include the progress variables for this target.
include g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/progress.make

# Include the compile flags for this target's objects.
include g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/flags.make

g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/test_simulator2d.cpp.o: g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/flags.make
g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/test_simulator2d.cpp.o: ../g2o/apps/g2o_simulator/test_simulator2d.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/test_simulator2d.cpp.o"
	cd /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/build/g2o/apps/g2o_simulator && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/g2o_simulator2d_application.dir/test_simulator2d.cpp.o -c /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/g2o/apps/g2o_simulator/test_simulator2d.cpp

g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/test_simulator2d.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/g2o_simulator2d_application.dir/test_simulator2d.cpp.i"
	cd /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/build/g2o/apps/g2o_simulator && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/g2o/apps/g2o_simulator/test_simulator2d.cpp > CMakeFiles/g2o_simulator2d_application.dir/test_simulator2d.cpp.i

g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/test_simulator2d.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/g2o_simulator2d_application.dir/test_simulator2d.cpp.s"
	cd /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/build/g2o/apps/g2o_simulator && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/g2o/apps/g2o_simulator/test_simulator2d.cpp -o CMakeFiles/g2o_simulator2d_application.dir/test_simulator2d.cpp.s

g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/test_simulator2d.cpp.o.requires:
.PHONY : g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/test_simulator2d.cpp.o.requires

g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/test_simulator2d.cpp.o.provides: g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/test_simulator2d.cpp.o.requires
	$(MAKE) -f g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/build.make g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/test_simulator2d.cpp.o.provides.build
.PHONY : g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/test_simulator2d.cpp.o.provides

g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/test_simulator2d.cpp.o.provides.build: g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/test_simulator2d.cpp.o

# Object files for target g2o_simulator2d_application
g2o_simulator2d_application_OBJECTS = \
"CMakeFiles/g2o_simulator2d_application.dir/test_simulator2d.cpp.o"

# External object files for target g2o_simulator2d_application
g2o_simulator2d_application_EXTERNAL_OBJECTS =

../bin/g2o_simulator2d: g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/test_simulator2d.cpp.o
../bin/g2o_simulator2d: g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/build.make
../bin/g2o_simulator2d: ../lib/libg2o_simulator.so
../bin/g2o_simulator2d: ../lib/libg2o_types_slam2d_addons.so
../bin/g2o_simulator2d: ../lib/libg2o_types_slam2d.so
../bin/g2o_simulator2d: ../lib/libg2o_core.so
../bin/g2o_simulator2d: ../lib/libg2o_types_slam3d_addons.so
../bin/g2o_simulator2d: ../lib/libg2o_types_slam3d.so
../bin/g2o_simulator2d: ../lib/libg2o_core.so
../bin/g2o_simulator2d: ../lib/libg2o_stuff.so
../bin/g2o_simulator2d: ../lib/libg2o_opengl_helper.so
../bin/g2o_simulator2d: /usr/lib/x86_64-linux-gnu/libGLU.so
../bin/g2o_simulator2d: /usr/lib/x86_64-linux-gnu/libGL.so
../bin/g2o_simulator2d: /usr/lib/x86_64-linux-gnu/libSM.so
../bin/g2o_simulator2d: /usr/lib/x86_64-linux-gnu/libICE.so
../bin/g2o_simulator2d: /usr/lib/x86_64-linux-gnu/libX11.so
../bin/g2o_simulator2d: /usr/lib/x86_64-linux-gnu/libXext.so
../bin/g2o_simulator2d: g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ../../../../bin/g2o_simulator2d"
	cd /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/build/g2o/apps/g2o_simulator && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/g2o_simulator2d_application.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/build: ../bin/g2o_simulator2d
.PHONY : g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/build

g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/requires: g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/test_simulator2d.cpp.o.requires
.PHONY : g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/requires

g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/clean:
	cd /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/build/g2o/apps/g2o_simulator && $(CMAKE_COMMAND) -P CMakeFiles/g2o_simulator2d_application.dir/cmake_clean.cmake
.PHONY : g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/clean

g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/depend:
	cd /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/g2o/apps/g2o_simulator /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/build /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/build/g2o/apps/g2o_simulator /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/build/g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : g2o/apps/g2o_simulator/CMakeFiles/g2o_simulator2d_application.dir/depend
