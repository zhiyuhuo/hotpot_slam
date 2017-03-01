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
include g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/depend.make

# Include the progress variables for this target.
include g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/progress.make

# Include the compile flags for this target's objects.
include g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/flags.make

g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/csparse_helper.cpp.o: g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/flags.make
g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/csparse_helper.cpp.o: ../g2o/solvers/csparse/csparse_helper.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/csparse_helper.cpp.o"
	cd /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/build/g2o/solvers/csparse && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/csparse_extension.dir/csparse_helper.cpp.o -c /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/g2o/solvers/csparse/csparse_helper.cpp

g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/csparse_helper.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/csparse_extension.dir/csparse_helper.cpp.i"
	cd /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/build/g2o/solvers/csparse && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/g2o/solvers/csparse/csparse_helper.cpp > CMakeFiles/csparse_extension.dir/csparse_helper.cpp.i

g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/csparse_helper.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/csparse_extension.dir/csparse_helper.cpp.s"
	cd /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/build/g2o/solvers/csparse && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/g2o/solvers/csparse/csparse_helper.cpp -o CMakeFiles/csparse_extension.dir/csparse_helper.cpp.s

g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/csparse_helper.cpp.o.requires:
.PHONY : g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/csparse_helper.cpp.o.requires

g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/csparse_helper.cpp.o.provides: g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/csparse_helper.cpp.o.requires
	$(MAKE) -f g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/build.make g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/csparse_helper.cpp.o.provides.build
.PHONY : g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/csparse_helper.cpp.o.provides

g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/csparse_helper.cpp.o.provides.build: g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/csparse_helper.cpp.o

# Object files for target csparse_extension
csparse_extension_OBJECTS = \
"CMakeFiles/csparse_extension.dir/csparse_helper.cpp.o"

# External object files for target csparse_extension
csparse_extension_EXTERNAL_OBJECTS =

../lib/libg2o_csparse_extension.so: g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/csparse_helper.cpp.o
../lib/libg2o_csparse_extension.so: g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/build.make
../lib/libg2o_csparse_extension.so: ../lib/libg2o_ext_csparse.so
../lib/libg2o_csparse_extension.so: g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared library ../../../../lib/libg2o_csparse_extension.so"
	cd /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/build/g2o/solvers/csparse && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/csparse_extension.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/build: ../lib/libg2o_csparse_extension.so
.PHONY : g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/build

g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/requires: g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/csparse_helper.cpp.o.requires
.PHONY : g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/requires

g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/clean:
	cd /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/build/g2o/solvers/csparse && $(CMAKE_COMMAND) -P CMakeFiles/csparse_extension.dir/cmake_clean.cmake
.PHONY : g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/clean

g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/depend:
	cd /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/g2o/solvers/csparse /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/build /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/build/g2o/solvers/csparse /home/rokid/catkin_ws/src/hotpot_slam/Thirdparty/g2o/build/g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : g2o/solvers/csparse/CMakeFiles/csparse_extension.dir/depend

