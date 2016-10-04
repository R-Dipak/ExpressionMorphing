#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#


cmake_minimum_required(VERSION 2.8.4)

PROJECT(examples)

include (../dlib-18.18/dlib/cmake)

# Tell CMake to compile a program.  We do this with the ADD_EXECUTABLE()
# statement which takes the name of the output executable and then a list of
# .cpp files to compile.  Here each example consists of only one .cpp file but
# in general you will make programs that const of many .cpp files.
#ADD_EXECUTABLE(exec fl.cpp)
# Then we tell it to link with dlib.
#TARGET_LINK_LIBRARIES(fl dlib)




# Since there are a lot of examples I'm going to use a macro to simply this
# CMakeLists.txt file.  However, usually you will create only one executable in
# your cmake projects and use the syntax shown above.
MACRO(add_example name)
   ADD_EXECUTABLE(${name} ${name}.cpp)
   find_package( OpenCV REQUIRED )
   TARGET_LINK_LIBRARIES(${name} dlib ${OpenCV_LIBS} )
ENDMACRO()


#here we apply our macros 
add_example(source)

#find_package( OpenCV REQUIRED )
#add_executable( DisplayImage DisplayImage.cpp )
#target_link_libraries(  ${OpenCV_LIBS} )

find_package(OpenCV)
if (OpenCV_FOUND)
   include_directories(${OpenCV_INCLUDE_DIRS})

 #  ADD_EXECUTABLE(webcam_face_pose_ex webcam_face_pose_ex.cpp)
 #  TARGET_LINK_LIBRARIES(webcam_face_pose_ex dlib ${OpenCV_LIBS} )
endif()


if (DLIB_LINK_WITH_SQLITE3)
   add_example(sqlite_ex)
endif()


