#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org


cmake_minimum_required(VERSION 2.8.4)

PROJECT(examples)

include (../dlib-18.18/dlib/cmake)


MACRO(add_example name)
   ADD_EXECUTABLE(${name} ${name}.cpp)
   find_package( OpenCV REQUIRED )
   TARGET_LINK_LIBRARIES(${name} dlib ${OpenCV_LIBS} )
ENDMACRO()

add_example(fl)
add_example(Srcforimages)
add_example(drivervid)


find_package(OpenCV)
if (OpenCV_FOUND)
   include_directories(${OpenCV_INCLUDE_DIRS})

endif()

if (DLIB_LINK_WITH_SQLITE3)
   add_example(sqlite_ex)
endif()


