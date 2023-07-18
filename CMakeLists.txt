cmake_minimum_required(VERSION 3.5)

project(AXonColorOpenCV LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(
    CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_HOME_DIRECTORY}/bin
    )
add_subdirectory(/home/ginseng/Documents/chipizubov/libs/dlib/dlib dlib_build)
set(CMAKE_CURRENT_SOURCE_DIR ./)
file(GLOB SOURCES *.cpp *.h)

# Пути к библиотеке OpenNI2
set(OPENNI2_INCLUDE ${CMAKE_HOME_DIRECTORY}/libs/Include/)
set(OPENNI2_REDIST ${CMAKE_HOME_DIRECTORY}/libs/Redist/)

find_package( OpenCV REQUIRED )

add_executable(${PROJECT_NAME} main.cpp)
target_sources(${PROJECT_NAME} PRIVATE ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS} ${OPENNI2_INCLUDE} ./)
target_link_directories(${PROJECT_NAME} PRIVATE ${OPENNI2_REDIST})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} dlib::dlib libOpenNI2.so)
