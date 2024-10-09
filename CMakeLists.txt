PROJECT(CourseProject)

CMAKE_MINIMUM_REQUIRED(VERSION 3.11)

SET(CMAKE_CXX_STANDARD 17)

SET(CMAKE_CXX_STANDARD_REQUIRED ON)

SET(SOURCES LossFunction.cpp ActivationFunction.cpp GivensLayer.cpp GivensNet.cpp Random.cpp VectorOperations.cpp test.cpp exception.cpp)
ADD_LIBRARY(lib STATIC ${SOURCES})

ADD_EXECUTABLE(CourseProject main.cpp)

TARGET_LINK_LIBRARIES(CourseProject lib)
