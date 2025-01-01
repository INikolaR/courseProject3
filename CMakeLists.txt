PROJECT(CourseProject)

CMAKE_MINIMUM_REQUIRED(VERSION 3.11)
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

SET(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
ADD_SUBDIRECTORY(googletest EXCLUDE_FROM_ALL)

ENABLE_TESTING()

INCLUDE_DIRECTORIES(Eigen)

SET(SOURCES LossFunction.cpp ActivationFunction.cpp GivensLayer.cpp Net.cpp Random.cpp VectorOperations.cpp test.cpp exception.cpp MatrixLayer.cpp)
ADD_LIBRARY(lib STATIC ${SOURCES})
ADD_EXECUTABLE(CourseProject main.cpp)
TARGET_LINK_LIBRARIES(CourseProject lib)

ADD_EXECUTABLE(test_defaults tests/Test.cpp)
TARGET_INCLUDE_DIRECTORIES(test_defaults PUBLIC ${CMAKE_SOURCE_DIR})
TARGET_LINK_LIBRARIES(test_defaults gtest gtest_main lib)
ADD_TEST(NAME test_defaults COMMAND test_defaults)
