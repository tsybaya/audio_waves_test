cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(waves_test C)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

set(WAVES_TEST_SRC 
	src/main.c
	src/wav.c)

set (WAVES_TEST_HDR
	include/wav.h)
	
add_executable(${PROJECT_NAME} ${WAVES_TEST_SRC} ${WAVES_TEST_HDR})
target_include_directories(${PROJECT_NAME} PUBLIC "include/")
