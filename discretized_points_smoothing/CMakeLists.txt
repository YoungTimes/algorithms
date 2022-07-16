cmake_minimum_required(VERSION 3.0)


PROJECT(Smooth)

SET(OSQP_INCLUDE_DIRS, ~/Documents/algorithms/smooth/OSQP/include)
SET(OSQP_LIBRARY_DIRS, ~/Documents/algorithms/smooth/OSQP/lib)

include_directories(
	./
)
include_directories(
	${OSQP_INCLUDE_DIRS}
)

link_directories(
	${OSQP_LIBRARY_DIRS}
)

add_executable(${PROJECT_NAME} main.cpp fem_pos_deviation_osqp_interface.cpp fem_pos_deviation_sqp_osqp_interface.cpp)

target_link_libraries(${PROJECT_NAME} osqp.so)

install(TARGETS ${PROJECT_NAME}
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION libstatic
)