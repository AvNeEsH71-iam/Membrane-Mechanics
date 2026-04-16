cmake_minimum_required(VERSION 3.0.0)
project(fluctuation-analysis VERSION 0.1.0 LANGUAGES CXX CUDA)
enable_language(CUDA)
include(CTest)
enable_testing()

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(${PROJECT_NAME} "main_1.4(stable).cpp" "time_autocorr_2.0(stable).cu" "time_crosscorr_0.0(stable).cu")

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
