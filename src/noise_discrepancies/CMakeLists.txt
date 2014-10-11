
set(DEMO_SOURCES main.cpp utils.cpp)
include_directories(.)
include_directories(..)
source_group(noise_discrepancies FILES ${DEMO_SOURCES})

add_executable(noise_discrepancies ${DEMO_SOURCES})
target_link_libraries(noise_discrepancies ${OpenCV_LIBS}
                                          slic
                                          slic_segmentation)
