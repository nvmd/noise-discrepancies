
set(LIBRARY_SOURCES regular_segmentation.cpp regular_segmentation.hpp)
include_directories(.)
source_group(REGULAR_SEGMENTATION FILES ${LIBRARY_SOURCES})

add_library(regular_segmentation ${LIBRARY_SOURCES})
