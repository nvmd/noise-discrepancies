
set(LIBRARY_SOURCES slic_segmentation.cpp slic_segmentation.hpp)
include_directories(.)
include_directories(..)
source_group(SLIC_SEGMENTATION FILES ${LIBRARY_SOURCES})

add_library(slic_segmentation ${LIBRARY_SOURCES})
