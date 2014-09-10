
set(LIBRARY_SOURCES SLIC.cpp SLIC.h)
include_directories(.)
source_group(SLIC FILES ${LIBRARY_SOURCES})

add_library(slic ${LIBRARY_SOURCES})
