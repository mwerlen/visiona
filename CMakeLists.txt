cmake_minimum_required(VERSION 3.11)

project(detect)

## dependencies
include_directories(/usr/include/eigen3)
include_directories(include)
include_directories(/usr/lib/x86_64-linux-gnu  /usr/include/gtk-2.0/gtk /usr/include/glib-2.0/)

## detectlib
set(CMAKE_ENABLE_EXPORTS True)
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS "-std=c++11 -o3 -ggdb ${OTHERS_DEFINE} -static-libgcc -static-libstdc++")
#set(CMAKE_CXX_FLAGS "-std=c++11 -o3 -ggdb ${OTHERS_DEFINE}")
#set(CMAKE_EXE_LINKER_FLAGS "-static")

#Static Libs
set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})

# Visiona
add_library(visiona STATIC src/Visiona.cpp src/DebugPlotConfig.cpp src/MarkerDetector_impl.cpp src/MarkerDetectorConfig.cpp src/Timer.cpp)

# config ++
add_library(config++ STATIC IMPORTED)
set_target_properties(config++ PROPERTIES IMPORTED_LOCATION /usr/local/src/libconfig/lib/.libs/libconfig++.a)

#OpenCV
add_library(opencv_core STATIC IMPORTED)
set_target_properties(opencv_core PROPERTIES IMPORTED_LOCATION /usr/local/src/opencv2/build/lib/libopencv_core.a)

add_library(opencv_calib3d STATIC IMPORTED)
set_target_properties(opencv_calib3d PROPERTIES IMPORTED_LOCATION /usr/local/src/opencv2/build/lib/libopencv_calib3d.a)
target_link_libraries(opencv_calib3d INTERFACE opencv_core)

add_library(opencv_higui STATIC IMPORTED)
set_target_properties(opencv_higui PROPERTIES IMPORTED_LOCATION /usr/local/src/opencv2/build/lib/libopencv_higui.a)
target_link_libraries(opencv_higui INTERFACE opencv_core)

add_library(opencv_imgproc STATIC IMPORTED)
set_target_properties(opencv_imgproc PROPERTIES IMPORTED_LOCATION /usr/local/src/opencv2/build/lib/libopencv_imgproc.a)
target_link_libraries(opencv_imgproc INTERFACE opencv_core -static-libgcc -static-libstdc++)

# Dependencies
add_library(libIlmImf STATIC IMPORTED)
set_target_properties(libIlmImf PROPERTIES IMPORTED_LOCATION /usr/share/OpenCV/3rdparty/lib/libIlmImf.a)
add_library(libjasper STATIC IMPORTED)
set_target_properties(libjasper PROPERTIES IMPORTED_LOCATION /usr/local/src/opencv2/build/3rdparty/lib/liblibjasper.a)
add_library(jpeg STATIC IMPORTED)
set_target_properties(jpeg PROPERTIES IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libjpeg.a)
add_library(png STATIC IMPORTED)
set_target_properties(png PROPERTIES IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libpng.a)
add_library(libm STATIC IMPORTED)
set_target_properties(libm PROPERTIES IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libm.a)
add_library(libc STATIC IMPORTED)
set_target_properties(libc PROPERTIES IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libc_nonshared.a)
add_library(libc_nonshared STATIC IMPORTED)
set_target_properties(libc_nonshared PROPERTIES IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libc.a)
add_library(libz STATIC IMPORTED)
set_target_properties(libz PROPERTIES IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libz.a)
add_library(libstdc++ STATIC IMPORTED)
set_target_properties(libstdc++ PROPERTIES IMPORTED_LOCATION /usr/lib/gcc/x86_64-linux-gnu/8/libstdc++.a)

# Linking
target_link_libraries(visiona config++ opencv_core opencv_calib3d opencv_highgui opencv_imgproc)
target_link_libraries(visiona libIlmImf jpeg z pthread tiff lzma jbig png libjasper)
target_link_libraries(visiona libgtk-x11-2.0.so libgdk-x11-2.0.so libgobject-2.0.so libglib-2.0.so)
target_link_libraries(visiona libc libz libstdc++)
#target_link_libraries(visiona /usr/include/x86_64-linux-gnu/sys/signal.h)
#target_link_libraries(visiona /usr/include/x86_64-linux-gnu/sys/signal.h /usr/include/gtk-2.0/gtk/gtksignal.h /usr/include/gtk-2.0/gtk/gtkenums.h /usr/include/glib-2.0/glib-object.h /usr/include/glib-2.0/gobject/gbinding.h)

## Executables
add_executable(batchdetector src/batchdetector.cpp)
target_link_libraries(batchdetector visiona)

#add_executable(simpledetector src/simpledetector.cpp)
#target_link_libraries(simpledetector visiona)

#add_executable(testdetector src/testdetector.cpp)
#target_link_libraries(testdetector visiona) 
