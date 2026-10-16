cmake_minimum_required(VERSION 3.16)
project(wutxi_upscaler CXX)

set(CMAKE_CXX_STANDARD 17)
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/third_party/ncnn-install/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_executable(upscaler WIN32 src/main.cpp)
target_link_libraries(upscaler PRIVATE ncnn windowscodecs ole32 shell32 comdlg32 gdi32 user32 uuid advapi32)
target_link_options(upscaler PRIVATE -municode -static -static-libgcc -static-libstdc++)
set_target_properties(upscaler PROPERTIES OUTPUT_NAME "Upscaler" RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

if(EXISTS ${CMAKE_SOURCE_DIR}/models)
    add_custom_command(TARGET upscaler POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/models ${CMAKE_SOURCE_DIR}/bin/models)
endif()
