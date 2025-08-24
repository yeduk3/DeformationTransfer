cmake_minimum_required(VERSION 3.10)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(DeformationTransferForTriangleMeshes)

# Use C++17 Standard for ensuring <filesystem> stl.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 실행 파일이 생성될 디렉토리를 명시적으로 지정합니다.
# 보통 build 폴더 내에 bin 폴더를 만들어 사용합니다.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(main main.cpp)

if(APPLE)
    target_link_libraries(main PUBLIC
        "-framework OpenGL"
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreVideo"
    )
endif()

# Those paths should be modified,,,
target_include_directories(main PRIVATE /Users/gyu/Desktop/YGL/include)
target_link_libraries(main PRIVATE
    /Users/gyu/Desktop/YGL/library/libGLEW.2.2.0.dylib
    /Users/gyu/Desktop/YGL/library/libglfw.3.4.dylib
)

# --- 리소스 복사 부분 수정 ---

set(RESOURCE_DIR ${CMAKE_SOURCE_DIR}/resources)
file(GLOB_RECURSE RESOURCE_FILES "${RESOURCE_DIR}/*")

add_custom_target(copy_resources ALL
    # 2. $<TARGET_FILE_DIR:main> 대신 위에서 지정한 변수를 사용합니다.
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${RESOURCE_DIR} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resources
    SOURCES ${RESOURCE_FILES}
    COMMENT "Copying resources to build directory"
)

add_dependencies(main copy_resources)
