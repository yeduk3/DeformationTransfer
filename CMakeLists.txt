cmake_minimum_required(VERSION 3.14)

project(DeformationTransferForTriangleMeshes)

# Use C++17 Standard for ensuring <filesystem> stl.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# GLFW
find_package(glfw3 3.4 REQUIRED)
# GLEW
find_package(GLEW 2.2 REQUIRED)
# Use Eigen Lib
find_package(Eigen3 3.4 REQUIRED)

# Use BFGS Lib
# include(FetchContent)
# FetchContent_Declare(
#     LBFGSpp_content
#     GIT_REPOSITORY https://github.com/yixuan/LBFGSpp.git
#     GIT_TAG v0.4.0
# )
# FetchContent_MakeAvailable(LBFGSpp_content)

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

# Eigen Includes
target_include_directories(main PRIVATE ${EIGEN3_INCLUDE_DIR})
# YGL Includes, those paths should be modified,,,?
target_include_directories(main PRIVATE /Users/gyu/Desktop/YGL/include)
# LBFG lib
# target_link_libraries(main PRIVATE Eigen3::Eigen LBFGSpp::LBFGSpp)
target_link_libraries(main PRIVATE
    Eigen3::Eigen
    # lbfgspp
    GLEW::GLEW
    glfw
)

# --- 리소스 복사 부분 수정 ---

set(RESOURCE_DIR ${CMAKE_SOURCE_DIR}/resources)

add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${RESOURCE_DIR} $<TARGET_FILE_DIR:main>/resources
    COMMENT "Copying resources to target directory"
)
