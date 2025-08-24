#include "error.hpp"
#include <YGLWindow.hpp>
YGLWindow* window;
#include <objreader.hpp>
ObjData cat;
#include <program.hpp>
Program shader;
#include <camera.hpp>
#include <filesystem>

std::filesystem::path resPath;
std::filesystem::path catPath;

void init() {
    cat.loadObject(catPath.string().c_str(), "cat-reference.obj");
    cat.adjustCenter(true);
    cat.generateBuffers();
    shader.loadShader((resPath.string() + "/test.vert").c_str(), (resPath.string() + "/test.frag").c_str());
    camera.setPosition({-2, 0, 0});
    camera.glfwSetCallbacks(window->getGLFWWindow());




    glEnable(GL_DEPTH_TEST);


    glErr("after init");
}

void render() {
    glViewport(0, 0, window->width(), window->height());
    glClearColor(0, 0, 0.3f, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    
    glm::mat4 M(1);
    glm::mat4 V = camera.lookAt();
    glm::mat4 P = camera.perspective(window->aspect(), 0.1f, 1000.f);
    glm::mat4 MV = V * M;

    shader.use();
    shader.setUniform("NormalMat", glm::mat3(MV[0], MV[1], MV[2]));
    shader.setUniform("MVP", P * MV);
    glErr("after set uniforms");

    cat.render();
    glErr("after render a cat");
}

int main(int argc, char *argv[]) {
    std::filesystem::path exePath = argv[0];
    std::filesystem::path buildPath = exePath.parent_path();
    resPath = buildPath / "resources";
    catPath = resPath / "cat-poses";

    window = new YGLWindow(640, 480, "Deformation Transfer For Triangle Meshes");
    window->mainLoop(init, render);
    return 0;
}


