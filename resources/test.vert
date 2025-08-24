#version 410 core

layout(location = 0) in vec3 Position;
layout(location = 1) in vec3 Normal;

uniform mat4 MVP;
uniform mat3 NormalMat;

out vec3 VNormal;

void main() {
    VNormal = NormalMat * Normal;
    gl_Position = MVP * vec4(Position, 1);
}
