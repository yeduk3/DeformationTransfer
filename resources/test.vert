#version 410 core

layout(location = 0) in vec3 Position;
layout(location = 1) in vec3 Normal;

uniform mat4 MVP;
uniform mat4 MV;
uniform mat3 NormalMat;

out vec3 WNormal;
out vec3 VNormal;
out vec4 VPos;

void main() {
    WNormal = Normal;
    VNormal = NormalMat * Normal;
    vec4 pos4 = vec4(Position, 1);
    VPos = MV * pos4;
    gl_Position = MVP * pos4;
}
