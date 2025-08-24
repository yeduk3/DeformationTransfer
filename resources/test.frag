#version 410 core


in vec3 VNormal;


out vec4 FragColor;


void main() {
    FragColor = vec4(VNormal, 1);
}
