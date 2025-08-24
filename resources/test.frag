#version 410 core


in vec3 VNormal;
in vec4 VPos;

out vec4 FragColor;

uniform mat4 V;
uniform vec3 lightPosition = vec3(10, 10, 5);
uniform vec3 lightColor = vec3(160);
uniform vec3 diffuseColor = vec3(1);
uniform vec3 specularColor = vec3(0.3);
uniform float shininess = 10.f;

vec4 phong() {
    vec4 lp = V * vec4(lightPosition, 1);
    vec3 l = lp.xyz/lp.w - VPos.xyz;
    vec3 L = normalize(l);
    vec3 N = normalize(VNormal);
    vec3 R = 2 * dot(L, N) * N - L;
    vec3 I = lightColor / dot(l, l);

    vec3 ambient = diffuseColor * vec3(0.1);
    vec3 diffuse = I * diffuseColor * max(dot(L, N), 0);
    vec3 specular = I * specularColor * pow(max(R.z, 0), shininess);

    vec3 color = ambient + diffuse + specular;

    return vec4(pow(color, vec3(1/2.2)), 1);
}

void main() {
    FragColor = phong();
}
