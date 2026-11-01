#version 330
// Soft gaussian light blob, drawn additively.
in vec2 v_local;
in vec3 v_color;
out vec4 f_color;

void main() {
    float d2 = dot(v_local, v_local);
    float light = exp(-d2 * 4.0) * (1.0 - smoothstep(0.8, 1.0, d2));
    f_color = vec4(v_color * light, 1.0);
}
