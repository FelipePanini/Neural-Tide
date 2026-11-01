#version 330
// Ribbon of light: soft body in the finger's colour with a hot white-ish core.
uniform vec3 u_highlight;
uniform float u_intensity;
uniform float u_core;
in vec3 v_color;
in float v_alpha;
in float v_edge;
out vec4 f_color;

void main() {
    float body = 1.0 - smoothstep(0.3, 1.0, abs(v_edge));
    float core = exp(-v_edge * v_edge * 12.0);
    f_color = vec4((v_color * body * u_intensity + u_highlight * core * u_core) * v_alpha, 1.0);
}
