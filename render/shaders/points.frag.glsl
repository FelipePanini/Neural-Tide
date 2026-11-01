#version 330
// Soft round dot with a faint halo.
in vec4 v_color;
out vec4 f_color;

void main() {
    float d = length(gl_PointCoord - 0.5) * 2.0;
    float core = 1.0 - smoothstep(0.45, 0.75, d);
    float halo = (1.0 - smoothstep(0.0, 1.0, d)) * 0.35;
    float alpha = max(core, halo) * v_color.a;
    if (alpha <= 0.002) discard;
    f_color = vec4(v_color.rgb, alpha);
}
