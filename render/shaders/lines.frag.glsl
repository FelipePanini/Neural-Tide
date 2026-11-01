#version 330
uniform int u_premultiply;  // 1 when drawn with additive blending
in vec4 v_color;
in float v_edge;
in float v_feather;
out vec4 f_color;

void main() {
    float alpha = 1.0 - smoothstep(1.0 - v_feather, 1.0, abs(v_edge));
    float a = v_color.a * alpha;
    f_color = u_premultiply == 1 ? vec4(v_color.rgb * a, a) : vec4(v_color.rgb, a);
}
