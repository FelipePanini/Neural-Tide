#version 330
in vec4 v_color;
in float v_edge;
in float v_feather;
out vec4 f_color;

void main() {
    float alpha = 1.0 - smoothstep(1.0 - v_feather, 1.0, abs(v_edge));
    f_color = vec4(v_color.rgb, v_color.a * alpha);
}
