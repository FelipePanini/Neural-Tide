#version 330
// Dual-filter upsample (8-tap tent), added onto the next larger level.
uniform sampler2D u_src;
uniform vec2 u_texel;  // texel size of u_src

in vec2 v_uv;
out vec4 f_color;

void main() {
    vec2 h = u_texel;
    vec3 s = texture(u_src, v_uv + vec2(-2.0 * h.x, 0.0)).rgb;
    s += texture(u_src, v_uv + vec2(-h.x, h.y)).rgb * 2.0;
    s += texture(u_src, v_uv + vec2(0.0, 2.0 * h.y)).rgb;
    s += texture(u_src, v_uv + vec2(h.x, h.y)).rgb * 2.0;
    s += texture(u_src, v_uv + vec2(2.0 * h.x, 0.0)).rgb;
    s += texture(u_src, v_uv + vec2(h.x, -h.y)).rgb * 2.0;
    s += texture(u_src, v_uv + vec2(0.0, -2.0 * h.y)).rgb;
    s += texture(u_src, v_uv + vec2(-h.x, -h.y)).rgb * 2.0;
    f_color = vec4(s / 12.0, 1.0);
}
