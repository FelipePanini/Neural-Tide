#version 330
uniform sampler2D u_texture;
uniform float u_opacity;
in vec2 v_uv;
out vec4 f_color;

void main() {
    vec4 texel = texture(u_texture, v_uv);
    f_color = vec4(texel.rgb, texel.a * u_opacity);
}
