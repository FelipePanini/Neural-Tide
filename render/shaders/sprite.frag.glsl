#version 330
// Tinted glyph from the atlas, premultiplied for additive blending.
uniform sampler2D u_atlas;
in vec2 v_uv;
in vec3 v_color;
out vec4 f_color;

void main() {
    f_color = vec4(v_color * texture(u_atlas, v_uv).a, 1.0);
}
