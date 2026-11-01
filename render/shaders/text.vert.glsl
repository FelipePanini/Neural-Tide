#version 330
uniform vec2 u_screen;  // target size in pixels
in vec2 in_pos;         // pixels, origin top-left
in vec2 in_uv;
out vec2 v_uv;

void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos.x / u_screen.x * 2.0 - 1.0, 1.0 - in_pos.y / u_screen.y * 2.0, 0.0, 1.0);
}
