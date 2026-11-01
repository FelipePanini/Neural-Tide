#version 330
#include "common.glsl"
in vec2 in_pos;
in float in_size;  // pixels
in vec4 in_color;
out vec4 v_color;

void main() {
    v_color = in_color;
    gl_PointSize = in_size;
    gl_Position = world_to_clip(in_pos);
}
