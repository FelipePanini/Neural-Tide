#version 330
#include "common.glsl"
in vec2 in_pos;
in vec3 in_color;
in float in_alpha;
in float in_edge;  // -1 / +1 across the ribbon
out vec3 v_color;
out float v_alpha;
out float v_edge;

void main() {
    v_color = in_color;
    v_alpha = in_alpha;
    v_edge = in_edge;
    gl_Position = world_to_clip(in_pos);
}
