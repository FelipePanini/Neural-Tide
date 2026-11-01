#version 330
#include "common.glsl"
in vec2 in_pos;
in vec4 in_color;
in float in_edge;     // -1 / +1 across the line
in float in_feather;  // fraction of the half-width that fades out (anti-aliasing)
out vec4 v_color;
out float v_edge;
out float v_feather;

void main() {
    v_color = in_color;
    v_edge = in_edge;
    v_feather = in_feather;
    gl_Position = world_to_clip(in_pos);
}
