#version 330
#include "common.glsl"
in vec2 in_corner;   // unit quad corner, -1..1
in vec2 in_center;   // per instance, world space
in float in_radius;  // world units
in vec3 in_color;    // already multiplied by intensity
out vec2 v_local;
out vec3 v_color;

void main() {
    v_local = in_corner;
    v_color = in_color;
    gl_Position = world_to_clip(in_center + in_corner * in_radius);
}
