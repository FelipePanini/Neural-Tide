#version 330
#include "common.glsl"
in vec2 in_corner;      // unit quad corner, -1..1
in vec2 in_center;      // world space
in float in_radius;     // ring radius, world units
in float in_progress;   // 0..1
in float in_marker;     // stage tick as a fraction of the ring, < 0 = none
in float in_alpha;
in float in_flash;      // 1 → 0 after the hold completes
in vec3 in_color;

uniform float u_quad_scale;

out vec2 v_local;       // in ring radii: the ring sits at length 1
out float v_progress;
out float v_marker;
out float v_alpha;
out float v_flash;
out vec3 v_color;

void main() {
    v_local = in_corner * u_quad_scale;
    v_progress = in_progress;
    v_marker = in_marker;
    v_alpha = in_alpha;
    v_flash = in_flash;
    v_color = in_color;
    gl_Position = world_to_clip(in_center + in_corner * in_radius * u_quad_scale);
}
