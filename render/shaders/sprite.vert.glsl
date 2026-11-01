#version 330
#include "common.glsl"
in vec2 in_corner;     // unit quad corner, -1..1
in vec2 in_center;     // world space
in vec2 in_half_size;  // world units
in vec4 in_uv;         // u0, v0, u1, v1 in the atlas
in vec3 in_color;      // tint × intensity
in float in_angle;     // radians

out vec2 v_uv;
out vec3 v_color;

void main() {
    vec2 local = in_corner * in_half_size;
    float s = sin(in_angle);
    float c = cos(in_angle);
    gl_Position = world_to_clip(in_center + vec2(c * local.x - s * local.y, s * local.x + c * local.y));
    v_uv = mix(in_uv.xy, in_uv.zw, in_corner * 0.5 + 0.5);
    v_color = in_color;
}
