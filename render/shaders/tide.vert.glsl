#version 330
#include "common.glsl"
// One instanced quad per particle, stretched along its velocity (motion blur).
in vec2 in_corner;  // unit quad corner, -1..1 (x along the motion, y across)
in vec2 in_pos;
in vec2 in_vel;
in vec2 in_info;    // x = life alpha, y = colour seed 0..1

uniform float u_px;            // world units per pixel
uniform float u_size_px;
uniform float u_streak_s;
uniform float u_max_streak_px;
uniform float u_fast_speed;
uniform float u_intensity;
uniform vec3 u_primary;
uniform vec3 u_secondary;
uniform vec3 u_highlight;
uniform float u_secondary_share;  // fraction of particles in the secondary colour

out vec2 v_local;
out vec3 v_color;

void main() {
    float speed = length(in_vel);
    vec2 dir = speed > 1e-5 ? in_vel / speed : vec2(1.0, 0.0);
    vec2 across = vec2(-dir.y, dir.x);
    float streak = min(speed * u_streak_s / u_px, u_max_streak_px) * u_px;
    float half_width = 0.5 * u_size_px * u_px;
    float half_length = half_width + 0.5 * streak;
    vec2 center = in_pos - dir * 0.5 * streak;  // the streak trails behind the particle
    gl_Position = world_to_clip(center + dir * in_corner.x * half_length + across * in_corner.y * half_width);

    float fast = smoothstep(0.0, u_fast_speed, speed);
    float secondary = smoothstep(1.0 - u_secondary_share - 0.15, 1.0 - u_secondary_share + 0.15, in_info.y);
    vec3 base = mix(u_primary, u_secondary, secondary);
    v_color = mix(base, u_highlight, fast * fast) * (mix(0.5, 1.25, fast) * in_info.x * u_intensity);
    v_local = in_corner;
}
