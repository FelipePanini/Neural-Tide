#version 330
// Progress ring: faint track, glowing filled arc (clockwise from the top),
// a bright head at the end of the arc, an optional stage tick and a burst
// that expands outwards when the hold completes. Drawn additively.
in vec2 v_local;
in float v_progress;
in float v_marker;
in float v_alpha;
in float v_flash;
in vec3 v_color;

uniform float u_half_width;  // half the stroke, in ring radii
uniform float u_time;

out vec4 f_color;

const float TAU = 6.28318530718;

void main() {
    float r = length(v_local);
    float aa = fwidth(r);
    float angle = fract(atan(v_local.x, v_local.y) / TAU + 1.0);  // 0 at the top, clockwise
    float band = 1.0 - smoothstep(u_half_width, u_half_width + aa, abs(r - 1.0));
    float filled = 1.0 - smoothstep(v_progress - 0.004, v_progress + 0.004, angle);
    float glow = exp(-pow((r - 1.0) / (u_half_width * 4.0), 2.0)) * filled;

    float head_angle = v_progress * TAU;
    vec2 head = vec2(sin(head_angle), cos(head_angle));
    float pulse = 0.75 + 0.25 * sin(u_time * 9.0);
    float head_glow = exp(-dot(v_local - head, v_local - head) * 40.0) * step(0.001, v_progress) * pulse;

    float tick = 0.0;
    if (v_marker >= 0.0) {
        float da = abs(angle - v_marker);
        da = min(da, 1.0 - da);
        tick = (1.0 - smoothstep(0.006, 0.012, da)) * (1.0 - smoothstep(u_half_width * 2.5, u_half_width * 3.5, abs(r - 1.0)));
    }

    float t = 1.0 - v_flash;  // 0 → 1 while the burst expands
    float burst_r = 1.0 + 0.8 * t;
    float burst = v_flash * exp(-pow((r - burst_r) / (u_half_width * 3.0), 2.0));

    float light = band * (0.15 + 0.85 * filled) + glow * 0.45 + head_glow + tick * 0.8 + burst * 1.5;
    f_color = vec4(v_color * light * v_alpha, 1.0);
}
