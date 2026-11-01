#version 330
// Dual-filter downsample (5 taps). The first pass also keeps only the bright parts.
uniform sampler2D u_src;
uniform vec2 u_texel;  // texel size of u_src
uniform int u_prefilter;
uniform float u_threshold;
uniform float u_knee;

in vec2 v_uv;
out vec4 f_color;

vec3 soft_threshold(vec3 c) {
    float brightness = max(c.r, max(c.g, c.b));
    float rq = clamp(brightness - u_threshold + u_knee, 0.0, 2.0 * u_knee);
    rq = rq * rq / (4.0 * u_knee + 1e-5);
    return c * max(rq, brightness - u_threshold) / max(brightness, 1e-5);
}

void main() {
    vec3 s = texture(u_src, v_uv).rgb * 4.0;
    s += texture(u_src, v_uv - u_texel).rgb;
    s += texture(u_src, v_uv + u_texel).rgb;
    s += texture(u_src, v_uv + vec2(u_texel.x, -u_texel.y)).rgb;
    s += texture(u_src, v_uv - vec2(u_texel.x, -u_texel.y)).rgb;
    s *= 0.125;
    if (u_prefilter == 1) s = soft_threshold(s);
    f_color = vec4(s, 1.0);
}
