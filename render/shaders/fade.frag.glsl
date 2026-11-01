#version 330
// Trails: instead of clearing, last frame's light is diffused a little and decays.
uniform sampler2D u_prev;
uniform vec2 u_texel;
uniform float u_decay;      // exp(-dt / tau), frame-rate independent
uniform float u_floor;      // small subtraction so the tail reaches true black
uniform float u_diffusion;  // 0..1 blend towards the 4 neighbours

in vec2 v_uv;
out vec4 f_color;

void main() {
    vec3 c = texture(u_prev, v_uv).rgb;
    vec3 n = texture(u_prev, v_uv + vec2(u_texel.x, 0.0)).rgb
           + texture(u_prev, v_uv - vec2(u_texel.x, 0.0)).rgb
           + texture(u_prev, v_uv + vec2(0.0, u_texel.y)).rgb
           + texture(u_prev, v_uv - vec2(0.0, u_texel.y)).rgb;
    c = mix(c, n * 0.25, u_diffusion);
    f_color = vec4(max(c * u_decay - u_floor, 0.0), 1.0);
}
