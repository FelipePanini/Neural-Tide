#version 330
// Effect layer + bloom, tone-mapped and added on top of the background.
uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_bloom_intensity;
uniform float u_exposure;
uniform float u_whiteout;  // how much the brightest light bleaches towards white
uniform float u_dither;
uniform float u_seed;

in vec2 v_uv;
out vec4 f_color;

void main() {
    vec3 hdr = texture(u_scene, v_uv).rgb + texture(u_bloom, v_uv).rgb * u_bloom_intensity;
    // Tone map the peak channel and keep the hue, so dense neon stays neon
    // instead of washing out; only the very brightest spots bleach a little.
    float peak = max(hdr.r, max(hdr.g, hdr.b));
    float mapped = 1.0 - exp(-peak * u_exposure);
    vec3 color = hdr / max(peak, 1e-5) * mapped;
    color = mix(color, vec3(mapped), smoothstep(0.75, 1.0, mapped) * u_whiteout);
    float noise = fract(sin(dot(gl_FragCoord.xy + u_seed, vec2(12.9898, 78.233))) * 43758.5453);
    color += (noise - 0.5) * u_dither;
    f_color = vec4(max(color, 0.0), 1.0);
}
