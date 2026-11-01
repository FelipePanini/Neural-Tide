#version 330
// Effect layer + bloom, tone-mapped and added on top of the background.
// A mode change sends a ripple through the layer: pixels near the expanding
// ring are displaced radially and its leading edge glows a little.
uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_bloom_intensity;
uniform float u_exposure;
uniform float u_whiteout;  // how much the brightest light bleaches towards white
uniform float u_dither;
uniform float u_seed;
uniform float u_aspect;
uniform vec2 u_ripple_center;  // uv
uniform float u_ripple_radius;   // screen heights
uniform float u_ripple_width;
uniform float u_ripple_strength;
uniform vec3 u_ripple_color;     // rim colour, already scaled

in vec2 v_uv;
out vec4 f_color;

void main() {
    vec2 d = (v_uv - u_ripple_center) * vec2(u_aspect, 1.0);
    float dist = length(d);
    float x = (dist - u_ripple_radius) / u_ripple_width;
    float wave = exp(-x * x);
    vec2 dir = dist > 1e-5 ? d / dist : vec2(0.0);
    vec2 uv = v_uv - dir * (wave * u_ripple_strength) / vec2(u_aspect, 1.0);

    vec3 hdr = texture(u_scene, uv).rgb + texture(u_bloom, uv).rgb * u_bloom_intensity + u_ripple_color * wave;
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
