#version 330
// Anti-aliased wireframe drawn from the grid coordinates (constant pixel
// width at any distance), lit up on wave crests and faded by distance fog.
uniform vec3 u_primary;
uniform vec3 u_secondary;
uniform vec3 u_highlight;
uniform float u_line_px;
uniform float u_line_intensity;
uniform float u_crest_intensity;
uniform float u_fill_intensity;
uniform float u_fog_density;
uniform float u_cols;
uniform float u_edge_fade;
uniform float u_intensity;

in vec2 v_grid;
in float v_height;
in float v_dist;
out vec4 f_color;

void main() {
    vec2 w = max(fwidth(v_grid), vec2(1e-5));
    vec2 to_line = abs(fract(v_grid - 0.5) - 0.5) / w;  // pixels to the nearest grid line
    float line = 1.0 - smoothstep(0.5 * u_line_px, 0.5 * u_line_px + 1.0, min(to_line.x, to_line.y));
    line *= 1.0 - smoothstep(0.3, 0.8, max(w.x, w.y));  // fade where cells shrink below ~2 px (no moiré)

    float crest = smoothstep(0.02, 0.3, v_height);
    float trough = smoothstep(0.02, 0.3, -v_height);
    vec3 color = mix(mix(u_primary, u_secondary, trough), u_highlight, crest);
    float edge = clamp(min(v_grid.x, u_cols - v_grid.x) / (u_cols * u_edge_fade), 0.0, 1.0);
    float fog = exp(-v_dist * u_fog_density);
    float light = line * (u_line_intensity + crest * u_crest_intensity) + u_fill_intensity * (crest + 0.3 * trough);
    f_color = vec4(color * light * fog * edge * u_intensity, 1.0);
}
