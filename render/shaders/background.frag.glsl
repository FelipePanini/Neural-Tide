#version 330
// Dimmed, cool-tinted camera feed behind the effects.
uniform sampler2D u_camera;
uniform vec4 u_uv_rect;      // visible image region: u0, v0, du, dv (v points down)
uniform float u_visibility;  // eased 0..1, toggled with C
uniform float u_dim;
uniform float u_desaturate;
uniform vec3 u_tint;
uniform float u_vignette;

in vec2 v_uv;
out vec4 f_color;

void main() {
    vec2 uv = vec2(u_uv_rect.x + v_uv.x * u_uv_rect.z, u_uv_rect.y + (1.0 - v_uv.y) * u_uv_rect.w);
    vec3 cam = texture(u_camera, uv).rgb;
    float luma = dot(cam, vec3(0.299, 0.587, 0.114));
    cam = mix(cam, vec3(luma), u_desaturate) * u_tint * u_dim;
    vec2 c = v_uv - 0.5;
    float vignette = clamp(1.0 - u_vignette * dot(c, c) * 2.0, 0.0, 1.0);
    f_color = vec4(cam * vignette * u_visibility, 1.0);
}
