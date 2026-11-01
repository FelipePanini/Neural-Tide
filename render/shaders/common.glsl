// Shared helpers, included by other shaders right after their #version line.
uniform float u_aspect;  // window width / height

// World space: y in [-1, 1], x in [-aspect, aspect].
vec4 world_to_clip(vec2 p) {
    return vec4(p.x / u_aspect, p.y, 0.0, 1.0);
}
