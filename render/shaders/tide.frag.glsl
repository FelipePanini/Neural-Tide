#version 330
// Soft capsule: bright core, feathered edges.
in vec2 v_local;
in vec3 v_color;
out vec4 f_color;

void main() {
    float across = 1.0 - smoothstep(0.2, 1.0, abs(v_local.y));
    float along = 1.0 - smoothstep(0.55, 1.0, abs(v_local.x));
    f_color = vec4(v_color * across * along, 1.0);
}
