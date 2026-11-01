#version 330
// Height-field surface laid out on the ground plane and seen in perspective.
uniform mat4 u_view_proj;
uniform vec2 u_origin;       // world x of column 0, world z of row 0
uniform vec2 u_cell;         // world size of one cell along x and along -z
uniform float u_height_scale;
uniform vec3 u_eye;

in vec2 in_grid;    // (column, row)
in float in_height;

out vec2 v_grid;
out float v_height;
out float v_dist;

void main() {
    vec3 p = vec3(u_origin.x + in_grid.x * u_cell.x, in_height * u_height_scale, u_origin.y - in_grid.y * u_cell.y);
    v_grid = in_grid;
    v_height = in_height;
    v_dist = distance(p, u_eye);
    gl_Position = u_view_proj * vec4(p, 1.0);
}
