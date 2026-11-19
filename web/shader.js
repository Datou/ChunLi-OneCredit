// Fullscreen WebGL2 rendering for the game texture.
export const VERT = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = vec2(aPos.x * 0.5 + 0.5, 0.5 - aPos.y * 0.5);
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

export const FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uVideo;
uniform vec2 uTexel;
uniform float uRows;
uniform float uActive;
uniform float uOpacity;
uniform float uDedither;
uniform float uSoften;

void main() {
  vec3 center = texture(uVideo, vUv).rgb;
  if (uActive < 0.5) {
    outColor = vec4(center, 1.0);
    return;
  }
  vec2 stepX = vec2(uTexel.x, 0.0);
  vec2 stepY = vec2(0.0, uTexel.y);
  vec3 nearby = (texture(uVideo, vUv - stepX).rgb + texture(uVideo, vUv + stepX).rgb
                + texture(uVideo, vUv - stepY).rgb + texture(uVideo, vUv + stepY).rgb) * 0.25;
  vec3 color = mix(center, nearby, 0.5 * uDedither);
  // Phase is measured in game rows, not output pixels: each of the 200 rows
  // gets the same shading whether the canvas is 1:1 with the screen or not.
  float separation = abs(fract(vUv.y * uRows) - 0.5);
  float shade = smoothstep(0.23 - 0.12 * uSoften, 0.5, separation);
  outColor = vec4(color * (1.0 - 0.65 * uOpacity * shade), 1.0);
}`;

export const UNIFORMS = ["uTexel", "uRows", "uActive", "uOpacity", "uDedither", "uSoften"];
