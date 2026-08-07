/**
 * Shaders, inlined as strings.
 *
 * Your original component read these with document.getElementById() inside
 * useEffect, but the <script> tags live in that same component's return — so on
 * first mount they don't exist yet and the lookup returns null. Inlining is the
 * fix, and it also means the shaders can't be clobbered by anything else on the
 * page.
 *
 * The noise is exactly your code. The only addition is at the very bottom of
 * the fragment shader: a tint mix, so the glob can be a specific colour on
 * demand. Without it the colours are emergent Perlin noise and there's no
 * reliable way to say "be blue now" — which your idle/listening/thinking states
 * need. tintMix = 0 gives back your original multicolour look untouched, and
 * that's exactly what the speaking state uses.
 */

const PERLIN = /* glsl */ `
  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 permute(vec4 x) { return mod289(((x * 34.0) + 1.0) * x); }
  vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
  vec3 fade(vec3 t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }

  float cnoise(vec3 P) {
    vec3 Pi0 = floor(P);
    vec3 Pi1 = Pi0 + vec3(1.0);
    Pi0 = mod289(Pi0);
    Pi1 = mod289(Pi1);
    vec3 Pf0 = fract(P);
    vec3 Pf1 = Pf0 - vec3(1.0);
    vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
    vec4 iy = vec4(Pi0.yy, Pi1.yy);
    vec4 iz0 = Pi0.zzzz;
    vec4 iz1 = Pi1.zzzz;
    vec4 ixy = permute(permute(ix) + iy);
    vec4 ixy0 = permute(ixy + iz0);
    vec4 ixy1 = permute(ixy + iz1);
    vec4 gx0 = ixy0 * (1.0 / 7.0);
    vec4 gy0 = fract(floor(gx0) * (1.0 / 7.0)) - 0.5;
    gx0 = fract(gx0);
    vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
    vec4 sz0 = step(gz0, vec4(0.0));
    gx0 -= sz0 * (step(0.0, gx0) - 0.5);
    gy0 -= sz0 * (step(0.0, gy0) - 0.5);
    vec4 gx1 = ixy1 * (1.0 / 7.0);
    vec4 gy1 = fract(floor(gx1) * (1.0 / 7.0)) - 0.5;
    gx1 = fract(gx1);
    vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
    vec4 sz1 = step(gz1, vec4(0.0));
    gx1 -= sz1 * (step(0.0, gx1) - 0.5);
    gy1 -= sz1 * (step(0.0, gy1) - 0.5);
    vec3 g000 = vec3(gx0.x, gy0.x, gz0.x);
    vec3 g100 = vec3(gx0.y, gy0.y, gz0.y);
    vec3 g010 = vec3(gx0.z, gy0.z, gz0.z);
    vec3 g110 = vec3(gx0.w, gy0.w, gz0.w);
    vec3 g001 = vec3(gx1.x, gy1.x, gz1.x);
    vec3 g101 = vec3(gx1.y, gy1.y, gz1.y);
    vec3 g011 = vec3(gx1.z, gy1.z, gz1.z);
    vec3 g111 = vec3(gx1.w, gy1.w, gz1.w);
    vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
    g000 *= norm0.x; g010 *= norm0.y; g100 *= norm0.z; g110 *= norm0.w;
    vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
    g001 *= norm1.x; g011 *= norm1.y; g101 *= norm1.z; g111 *= norm1.w;
    float n000 = dot(g000, Pf0);
    float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
    float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
    float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
    float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
    float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
    float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
    float n111 = dot(g111, Pf1);
    vec3 fade_xyz = fade(Pf0);
    vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
    vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
    float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
    return SCALE * n_xyz;
  }
`;

export const VERTEX_SHADER = /* glsl */ `
  varying vec3 vNormal;
  uniform float time;
  uniform float morph;
  uniform float psize;

  #define SCALE 1.2
  ${PERLIN}

  void main() {
    float f = morph * cnoise(normal + time);
    vNormal = normalize(normal);
    vec4 pos = vec4(position + f * normal, 1.0);
    gl_Position = projectionMatrix * modelViewMatrix * pos;
    gl_PointSize = psize;
  }
`;

export const FRAGMENT_SHADER = /* glsl */ `
  varying vec3 vNormal;
  uniform float time;
  uniform float RGBr;
  uniform float RGBg;
  uniform float RGBb;
  uniform float RGBn;
  uniform float RGBm;
  uniform float dnoise;

  // Added: lets a state drive the colour deterministically.
  uniform vec3  tint;
  uniform float tintMix;
  // Added: the raw noise spends a lot of its range negative, which clamps to
  // black. Gain lifts the lit regions so the untinted speaking state reads as
  // vivid instead of mostly dark. 1.0 is your original.
  uniform float gain;

  vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
  }

  vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
  }

  #define SCALE 2.2
  ${PERLIN}

  void main() {
    float r = cnoise(RGBr * (vNormal + time));
    float g = cnoise(RGBg * (vNormal + time));
    float b = cnoise(RGBb * (vNormal + time));
    float n = 50.0 * cnoise(RGBn * vNormal) * cnoise(RGBm * (vNormal + time));
    n -= 0.10 * cnoise(dnoise * vNormal);

    vec3 raw = vec3(r + n, g + n, b + n);
    vec3 color = clamp(raw, 0.0, 4.0);

    // Pull the hue toward the tint instead of painting the tint over the top.
    //
    // A straight mix() toward a flat colour flattens everything — every pixel
    // converges on the same green and the noise stops being visible. Rotating
    // hue keeps each pixel's own character, so the surface stays multicoloured
    // while the palette as a whole leans green, or blue, or amber. tintMix is
    // how far around the wheel each pixel travels, not how much of it gets
    // overwritten.
    if (tintMix > 0.001) {
      // Hue and brightness come from the *unclamped* field. The clamped one
      // is black across much of the sphere, and black has no hue — sampling
      // it would throw away most of the variation before we started.
      vec3 centred = clamp(raw * 0.5 + 0.5, 0.0, 1.0);
      vec3 hsv = rgb2hsv(centred);
      vec3 target = rgb2hsv(tint);

      // Shortest way round the wheel, so red doesn't detour through cyan.
      float dh = target.x - hsv.x;
      dh -= floor(dh + 0.5);
      hsv.x = fract(hsv.x + dh * tintMix);

      // Saturation up, so near-grey regions take the colour too.
      hsv.y = mix(hsv.y, clamp(hsv.y + 0.5, 0.6, 1.0), tintMix);

      // Brightness from the raw field, mapped into a range that's lit
      // everywhere but still varies across the surface.
      float v = clamp(dot(raw, vec3(0.3333)) * 0.55 + 0.66, 0.18, 1.15);
      hsv.z = mix(hsv.z, v, tintMix);

      color = hsv2rgb(hsv);
    }

    gl_FragColor = vec4(color * gain, 1.0);
  }
`;
