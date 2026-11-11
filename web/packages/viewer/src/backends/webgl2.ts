import type { RenderBackend, SceneState, View } from "../types.js";
import { CUTOFF_RADIUS, INSTANCE_FLOATS, buildInstances } from "../math.js";
import type { OverlayState } from "../overlays.js";
import { BackendBase, CELL_TINT_GLSL, COLORMAP_GLSL, DOT_CORE, DOT_CORE_ALPHA, DOT_HALO_ALPHA, INSTANCE_ATTRIBUTES, RING_CORE, RING_CORE_ALPHA, RING_FADE_GLSL, RING_HALO_ALPHA, SPLAT_CONTRIBUTION_GLSL, VORONOI_TINT_MIX, centerMarker, maxOf, planarToRgba, type BackendExtras } from "./common.js";

const SPLAT_VS = `#version 300 es
precision highp float;
uniform vec4 uView; uniform vec2 uCanvas;
layout(location=0) in vec2 aCenter; layout(location=1) in vec3 aConic; layout(location=2) in vec3 aColor;
layout(location=3) in vec2 aExt; layout(location=4) in float aValid;
flat out vec2 vCenter; flat out vec3 vConic; flat out vec3 vColor;
const vec2 corners[6] = vec2[6](vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0), vec2(-1.0,1.0), vec2(1.0,-1.0), vec2(1.0,1.0));
void main() {
  vec2 c = corners[gl_VertexID];
  vec2 dc = vec2(uView.x * (aCenter.x + 0.5) + uView.y, uView.x * (aCenter.y + 0.5) + uView.z);
  vec2 hv = aExt * uView.x + vec2(1.0);
  vec2 p = dc + c * hv;
  if (aValid < 0.5) p = vec2(-10.0);
  gl_Position = vec4(p.x / uCanvas.x * 2.0 - 1.0, 1.0 - p.y / uCanvas.y * 2.0, 0.0, 1.0);
  vCenter = aCenter; vConic = aConic; vColor = aColor;
}`;

const SPLAT_FS = `#version 300 es
precision highp float;
uniform vec4 uView; uniform vec2 uCanvas;
flat in vec2 vCenter; flat in vec3 vConic; flat in vec3 vColor;
out vec4 o;
${SPLAT_CONTRIBUTION_GLSL}
void main() {
  float X = gl_FragCoord.x; float Y = uCanvas.y - gl_FragCoord.y;
  float sx = (X - uView.y) / uView.x - 0.5; float sy = (Y - uView.z) / uView.x - 0.5;
  float a = splat_alpha(vCenter.x - sx, vCenter.y - sy, vConic);
  if (a <= 0.0) discard;
  o = vec4(vColor * a, 1.0);
}`;

const FULLSCREEN_VS = `#version 300 es
precision highp float;
const vec2 pts[3] = vec2[3](vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
void main() { gl_Position = vec4(pts[gl_VertexID], 0.0, 1.0); }`;

const RESOLVE_FS = `#version 300 es
precision highp float;
uniform vec4 uView; uniform vec2 uCanvas; uniform vec2 uSrc; uniform float uDensityMax;
uniform int uFlags; uniform int uHasImage; uniform int uHasDensity; uniform int uHasVoronoi; uniform float uBorderMix;
uniform sampler2D uAccum; uniform sampler2D uImage; uniform sampler2D uDensity; uniform highp usampler2D uOwners;
out vec4 o;
${COLORMAP_GLSL}
${CELL_TINT_GLSL}
void main() {
  ivec2 px = ivec2(gl_FragCoord.xy);
  vec3 r = clamp(texelFetch(uAccum, px, 0).rgb, 0.0, 1.0);
  r = floor(r * 255.0 + 0.5) / 255.0;
  vec2 dpos = vec2(gl_FragCoord.x, uCanvas.y - gl_FragCoord.y);
  vec2 s = (dpos - vec2(uView.y, uView.z)) / uView.x - 0.5;
  ivec2 si = ivec2(floor(s + 0.5));
  bool inside = si.x >= 0 && si.y >= 0 && si.x < int(uSrc.x) && si.y < int(uSrc.y);
  vec3 outc = r;
  if (!inside) {
    outc = vec3(0.12, 0.12, 0.13);
  } else {
    vec3 im = vec3(0.0);
    if (uHasImage == 1) im = texelFetch(uImage, si, 0).rgb;
    if ((uFlags & 1) != 0 && uHasImage == 1) outc = im;
    if ((uFlags & 2) != 0 && uHasImage == 1) outc = clamp(abs(r - im) * uView.w, 0.0, 1.0);
    if ((uFlags & 4) != 0 && uHasDensity == 1) { float d = texelFetch(uDensity, si, 0).r / max(uDensityMax, 1e-6); outc = colormap(clamp(d, 0.0, 1.0)); }
    if ((uFlags & 8) != 0 && uHasVoronoi == 1) {
      // Every cell is tinted by its owner; a border is one display pixel wide, found by comparing
      // with the source pixels under the next display pixel right and down.
      uint o = texelFetch(uOwners, si, 0).r;
      outc = mix(outc, cell_tint(o), ${VORONOI_TINT_MIX});
      ivec2 sr = ivec2(floor(s + vec2(1.0 / uView.x, 0.0) + 0.5));
      ivec2 sd = ivec2(floor(s + vec2(0.0, 1.0 / uView.x) + 0.5));
      bool border = (sr.x < int(uSrc.x) && texelFetch(uOwners, sr, 0).r != o) || (sd.y < int(uSrc.y) && texelFetch(uOwners, sd, 0).r != o);
      if (border) outc = mix(outc, vec3(0.04, 0.04, 0.05), uBorderMix);
    }
  }
  o = vec4(outc, 1.0);
}`;

/** Center discs from the splat instance buffer; see DOT_WGSL in the WebGPU backend. Premultiplied output. */
const DOT_VS = `#version 300 es
precision highp float;
uniform vec4 uDot; uniform vec2 uCanvas; uniform float uAlpha;
layout(location=0) in vec2 aCenter; layout(location=1) in vec3 aConic; layout(location=2) in vec3 aColor;
layout(location=3) in vec2 aExt; layout(location=4) in float aValid;
flat out vec2 vDc;
const vec2 corners[6] = vec2[6](vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0), vec2(-1.0,1.0), vec2(1.0,-1.0), vec2(1.0,1.0));
void main() {
  vec2 dc = vec2(uDot.x * (aCenter.x + 0.5) + uDot.y, uDot.x * (aCenter.y + 0.5) + uDot.z);
  vec2 p = dc + corners[gl_VertexID] * (uDot.w + 2.0);
  if (aValid < 0.5) p = vec2(-10.0);
  gl_Position = vec4(p.x / uCanvas.x * 2.0 - 1.0, 1.0 - p.y / uCanvas.y * 2.0, 0.0, 1.0);
  vDc = dc;
}`;

const DOT_FS = `#version 300 es
precision highp float;
uniform vec4 uDot; uniform vec2 uCanvas; uniform float uAlpha;
flat in vec2 vDc;
out vec4 o;
void main() {
  vec2 dpos = vec2(gl_FragCoord.x, uCanvas.y - gl_FragCoord.y);
  float d = length(dpos - vDc);
  float core = 1.0 - smoothstep(uDot.w - 0.5, uDot.w + 0.5, d);
  float halo = 1.0 - smoothstep(uDot.w + 0.5, uDot.w + 1.5, d);
  if (halo <= 0.002) discard;
  float alpha = max(core * ${DOT_CORE_ALPHA}, halo * ${DOT_HALO_ALPHA}) * uAlpha;
  o = vec4(vec3(${DOT_CORE[0]}, ${DOT_CORE[1]}, ${DOT_CORE[2]}) * core * ${DOT_CORE_ALPHA} * uAlpha, alpha);
}`;

/** Ellipse rings from the splat instance buffer; see RING_WGSL in the WebGPU backend. Premultiplied output. */
const RING_VS = `#version 300 es
precision highp float;
uniform vec4 uRing; uniform vec2 uCanvas; uniform float uWidth;
layout(location=0) in vec2 aCenter; layout(location=1) in vec3 aConic; layout(location=2) in vec3 aColor;
layout(location=3) in vec2 aExt; layout(location=4) in float aValid;
flat out vec2 vCenter; flat out vec3 vConic; flat out float vFade;
const vec2 corners[6] = vec2[6](vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0), vec2(-1.0,1.0), vec2(1.0,-1.0), vec2(1.0,1.0));
void main() {
  vec2 c = corners[gl_VertexID];
  vec2 dc = vec2(uRing.x * (aCenter.x + 0.5) + uRing.y, uRing.x * (aCenter.y + 0.5) + uRing.z);
  vec2 radius = aExt * (uRing.w / ${CUTOFF_RADIUS}) * uRing.x;
  vec2 p = dc + c * (radius + vec2(uWidth + 2.5));
  if (aValid < 0.5) p = vec2(-10.0);
  gl_Position = vec4(p.x / uCanvas.x * 2.0 - 1.0, 1.0 - p.y / uCanvas.y * 2.0, 0.0, 1.0);
  vCenter = aCenter; vConic = aConic;
  float rpx = max(radius.x, radius.y);
  vFade = ${RING_FADE_GLSL};
}`;

const RING_FS = `#version 300 es
precision highp float;
uniform vec4 uRing; uniform vec2 uCanvas; uniform float uWidth;
flat in vec2 vCenter; flat in vec3 vConic; flat in float vFade;
out vec4 o;
void main() {
  float X = gl_FragCoord.x; float Y = uCanvas.y - gl_FragCoord.y;
  float sx = (X - uRing.y) / uRing.x - 0.5; float sy = (Y - uRing.z) / uRing.x - 0.5;
  float dx = vCenter.x - sx; float dy = vCenter.y - sy;
  float d = sqrt(max(vConic.x * dx * dx + 2.0 * vConic.y * dx * dy + vConic.z * dy * dy, 0.0));
  float px = abs(d - uRing.w) / max(fwidth(d), 1e-6);
  float core = 1.0 - smoothstep(uWidth * 0.5 - 0.5, uWidth * 0.5 + 0.5, px);
  float halo = 1.0 - smoothstep(uWidth * 0.5 + 0.5, uWidth * 0.5 + 1.5, px);
  if (halo <= 0.002) discard;
  float alpha = max(core * ${RING_CORE_ALPHA}, halo * ${RING_HALO_ALPHA}) * vFade;
  o = vec4(vec3(${RING_CORE[0]}, ${RING_CORE[1]}, ${RING_CORE[2]}) * core * ${RING_CORE_ALPHA} * vFade, alpha);
}`;


function compile(gl: WebGL2RenderingContext, vs: string, fs: string, label: string): WebGLProgram {
  const make = (type: number, src: string) => {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(`${label}: ${gl.getShaderInfoLog(s)}`);
    return s;
  };
  const p = gl.createProgram()!;
  gl.attachShader(p, make(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, make(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(`${label}: ${gl.getProgramInfoLog(p)}`);
  return p;
}

export class WebGl2Backend extends BackendBase implements RenderBackend, BackendExtras {
  readonly kind = "webgl2" as const;
  readonly gl: WebGL2RenderingContext;
  /** "rgba32float" when EXT_float_blend allows 32-bit accumulation, else "rgba16float". */
  readonly accumFormat: "rgba32float" | "rgba16float";
  private readonly instanceBuf: WebGLBuffer;
  private readonly instanceVao: WebGLVertexArrayObject;
  private accumTex: WebGLTexture | null = null;
  private accumFbo: WebGLFramebuffer | null = null;
  private displayTex: WebGLTexture | null = null;
  private displayFbo: WebGLFramebuffer | null = null;
  private readonly imageTex: WebGLTexture;
  private readonly densityTex: WebGLTexture;
  private readonly ownerTex: WebGLTexture;
  private readonly splatProg: WebGLProgram;
  private readonly resolveProg: WebGLProgram;
  private readonly dotProg: WebGLProgram;
  private readonly ringProg: WebGLProgram;
  private readonly accumInternalFormat: number;
  private readonly accumType: number;

  constructor(readonly canvas: HTMLCanvasElement) {
    super();
    const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, premultipliedAlpha: false, preserveDrawingBuffer: false, depth: false, stencil: false });
    if (!gl) throw new Error("WebGL2 is not available");
    this.gl = gl;
    const cbf = gl.getExtension("EXT_color_buffer_float");
    const cbhf = gl.getExtension("EXT_color_buffer_half_float");
    if (!cbf && !cbhf) throw new Error("WebGL2 backend needs EXT_color_buffer_float or EXT_color_buffer_half_float for float accumulation");
    const floatBlend = gl.getExtension("EXT_float_blend");
    if (cbf && floatBlend) {
      this.accumInternalFormat = gl.RGBA32F;
      this.accumType = gl.FLOAT;
      this.accumFormat = "rgba32float";
    } else {
      this.accumInternalFormat = gl.RGBA16F;
      this.accumType = gl.HALF_FLOAT;
      this.accumFormat = "rgba16float";
    }
    this.splatProg = compile(gl, SPLAT_VS, SPLAT_FS, "viewer splat");
    this.resolveProg = compile(gl, FULLSCREEN_VS, RESOLVE_FS, "viewer resolve");
    this.dotProg = compile(gl, DOT_VS, DOT_FS, "viewer dots");
    this.ringProg = compile(gl, RING_VS, RING_FS, "viewer rings");

    this.instanceBuf = gl.createBuffer()!;
    this.instanceVao = gl.createVertexArray()!;
    gl.bindVertexArray(this.instanceVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuf);
    const stride = INSTANCE_FLOATS * 4;
    const attr = (loc: number, size: number, offset: number) => {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, offset);
      gl.vertexAttribDivisor(loc, 1);
    };
    for (const a of INSTANCE_ATTRIBUTES) attr(a.location, a.size, a.offset);
    gl.bindVertexArray(null);

    this.imageTex = gl.createTexture()!;
    this.densityTex = gl.createTexture()!;
    this.ownerTex = gl.createTexture()!;
    for (const t of [this.imageTex, this.densityTex, this.ownerTex]) {
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }
    gl.bindTexture(gl.TEXTURE_2D, this.imageTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 1, 1, 0, gl.RGBA, gl.FLOAT, new Float32Array(4));
    gl.bindTexture(gl.TEXTURE_2D, this.densityTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, 1, 1, 0, gl.RED, gl.FLOAT, new Float32Array(1));
    gl.bindTexture(gl.TEXTURE_2D, this.ownerTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32UI, 1, 1, 0, gl.RED_INTEGER, gl.UNSIGNED_INT, new Uint32Array(1));
    this.resize(canvas.width || 1, canvas.height || 1);
  }

  private makeTarget(internalFormat: number, format: number, type: number, w: number, h: number): { tex: WebGLTexture; fbo: WebGLFramebuffer } {
    const { gl } = this;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);
    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (status !== gl.FRAMEBUFFER_COMPLETE) throw new Error(`WebGL2 framebuffer incomplete: ${status}`);
    return { tex, fbo };
  }

  resize(width: number, height: number): void {
    width = Math.max(1, Math.floor(width));
    height = Math.max(1, Math.floor(height));
    if (this.canvas.width !== width) this.canvas.width = width;
    if (this.canvas.height !== height) this.canvas.height = height;
    if (width === this.width && height === this.height && this.accumFbo) return;
    const { gl } = this;
    this.width = width;
    this.height = height;
    if (this.accumTex) gl.deleteTexture(this.accumTex);
    if (this.accumFbo) gl.deleteFramebuffer(this.accumFbo);
    if (this.displayTex) gl.deleteTexture(this.displayTex);
    if (this.displayFbo) gl.deleteFramebuffer(this.displayFbo);
    ({ tex: this.accumTex, fbo: this.accumFbo } = this.makeTarget(this.accumInternalFormat, gl.RGBA, this.accumType, width, height));
    ({ tex: this.displayTex, fbo: this.displayFbo } = this.makeTarget(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, width, height));
  }

  setState(state: SceneState, sourceWidth: number, sourceHeight: number): void {
    const { gl } = this;
    this.srcW = sourceWidth;
    this.srcH = sourceHeight;
    const inst = buildInstances(state, sourceWidth, sourceHeight);
    this.instanceCount = state.xy.length / 2;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuf);
    gl.bufferData(gl.ARRAY_BUFFER, inst, gl.STATIC_DRAW);
  }

  setImage(image: Float32Array | null, width: number, height: number): void {
    const { gl } = this;
    gl.bindTexture(gl.TEXTURE_2D, this.imageTex);
    if (!image) { gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 1, 1, 0, gl.RGBA, gl.FLOAT, new Float32Array(4)); this.hasImage = false; return; }
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, planarToRgba(image, width, height));
    this.hasImage = true;
  }

  setDensity(density: Float32Array | null, width: number, height: number): void {
    const { gl } = this;
    gl.bindTexture(gl.TEXTURE_2D, this.densityTex);
    if (!density) { gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, 1, 1, 0, gl.RED, gl.FLOAT, new Float32Array(1)); this.hasDensity = false; return; }
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, density);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    this.densityMax = maxOf(density);
    this.hasDensity = true;
  }

  setOwners(owners: Uint32Array | null, width: number, height: number): void {
    const { gl } = this;
    gl.bindTexture(gl.TEXTURE_2D, this.ownerTex);
    if (!owners) { gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32UI, 1, 1, 0, gl.RED_INTEGER, gl.UNSIGNED_INT, new Uint32Array(1)); this.hasVoronoi = false; return; }
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32UI, width, height, 0, gl.RED_INTEGER, gl.UNSIGNED_INT, owners);
    this.hasVoronoi = true;
  }

  render(view: View): void {
    if (this.disposed || !this.accumFbo || !this.displayFbo) return;
    const { gl } = this;
    const w = this.width, h = this.height;
    gl.viewport(0, 0, w, h);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accumFbo);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (this.instanceCount > 0) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);
      gl.useProgram(this.splatProg);
      gl.uniform4f(gl.getUniformLocation(this.splatProg, "uView"), view.zoom, view.tx, view.ty, 0);
      gl.uniform2f(gl.getUniformLocation(this.splatProg, "uCanvas"), w, h);
      gl.bindVertexArray(this.instanceVao);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.instanceCount);
      gl.bindVertexArray(null);
      gl.disable(gl.BLEND);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.displayFbo);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.resolveProg);
    const u = (name: string) => gl.getUniformLocation(this.resolveProg, name);
    const r = this.resolveParams(view);
    gl.uniform4f(u("uView"), r.zoom, r.tx, r.ty, r.gain);
    gl.uniform2f(u("uCanvas"), r.canvasW, r.canvasH);
    gl.uniform2f(u("uSrc"), r.srcW, r.srcH);
    gl.uniform1f(u("uDensityMax"), r.densityMax);
    gl.uniform1i(u("uFlags"), r.flags);
    gl.uniform1i(u("uHasImage"), r.hasImage);
    gl.uniform1i(u("uHasDensity"), r.hasDensity);
    gl.uniform1i(u("uHasVoronoi"), r.hasVoronoi);
    gl.uniform1f(u("uBorderMix"), r.borderMix);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.accumTex); gl.uniform1i(u("uAccum"), 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.imageTex); gl.uniform1i(u("uImage"), 1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.densityTex); gl.uniform1i(u("uDensity"), 2);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, this.ownerTex); gl.uniform1i(u("uOwners"), 3);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const drawRings = this.overlays.ellipses && this.instanceCount > 0;
    const drawDots = this.overlays.centers && this.instanceCount > 0;
    if (drawRings || drawDots) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      if (drawRings) {
        gl.useProgram(this.ringProg);
        gl.uniform4f(gl.getUniformLocation(this.ringProg, "uRing"), view.zoom, view.tx, view.ty, this.overlays.ellipseSigma);
        gl.uniform2f(gl.getUniformLocation(this.ringProg, "uCanvas"), w, h);
        gl.uniform1f(gl.getUniformLocation(this.ringProg, "uWidth"), this.overlays.ellipseWidth);
        gl.bindVertexArray(this.instanceVao);
        gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.instanceCount);
      }
      if (drawDots) {
        const m = centerMarker(view.zoom, this.srcW, this.srcH, this.instanceCount);
        gl.useProgram(this.dotProg);
        gl.uniform4f(gl.getUniformLocation(this.dotProg, "uDot"), view.zoom, view.tx, view.ty, m.radius);
        gl.uniform2f(gl.getUniformLocation(this.dotProg, "uCanvas"), w, h);
        gl.uniform1f(gl.getUniformLocation(this.dotProg, "uAlpha"), m.alpha);
        gl.bindVertexArray(this.instanceVao);
        gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.instanceCount);
      }
      gl.bindVertexArray(null);
      gl.disable(gl.BLEND);
    }
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.displayFbo);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
    gl.blitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.COLOR_BUFFER_BIT, gl.NEAREST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  async readPixels(): Promise<Uint8ClampedArray> {
    const { gl } = this;
    const w = this.width, h = this.height;
    const bottomUp = new Uint8Array(w * h * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.displayFbo);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, bottomUp);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const out = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) out.set(bottomUp.subarray((h - 1 - y) * w * 4, (h - y) * w * 4), y * w * 4);
    return out;
  }

  dispose(): void {
    this.disposed = true;
    const { gl } = this;
    gl.deleteProgram(this.splatProg);
    gl.deleteProgram(this.resolveProg);
    gl.deleteProgram(this.dotProg);
    gl.deleteProgram(this.ringProg);
    gl.deleteBuffer(this.instanceBuf);
    gl.deleteVertexArray(this.instanceVao);
    if (this.accumTex) gl.deleteTexture(this.accumTex);
    if (this.displayTex) gl.deleteTexture(this.displayTex);
    if (this.accumFbo) gl.deleteFramebuffer(this.accumFbo);
    if (this.displayFbo) gl.deleteFramebuffer(this.displayFbo);
    gl.deleteTexture(this.imageTex);
    gl.deleteTexture(this.densityTex);
    gl.deleteTexture(this.ownerTex);
  }
}
