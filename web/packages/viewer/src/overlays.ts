/** Overlay settings shared by the viewer and its backends; the marks themselves are drawn by the backends' shaders. */

export interface OverlayState {
  image: boolean;
  diff: boolean;
  density: boolean;
  /** Voronoi cell borders of the current state's centers. */
  voronoi: boolean;
  centers: boolean;
  /** Antialiased rings at `ellipseSigma` standard deviations, drawn in the fragment shader. */
  ellipses: boolean;
  /** Multiplier applied to |render - image| in diff mode. */
  diffGain: number;
  /** Mahalanobis radius of the ellipse rings; 1 shows the shape, 3.33 the alpha cutoff. */
  ellipseSigma: number;
  /** Ring core width in display pixels. */
  ellipseWidth: number;
}

export const DEFAULT_OVERLAYS: OverlayState = { image: false, diff: false, density: false, voronoi: false, centers: false, ellipses: false, diffGain: 4, ellipseSigma: 1, ellipseWidth: 1.25 };
