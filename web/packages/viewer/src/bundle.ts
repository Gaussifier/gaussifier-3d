/** Single-file bundle entry: everything in the package plus the custom element, registered on import. */
export * from "./index.js";
export { GaussifierViewerElement, defineGaussifierViewer, loadSceneFromUrl, decodeScene } from "./element.js";
import { defineGaussifierViewer } from "./element.js";
defineGaussifierViewer();
