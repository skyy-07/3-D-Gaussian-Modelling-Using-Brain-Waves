import * as THREE from 'three';
import {
  LogLevel,
  OrbitControls,
  RenderMode,
  SceneRevealMode,
  Viewer,
  WebXRMode,
} from '@mkkellogg/gaussian-splats-3d';

/**
 * Thin, defensive wrapper around GaussianSplats3D.
 *
 * Design decisions that matter for WebXR:
 *  - The viewer owns the render loop and uses `renderer.setAnimationLoop`, which is
 *    required for immersive sessions. We do not start a second rAF loop.
 *  - We ask for the `local` reference space. Its origin is the headset at session
 *    start, so the virtual camera in the manifest can be mapped to `(0, 0, 0)` and
 *    the room appears at exactly the viewpoint it was lifted from.
 *  - The splat scene is translated by `-camera_position` instead of moving the
 *    camera. That single offset makes desktop and headset agree, and keeps the
 *    environment shell centred on the viewer.
 *  - Scene swaps remove + add rather than loading four volumes at once: Quest memory
 *    and the CPU splat sort are the scarce resources, not disk cache.
 */
export class SplatStage {
  constructor({
    container,
    xrOverlay,
    onReady,
    onProgress,
    onError,
    onXRChange,
    onTrigger,
  }) {
    this.container = container;
    this.xrOverlay = xrOverlay;
    this.onReady = onReady;
    this.onProgress = onProgress;
    this.onError = onError;
    this.onXRChange = onXRChange;
    this.onTrigger = onTrigger;

    this.viewer = null;
    this.controls = null;
    this.offset = [0, 0, -4];
    this.loadedUrl = null;
    this.pending = null;
    this.runner = null;
    this.xr = false;
    this.savedView = null;
  }

  init() {
    const sessionInit = {
      optionalFeatures: ['dom-overlay', 'hand-tracking', 'local-floor', 'bounded-floor'],
    };
    if (this.xrOverlay) sessionInit.domOverlay = { root: this.xrOverlay };

    this.viewer = new Viewer({
      rootElement: this.container,
      cameraUp: [0, 1, 0],
      initialCameraPosition: [0, 0, 0],
      initialCameraLookAt: [0, 0, -1],
      webXRMode: WebXRMode.VR,
      webXRSessionInit: sessionInit,
      // We manage desktop orbit ourselves because the built-in controls are disabled
      // whenever webXRMode is set.
      useBuiltInControls: false,
      selfDrivenMode: true,
      sharedMemoryForWorkers: Boolean(window.crossOriginIsolated),
      gpuAcceleratedSort: false,
      integerBasedSort: true,
      splatSortDistanceMapPrecision: 16,
      sceneRevealMode: SceneRevealMode.Instant,
      renderMode: RenderMode.Always,
      sphericalHarmonicsDegree: 0,
      halfPrecisionCovariancesOnGPU: true,
      ignoreDevicePixelRatio: false,
      logLevel: LogLevel.None,
    });

    const renderer = this.viewer.renderer;
    renderer.setClearColor(new THREE.Color(0x08090c), 1);
    // `local` (not `local-floor`) keeps the manifest camera at eye height instead of
    // dropping the room to the user's feet.
    renderer.xr.setReferenceSpaceType('local');

    const camera = this.viewer.camera;
    camera.fov = 60; // match the keyframe / CameraModel hfov
    camera.updateProjectionMatrix();

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, -4);
    controls.enableDamping = true;
    controls.dampingFactor = 0.075;
    controls.rotateSpeed = 0.55;
    controls.zoomSpeed = 0.7;
    controls.panSpeed = 0.55;
    controls.minDistance = 0.8;
    controls.maxDistance = 10;
    controls.minPolarAngle = Math.PI * 0.16;
    controls.maxPolarAngle = Math.PI * 0.64;
    controls.update();
    this.controls = controls;
    this.viewer.controls = controls;

    const xr = renderer.xr;
    xr.addEventListener('sessionstart', () => {
      this.xr = true;
      this.savedView = {
        position: camera.position.clone(),
        quaternion: camera.quaternion.clone(),
        target: controls.target.clone(),
      };
      controls.enabled = false;
      this.onXRChange?.(true);
    });
    xr.addEventListener('sessionend', () => {
      this.xr = false;
      controls.enabled = true;
      if (this.savedView) {
        camera.position.copy(this.savedView.position);
        camera.quaternion.copy(this.savedView.quaternion);
        controls.target.copy(this.savedView.target);
        controls.update();
      }
      this.onXRChange?.(false);
    });

    // Trigger / squeeze let the wearer scrub the memory without a DOM overlay.
    for (let i = 0; i < 2; i += 1) {
      const controller = xr.getController(i);
      controller.addEventListener('selectstart', () => this.onTrigger?.('next'));
      controller.addEventListener('squeezestart', () => this.onTrigger?.('prev'));
      this.viewer.threeScene.add(controller);
    }

    this.viewer.start();
    this._restyleVRButton();
  }

  _restyleVRButton() {
    // The library appends its own button during construction. Keep the behaviour,
    // replace the look. `isSessionSupported` is async, so retry briefly.
    let tries = 0;
    const find = () => {
      const button = document.getElementById('VRButton') || this.container.querySelector('button, a');
      if (button) {
        button.classList.add('mnemo-vr');
        // Escape the canvas stacking context so the pill sits with the HUD.
        const host = this.container.parentElement || document.body;
        if (button.parentElement !== host) host.appendChild(button);
        if (button.tagName === 'BUTTON') button.setAttribute('aria-label', 'Enter virtual reality');

        // Replace the library's developer-facing labels with demo-facing copy.
        const relabel = () => {
          const text = (button.textContent || '').trim();
          if (/not supported|not allowed/i.test(text)) {
            button.textContent = 'WebXR · headset required';
            button.classList.add('is-unavailable');
            button.setAttribute('aria-disabled', 'true');
          } else if (/enter vr/i.test(text)) {
            button.textContent = 'Enter VR';
            button.classList.remove('is-unavailable');
            button.removeAttribute('aria-disabled');
          }
        };
        new MutationObserver(relabel).observe(button, {
          childList: true,
          characterData: true,
          subtree: true,
        });
        relabel();
        return true;
      }
      if (tries < 20) {
        tries += 1;
        setTimeout(find, 100);
      }
      return false;
    };
    find();
  }

  /** Map the manifest camera to the viewer origin by moving the scene, not the eye. */
  setFraming(cameraPosition) {
    const p = cameraPosition || [0, 0, 4];
    this.offset = [-Number(p[0] || 0), -Number(p[1] || 0), -Number(p[2] || 0)];
    this.resetView();
  }

  resetView() {
    if (!this.viewer) return;
    this.viewer.camera.position.set(0, 0, 0);
    this.controls.target.set(this.offset[0], this.offset[1], this.offset[2]);
    this.controls.update();
  }

  /**
   * Load a frame. Rapid calls are coalesced: the latest request wins, and the
   * previous download is allowed to finish rather than aborting the renderer.
   * Every caller during a burst receives the same promise, which resolves when the
   * queue is empty — exactly what the busy indicator needs.
   */
  show(url) {
    if (url) this.pending = url;
    if (!this.pending) return Promise.resolve();
    if (url === this.loadedUrl && !this.runner) {
      this.pending = null;
      return Promise.resolve();
    }
    if (this.runner) return this.runner;

    this.runner = this._drain().finally(() => {
      this.runner = null;
      // A request that arrived in the microtask gap gets its own drain.
      if (this.pending) this.show(this.pending);
    });
    return this.runner;
  }

  async _drain() {
    while (this.pending) {
      const target = this.pending;
      this.pending = null;
      if (target === this.loadedUrl) continue;

      try {
        if (this.viewer.splatMesh.getSplatCount() > 0) {
          await this.viewer.removeSplatScene(0, false);
          this.loadedUrl = null;
        }

        await this.viewer.addSplatScene(target, {
          splatAlphaRemovalThreshold: 4,
          showLoadingUI: false,
          progressiveLoad: false,
          position: this.offset,
          onProgress: (pct, label, status) => this.onProgress?.(pct, label, status),
        });
        this.loadedUrl = target;
        this.onReady?.(target);
      } catch (error) {
        this.onError?.(error);
      }
    }
  }

  get splatCount() {
    return this.viewer?.splatMesh?.getSplatCount?.() ?? 0;
  }
}
