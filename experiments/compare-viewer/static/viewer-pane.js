/**
 * Full-featured 3D scene pane (background cloud, object pts, boxes, labels, crops).
 * Used by the A/B compare viewer — one instance per side (a / b).
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

const FLIP_Y = true;
const fy = y => (FLIP_Y ? -y : y);

export class ViewerPane {
  /**
   * @param {string} key - 'a' | 'b'
   * @param {HTMLElement} container - pane root (position:relative)
   * @param {string} urlPrefix - e.g. 'a' or 'b' for fetch URLs
   * @param {{ onCameraChange?: () => void, onSelect?: (idx: number) => void }} hooks
   */
  constructor(key, container, urlPrefix, hooks = {}) {
    this.key = key;
    this.urlPrefix = urlPrefix;
    this.hooks = hooks;
    this.objects = [];
    this.objMeshes = [];
    this.bgCloud = null;
    this.metadata = null;
    this.selectedIdx = -1;
    this.isolatedIdx = -1;
    this.lastClickTime = 0;
    this.layers = { bg: true, 'obj-pts': true, boxes: true, labels: true, billboards: true };

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.01, 2000);
    this.camera.position.set(0, 30, 60);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    this.renderer.setClearColor(0x0d0d0d);
    container.appendChild(this.renderer.domElement);

    this.css2dLayer = document.createElement('div');
    this.css2dLayer.style.cssText = 'position:absolute;inset:0;pointer-events:none;overflow:hidden';
    container.appendChild(this.css2dLayer);
    this.css2d = new CSS2DRenderer({ element: this.css2dLayer });

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.addEventListener('change', () => this.hooks.onCameraChange?.());

    this.raycaster = new THREE.Raycaster();
    this.raycaster.params.Points.threshold = 0.06;
    this.renderer.domElement.addEventListener('click', e => this._onClick(e));
  }

  resize(w, h) {
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.css2d.setSize(w, h);
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.css2d.render(this.scene, this.camera);
  }

  setLayers(layers) {
    this.layers = { ...this.layers, ...layers };
    this._applyLayerVisibility();
  }

  syncFrom(other) {
    this.camera.position.copy(other.camera.position);
    this.camera.quaternion.copy(other.camera.quaternion);
    this.controls.target.copy(other.controls.target);
    this.controls.update();
  }

  async load(cacheBust = false) {
    const q = cacheBust ? `?v=${Date.now()}` : '';
    const metaResp = await fetch(`${this.urlPrefix}/metadata.json${q}`);
    this.metadata = await metaResp.json();
    if (this.metadata.has_bg_cloud) {
      await this._loadBgCloud(q);
    }
    const resp = await fetch(`${this.urlPrefix}/objects.json${q}`);
    this.objects = await resp.json();
    for (const obj of this.objects) {
      obj.mean[1] = fy(obj.mean[1]);
      if (obj.bbox_min) obj.bbox_min[1] = fy(obj.bbox_min[1]);
      if (obj.bbox_max) obj.bbox_max[1] = fy(obj.bbox_max[1]);
    }
    await this._buildObjects();
    this._applyLayerVisibility();
    this.centreCamera();
    return this.metadata;
  }

  async reload() {
    this._clearScene();
    return this.load(true);
  }

  _clearScene() {
    const disposeObj = obj => {
      if (!obj) return;
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
        else obj.material.dispose();
      }
    };
    for (const m of this.objMeshes) {
      if (m.pts) this.scene.remove(m.pts);
      if (m.box) this.scene.remove(m.box);
      if (m.label) this.scene.remove(m.label);
      if (m.billboard) this.scene.remove(m.billboard);
      disposeObj(m.pts);
      disposeObj(m.box);
      if (m.billboard) {
        m.billboard.material?.map?.dispose();
        disposeObj(m.billboard);
      }
      if (m.label?.element?.parentNode) m.label.element.remove();
    }
    if (this.bgCloud) {
      disposeObj(this.bgCloud);
      this.scene.remove(this.bgCloud);
    }
    this.objMeshes = [];
    this.objects = [];
    this.bgCloud = null;
    this.selectedIdx = -1;
    this.isolatedIdx = -1;
  }

  selectObject(ki, flyTo = false, silent = false) {
    if (this.selectedIdx >= 0 && this.selectedIdx < this.objMeshes.length) {
      const prev = this.objMeshes[this.selectedIdx];
      if (prev.pts) prev.pts.material.size = 0.04;
      if (prev.box) prev.box.material.opacity = 0.4;
    }
    this.selectedIdx = ki;
    if (ki >= 0 && ki < this.objMeshes.length) {
      const cur = this.objMeshes[ki];
      if (cur.pts) cur.pts.material.size = 0.1;
      if (cur.box) cur.box.material.opacity = 1.0;
    }
    if (flyTo && ki >= 0) {
      const obj = this.objects[ki];
      const target = new THREE.Vector3(...obj.mean);
      this.controls.target.copy(target);
      this.camera.position.set(obj.mean[0] + 6, obj.mean[1] + 6, obj.mean[2] + 6);
      this.controls.update();
    }
    if (!silent) this.hooks.onSelect?.(ki);
  }

  flyToIndex(ki) {
    this.selectObject(ki, true);
  }

  centreCamera() {
    if (!this.objects.length) return;
    const pts = this.objects.map(o => o.mean);
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]), zs = pts.map(p => p[2]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const cz = (Math.min(...zs) + Math.max(...zs)) / 2;
    const span = Math.max(
      Math.max(...xs) - Math.min(...xs),
      Math.max(...zs) - Math.min(...zs),
      5,
    );
    this.controls.target.set(cx, cy, cz);
    this.camera.position.set(cx, cy + span * 0.5, cz + span * 0.7);
    this.controls.update();
  }

  getSelectedObject() {
    return this.selectedIdx >= 0 ? this.objects[this.selectedIdx] : null;
  }

  _onClick(e) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(mouse, this.camera);
    const testObjs = this.objMeshes.filter(m => m.pts && m.pts.visible).map(m => m.pts);
    const hits = this.raycaster.intersectObjects(testObjs, false);
    const now = Date.now();
    if (hits.length) {
      const ki = hits[0].object.userData.objectIndex;
      const dbl = now - this.lastClickTime < 300 && ki === this.selectedIdx;
      this.lastClickTime = now;
      if (dbl) this._toggleIsolation(ki);
      else this.selectObject(ki, false);
    } else if (now - this.lastClickTime > 300) {
      this.selectObject(-1, false);
    }
    this.lastClickTime = now;
  }

  _toggleIsolation(ki) {
    if (this.isolatedIdx === ki) {
      this.isolatedIdx = -1;
      this._applyLayerVisibility();
      this.hooks.onIsolate?.(-1);
    } else {
      this.isolatedIdx = ki;
      this.hooks.onIsolate?.(ki);
      for (let i = 0; i < this.objMeshes.length; i++) {
        const vis = i === ki;
        const m = this.objMeshes[i];
        if (m.pts) m.pts.visible = vis && this.layers['obj-pts'];
        if (m.box) m.box.visible = vis && this.layers.boxes;
        if (m.label) m.label.visible = vis && this.layers.labels;
        if (m.billboard) m.billboard.visible = vis && this.layers.billboards;
      }
      if (this.bgCloud) this.bgCloud.visible = false;
    }
  }

  _applyLayerVisibility() {
    if (this.bgCloud) this.bgCloud.visible = this.layers.bg;
    for (let i = 0; i < this.objMeshes.length; i++) {
      const show = this.isolatedIdx < 0 || i === this.isolatedIdx;
      const m = this.objMeshes[i];
      if (m.pts) m.pts.visible = show && this.layers['obj-pts'];
      if (m.box) m.box.visible = show && this.layers.boxes;
      if (m.label) m.label.visible = show && this.layers.labels;
      if (m.billboard) m.billboard.visible = show && this.layers.billboards;
    }
  }

  async _loadBgCloud(q = '') {
    const resp = await fetch(`${this.urlPrefix}/bg_cloud.bin${q}`);
    if (!resp.ok) return;
    const buf = await resp.arrayBuffer();
    const count = new Uint32Array(buf, 8, 1)[0];
    const data = new DataView(buf, 16);
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);
    const STRIDE = 16;
    for (let i = 0; i < count; i++) {
      const off = i * STRIDE;
      pos[i * 3] = data.getFloat32(off, true);
      pos[i * 3 + 1] = fy(data.getFloat32(off + 4, true));
      pos[i * 3 + 2] = data.getFloat32(off + 8, true);
      col[i * 3] = data.getUint8(off + 12) / 255;
      col[i * 3 + 1] = data.getUint8(off + 13) / 255;
      col[i * 3 + 2] = data.getUint8(off + 14) / 255;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    const mat = new THREE.PointsMaterial({ size: 0.025, vertexColors: true, sizeAttenuation: true });
    this.bgCloud = new THREE.Points(geo, mat);
    this.scene.add(this.bgCloud);
  }

  async _buildObjects() {
    const billboardPromises = [];
    for (let ki = 0; ki < this.objects.length; ki++) {
      const obj = this.objects[ki];
      const mesh = { id: obj.id, pts: null, box: null, label: null, billboard: null };

      if (obj.pts_b64) {
        const xyz = this._b64ToXYZ(obj.pts_b64);
        const pts = this._makeObjCloud(xyz, obj.color);
        pts.userData.objectIndex = ki;
        this.scene.add(pts);
        mesh.pts = pts;
      }
      if (obj.bbox_min && obj.bbox_max) {
        const box = this._makeBox(obj.bbox_min, obj.bbox_max, obj.color);
        box.userData.objectIndex = ki;
        this.scene.add(box);
        mesh.box = box;
      }
      const labelY = obj.bbox_max
        ? Math.max(obj.bbox_min[1], obj.bbox_max[1]) + 0.08
        : obj.mean[1] + 0.3;
      const label = this._makeLabel(obj.label, [obj.mean[0], labelY, obj.mean[2]], obj.color);
      this.scene.add(label);
      mesh.label = label;
      this.objMeshes.push(mesh);

      if (obj.crop_rel) {
        const url = `${this.urlPrefix}/${obj.crop_rel}`;
        const p = this._makeBillboard(url, obj.mean).then(spr => {
          if (spr) {
            this.scene.add(spr);
            mesh.billboard = spr;
            const show = this.isolatedIdx < 0 || ki === this.isolatedIdx;
            spr.visible = show && this.layers.billboards;
          }
        });
        billboardPromises.push(p);
      }
    }
    await Promise.allSettled(billboardPromises);
  }

  _b64ToXYZ(b64) {
    const bin = atob(b64);
    const buf = new ArrayBuffer(bin.length);
    const u8 = new Uint8Array(buf);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    const f32 = new Float32Array(buf);
    for (let i = 1; i < f32.length; i += 3) f32[i] = fy(f32[i]);
    return f32;
  }

  _makeBox(bmin, bmax, colorHex) {
    const ylo = Math.min(bmin[1], bmax[1]);
    const yhi = Math.max(bmin[1], bmax[1]);
    const geo = new THREE.BoxGeometry(bmax[0] - bmin[0], yhi - ylo, bmax[2] - bmin[2]);
    const edges = new THREE.EdgesGeometry(geo);
    geo.dispose();
    const col = new THREE.Color(colorHex);
    const ls = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.4 }),
    );
    ls.position.set((bmax[0] + bmin[0]) / 2, (yhi + ylo) / 2, (bmax[2] + bmin[2]) / 2);
    return ls;
  }

  _makeLabel(text, pos, colorHex) {
    const div = document.createElement('div');
    div.className = 'label3d';
    div.textContent = text.length > 28 ? text.slice(0, 27) + '…' : text;
    div.style.color = colorHex;
    const obj = new CSS2DObject(div);
    obj.position.set(pos[0], pos[1], pos[2]);
    return obj;
  }

  _makeObjCloud(xyzFlat, colorHex) {
    const n = xyzFlat.length / 3;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(xyzFlat, 3));
    const col = new THREE.Color(colorHex);
    const ca = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      ca[i * 3] = col.r;
      ca[i * 3 + 1] = col.g;
      ca[i * 3 + 2] = col.b;
    }
    geo.setAttribute('color', new THREE.BufferAttribute(ca, 3));
    return new THREE.Points(
      geo,
      new THREE.PointsMaterial({ size: 0.04, vertexColors: true, sizeAttenuation: true }),
    );
  }

  _makeBillboard(url, pos) {
    return new Promise(resolve => {
      new THREE.TextureLoader().load(
        url,
        tex => {
          tex.colorSpace = THREE.SRGBColorSpace;
          const w = tex.image?.width || 1;
          const h = tex.image?.height || 1;
          const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, opacity: 0.92 });
          const spr = new THREE.Sprite(mat);
          const ht = 0.8;
          spr.scale.set(ht * (w / h), ht, 1);
          spr.position.set(pos[0], pos[1] + 0.6, pos[2]);
          resolve(spr);
        },
        undefined,
        () => resolve(null),
      );
    });
  }
}
