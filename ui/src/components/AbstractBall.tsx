/**
 * The glob. Your Three.js component, with three changes:
 *
 * 1. Shaders come from lib/shaders.ts instead of document.getElementById().
 *    The original lookup ran in useEffect against <script> tags in the same
 *    component's return, so on first mount they didn't exist yet.
 *
 * 2. The scene is built once. Your version listed every prop in the useEffect
 *    dependency array, which tore down and rebuilt the renderer, geometry and
 *    shader program on every colour change — several times a second while
 *    talking. Uniforms are updated through a ref in the animation loop instead.
 *
 * 3. The gsap block that randomised RGBr/g/b/n/m is gone. It was overwriting
 *    exactly the uniforms that carry state colour, so blue-idle would have
 *    drifted to something arbitrary within a second. gsap still drives the
 *    transitions — it just tweens toward the target state now instead of
 *    toward Math.random().
 */

import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import type { GlobConfig } from '../lib/globStates';
import { FRAGMENT_SHADER, VERTEX_SHADER } from '../lib/shaders';

interface Props extends Partial<GlobConfig> {
  className?: string;
}

export default function AbstractBall({
  perlinTime = 25.0,
  perlinMorph = 25.0,
  perlinDNoise = 0.0,
  chromaRGBr = 7.5,
  chromaRGBg = 5.0,
  chromaRGBb = 7.0,
  chromaRGBn = 1.0,
  chromaRGBm = 1.0,
  tint = [0.2, 0.45, 1.0],
  tintMix = 0.9,
  gain = 1.0,
  sphereWireframe = false,
  spherePoints = false,
  spherePsize = 1.0,
  cameraSpeedY = 0.0,
  cameraSpeedX = 0.0,
  cameraZoom = 175,
  className,
}: Props) {
  const mountRef = useRef<HTMLDivElement>(null);

  // Every frame reads from here. Props write to it on each render. That's the
  // trick that lets the scene stay alive across prop changes — built once so
  // adding a uniform can't drift between the initialiser and the update.
  const current: GlobConfig = {
    perlinTime, perlinMorph, perlinDNoise,
    chromaRGBr, chromaRGBg, chromaRGBb, chromaRGBn, chromaRGBm,
    tint, tintMix, gain, sphereWireframe, spherePoints, spherePsize,
    cameraSpeedY, cameraSpeedX, cameraZoom,
  };
  const live = useRef<GlobConfig>(current);
  live.current = current;

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const width = mount.clientWidth;
    const height = mount.clientHeight;

    const uniforms = {
      time: { value: 0.0 },
      RGBr: { value: chromaRGBr / 10 },
      RGBg: { value: chromaRGBg / 10 },
      RGBb: { value: chromaRGBb / 10 },
      RGBn: { value: chromaRGBn / 100 },
      RGBm: { value: chromaRGBm },
      morph: { value: perlinMorph },
      dnoise: { value: perlinDNoise },
      psize: { value: spherePsize },
      tint: { value: new THREE.Vector3(...tint) },
      tintMix: { value: tintMix },
      gain: { value: gain },
    };

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(20, width / height, 1, 1000);
    camera.position.set(0, 10, 300 - cameraZoom);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    mount.appendChild(renderer.domElement);

    const geometry = new THREE.IcosahedronGeometry(20, 20);
    const material = new THREE.ShaderMaterial({
      uniforms,
      side: THREE.DoubleSide,
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      wireframe: sphereWireframe,
    });

    const mesh = new THREE.Mesh(geometry, material);
    const point = new THREE.Points(geometry, material);
    scene.add(mesh);
    scene.add(point);

    let frameId = 0;

    const animate = () => {
      const c = live.current;

      uniforms.time.value += c.perlinTime / 10000;
      uniforms.morph.value = c.perlinMorph;
      uniforms.dnoise.value = c.perlinDNoise;
      uniforms.RGBr.value = c.chromaRGBr / 10;
      uniforms.RGBg.value = c.chromaRGBg / 10;
      uniforms.RGBb.value = c.chromaRGBb / 10;
      uniforms.RGBn.value = c.chromaRGBn / 100;
      uniforms.RGBm.value = c.chromaRGBm;
      uniforms.psize.value = c.spherePsize;
      uniforms.tint.value.set(c.tint[0], c.tint[1], c.tint[2]);
      uniforms.tintMix.value = c.tintMix;
      uniforms.gain.value = c.gain;

      mesh.rotation.y += c.cameraSpeedY / 100;
      mesh.rotation.z += c.cameraSpeedX / 100;
      point.rotation.y = mesh.rotation.y;
      point.rotation.z = mesh.rotation.z;

      material.wireframe = c.sphereWireframe;
      mesh.visible = !c.spherePoints;
      point.visible = c.spherePoints;

      // Eased rather than jumped, so a zoom change reads as a move not a cut.
      const targetZ = 300 - c.cameraZoom;
      camera.position.z += (targetZ - camera.position.z) * 0.04;

      camera.lookAt(scene.position);
      renderer.render(scene, camera);
      frameId = requestAnimationFrame(animate);
    };
    animate();

    const handleResize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(frameId);
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
      geometry.dispose();
      material.dispose();
      renderer.dispose();
    };
    // Intentionally empty: the scene is built once and driven by the live ref.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return <div ref={mountRef} className={className} />;
}
