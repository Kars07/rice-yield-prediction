import { useEffect, useRef, useMemo } from "react";
import { useMap, GeoJSON } from "react-leaflet";
import L from "leaflet";
import { type StateMetrics } from "@/data/csvProcessor";

interface Props {
  geoData: GeoJSON.FeatureCollection;
  metrics: Record<string, StateMetrics> | null;
}

function pip(lat: number, lng: number, ring: number[][]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i], [xj, yj] = ring[j];
    if (yi > lat !== yj > lat && lng < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function insideFeature(lat: number, lng: number, f: any): boolean {
  const { type, coordinates } = f.geometry;
  if (type === "Polygon") return pip(lat, lng, coordinates[0]);
  if (type === "MultiPolygon") return coordinates.some((p: number[][][]) => pip(lat, lng, p[0]));
  return false;
}

function bboxOf(f: any): [number, number, number, number] {
  const all: number[][] = [];
  const { type, coordinates } = f.geometry;
  if (type === "Polygon") coordinates[0].forEach((c: number[]) => all.push(c));
  else if (type === "MultiPolygon") coordinates.forEach((p: number[][][]) => p[0].forEach((c: number[]) => all.push(c)));
  let minLat = Infinity, maxLat = -Infinity, minLng = Infinity, maxLng = -Infinity;
  all.forEach(([lng, lat]) => { if (lat < minLat) minLat = lat; if (lat > maxLat) maxLat = lat; if (lng < minLng) minLng = lng; if (lng > maxLng) maxLng = lng; });
  return [minLat, maxLat, minLng, maxLng];
}

function hashStr(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  return h >>> 0;
}

function mulberry32(seed: number) {
  return () => { seed |= 0; seed = seed + 0x6d2b79f5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}

const STOPS = [
  { t: 0.00, r: 255, g: 255, b: 0   },
  { t: 0.28, r: 255, g: 180, b: 0   },
  { t: 0.45, r: 255, g: 100, b: 0   },
  { t: 0.62, r: 255, g: 20,  b: 0   },
  { t: 0.80, r: 180, g: 0,   b: 0   },
  { t: 1.00, r: 41,  g: 16,  b: 0   },
];

function tempToRgb(temp: number): [number, number, number] {
  // Map Nigeria's actual range 24–32°C to 0–1 for full gradient utilisation
  const v = Math.min(1, Math.max(0, (temp - 24) / 8));
  for (let i = 0; i < STOPS.length - 1; i++) {
    const lo = STOPS[i], hi = STOPS[i + 1];
    if (v >= lo.t && v <= hi.t) {
      const t = (v - lo.t) / (hi.t - lo.t);
      return [Math.round(lo.r + (hi.r - lo.r) * t), Math.round(lo.g + (hi.g - lo.g) * t), Math.round(lo.b + (hi.b - lo.b) * t)];
    }
  }
  return [41, 16, 0];
}

function rgba(r: number, g: number, b: number, a: number) { return `rgba(${r},${g},${b},${a})`; }

interface Cluster { lat: number; lng: number; temp: number; }
interface StateData { feature: any; avgTemp: number; clusters: Cluster[]; }

export const HeatSignatureLayer = ({ geoData, metrics }: Props) => {
  const map = useMap();
  const layerRef = useRef<any>(null);

  const stateData = useMemo((): StateData[] => {
    if (!metrics) return [];
    const out: StateData[] = [];
    geoData.features.forEach((feature: any) => {
      const name: string = feature?.properties?.shapeName ?? "";
      const m = metrics[name];
      if (!m) return;
      const bbox = bboxOf(feature);
      const [minLat, maxLat, minLng, maxLng] = bbox;
      const latSpan = maxLat - minLat || 1;
      const rng = mulberry32(hashStr(name));
      const clusters: Cluster[] = [];
      for (let c = 0; c < 120; c++) {
        let lat = 0, lng = 0, found = false;
        for (let t = 0; t < 60 && !found; t++) {
          lat = minLat + rng() * (maxLat - minLat);
          lng = minLng + rng() * (maxLng - minLng);
          if (insideFeature(lat, lng, feature)) found = true;
        }
        if (!found) continue;
        const s = hashStr(name) * 0.001;
        const spatial = (Math.sin(lat * 9.1 + s) * Math.cos(lng * 12.7 + s * 1.3) * 0.55
          + Math.cos(lat * 17.3 + s * 2) * Math.sin(lng * 8.4 + s * 0.7) * 0.30
          + Math.sin((lat + lng) * 6.5 + s * 1.7) * 0.15) * 7.0;  // ±7°C spread
        const latBias = ((lat - minLat) / latSpan - 0.5) * 4.0;
        clusters.push({ lat, lng, temp: Math.max(24, Math.min(32, m.avgTemperature + latBias + spatial)) });
      }
      out.push({ feature, avgTemp: m.avgTemperature, clusters });
    });
    return out;
  }, [geoData, metrics]);

  useEffect(() => {
    if (!stateData.length) return;

    const CanvasHeat = L.Layer.extend({
      _canvas: null as HTMLCanvasElement | null,

      onAdd(map: L.Map) {
        const pane = map.getPane("overlayPane")!;
        this._canvas = L.DomUtil.create("canvas", "", pane) as HTMLCanvasElement;
        Object.assign(this._canvas.style, { position: "absolute", top: "0", left: "0", pointerEvents: "none", zIndex: "200" });
        this._raf = null;
        // Throttled redraw — used for continuous pan so canvas stays locked to features
        this._throttledRedraw = () => {
          if (this._raf) cancelAnimationFrame(this._raf);
          this._raf = requestAnimationFrame(() => this._redraw());
        };
        map.on("move moveend zoomend resize", this._throttledRedraw, this);
        this._redraw();
      },

      onRemove(map: L.Map) {
        map.off("move moveend zoomend resize", this._throttledRedraw, this);
        if (this._raf) cancelAnimationFrame(this._raf);
        if (this._canvas) { L.DomUtil.remove(this._canvas); this._canvas = null; }
      },

      _redraw() {
        if (!this._canvas || !this._map) return;
        const map: L.Map = this._map;
        const size = map.getSize();
        this._canvas.width = size.x;
        this._canvas.height = size.y;
        L.DomUtil.setPosition(this._canvas, map.containerPointToLayerPoint([0, 0]));
        const ctx = this._canvas.getContext("2d")!;
        ctx.clearRect(0, 0, size.x, size.y);
        const zoom = map.getZoom();
        // Smaller tighter blobs — organic texture, no visible circles
        const cr = zoom < 7 ? 130 : zoom < 8 ? 80 : zoom < 9 ? 52 : zoom < 11 ? 32 : 20;

        stateData.forEach(({ feature, avgTemp, clusters }) => {
          ctx.save();

          // Set overall layer translucency — high enough to look like a raster, low enough to show basemap text
          ctx.globalAlpha = 0.82;

          // Build polygon clip path in screen pixels
          const path = new Path2D();
          const addRing = (ring: number[][]) => {
            ring.forEach(([lng, lat], i) => {
              const p = map.latLngToContainerPoint([lat, lng]);
              i === 0 ? path.moveTo(p.x, p.y) : path.lineTo(p.x, p.y);
            });
            path.closePath();
          };
          const { type, coordinates } = feature.geometry;
          if (type === "Polygon") coordinates.forEach((r: number[][]) => addRing(r));
          else if (type === "MultiPolygon") coordinates.forEach((poly: number[][][]) => poly.forEach((r: number[][]) => addRing(r)));

          ctx.clip(path);

          // 1. Base fill — state average temperature color (semi-transparent)
          const [br, bg, bb] = tempToRgb(avgTemp);
          ctx.fillStyle = rgba(br, bg, bb, 0.72);
          ctx.fill(path);

          // 2. Cluster radial gradients — tightly overlapping for smooth organic texture
          clusters.forEach(c => {
            const p = map.latLngToContainerPoint([c.lat, c.lng]);
            const [cr2, cg2, cb2] = tempToRgb(c.temp);
            const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, cr);
            g.addColorStop(0,   rgba(cr2, cg2, cb2, 0.60));
            g.addColorStop(0.4, rgba(cr2, cg2, cb2, 0.28));
            g.addColorStop(1,   rgba(cr2, cg2, cb2, 0));
            ctx.fillStyle = g;
            ctx.fillRect(p.x - cr, p.y - cr, cr * 2, cr * 2);
          });

          // 3. Second micro-variation pass — 30 smaller tighter blobs for pixel-level texture
          const microR = Math.max(18, cr * 0.35);
          clusters.slice(0, 30).forEach((c, idx) => {
            // Offset each micro-blob slightly from its parent cluster
            const offLat = c.lat + (Math.sin(idx * 2.3 + hashStr(feature.properties?.shapeName ?? "") * 0.001) * 0.15);
            const offLng = c.lng + (Math.cos(idx * 1.7 + hashStr(feature.properties?.shapeName ?? "") * 0.001) * 0.15);
            const pm = map.latLngToContainerPoint([offLat, offLng]);
            // Micro temp = cluster temp ± small nudge for texture
            const mt = Math.max(23, Math.min(37, c.temp + Math.sin(idx * 3.1) * 1.5));
            const [mr, mg, mb] = tempToRgb(mt);
            const gm = ctx.createRadialGradient(pm.x, pm.y, 0, pm.x, pm.y, microR);
            gm.addColorStop(0,   rgba(mr, mg, mb, 0.35));
            gm.addColorStop(0.6, rgba(mr, mg, mb, 0.10));
            gm.addColorStop(1,   rgba(mr, mg, mb, 0));
            ctx.fillStyle = gm;
            ctx.fillRect(pm.x - microR, pm.y - microR, microR * 2, microR * 2);
          });

          ctx.restore();
        });
      },
    });

    const layer = new (CanvasHeat as any)();
    layer.addTo(map);
    layerRef.current = layer;
    return () => { if (layerRef.current) { map.removeLayer(layerRef.current); layerRef.current = null; } };
  }, [map, stateData]);

  return (
    <>
      <GeoJSON data={geoData} style={{ color: "#222222", weight: 1, fillOpacity: 0, opacity: 0.85 }} />
      <div
        className="absolute z-[1000] bg-black/85 backdrop-blur-md text-white px-5 py-4 rounded-2xl shadow-2xl border border-white/15"
        style={{ bottom: "90px", left: "50%", transform: "translateX(-50%)", minWidth: 320 }}
      >
        <div className="text-[10px] font-bold mb-2.5 text-center tracking-widest uppercase text-white/70">Surface Temperature Heat Signatures</div>
        <div className="h-3.5 w-full rounded-full mb-2" style={{ background: "linear-gradient(to right, #ffff00 0%, #ffb400 28%, #ff6400 45%, #ff1400 62%, #b40000 80%, #291000 100%)" }} />
        <div className="flex justify-between text-[9px] font-semibold text-white/60">
          <span>24°C</span><span>27°C</span><span>30°C</span><span>33°C</span><span>35°C+</span>
        </div>
        <div className="text-[8px] text-white/35 text-center mt-2">Zoom in to reveal cluster-level variation</div>
      </div>
    </>
  );
};
