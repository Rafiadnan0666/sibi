// Joint (sendi) feature engineering + skeleton analytics.
// Feature contract MUST match training (python retrain_joint.py):
//   px = [x*W, y*H, z_raw] per joint (MediaPipe indices 0..20),
//   feat = (px - wrist) / ||middle_mcp - wrist||, flattened -> 63-vector.
// Mirror-agnostic: training saw mirrored copies, so pass hands through as-is.

import type { HandLandmark } from './types';

export const JOINT_NAMES = [
	'wrist',
	'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
	'index_mcp', 'index_pip', 'index_dip', 'index_tip',
	'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
	'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
	'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
];

export interface FingerInfo {
	key: string;
	name: string;
	joints: number[];
	color: string;
}

export const FINGERS: FingerInfo[] = [
	{ key: 'thumb', name: 'Jempol', joints: [1, 2, 3, 4], color: '#FF90E8' },
	{ key: 'index', name: 'Telunjuk', joints: [5, 6, 7, 8], color: '#FFD02B' },
	{ key: 'middle', name: 'Tengah', joints: [9, 10, 11, 12], color: '#7DF9FF' },
	{ key: 'ring', name: 'Manis', joints: [13, 14, 15, 16], color: '#7CFC98' },
	{ key: 'pinky', name: 'Kelingking', joints: [17, 18, 19, 20], color: '#C4B5FD' }
];

export const BONES: [number, number][] = [
	[0, 1], [1, 2], [2, 3], [3, 4],
	[0, 5], [5, 6], [6, 7], [7, 8],
	[0, 9], [9, 10], [10, 11], [11, 12],
	[0, 13], [13, 14], [14, 15], [15, 16],
	[0, 17], [17, 18], [18, 19], [19, 20]
];

type Pt = { x: number; y: number; z: number };

function pts(lm: HandLandmark[], W: number, H: number): Pt[] {
	return lm.map((p) => ({ x: p.x * W, y: p.y * H, z: p.z ?? 0 }));
}

/** Normalized 63-vector for the primary joint model (translation+scale invariant). */
export function landmarksToFeatures(lm: HandLandmark[], W: number, H: number): Float32Array {
	const P = pts(lm, W, H);
	const w = P[0];
	const mcp = P[9];
	// EXACT training replica: size in pixel space, z kept in raw MP units.
	const size =
		Math.hypot(mcp.x - w.x, mcp.y - w.y, mcp.z - w.z) + 1e-6;
	const out = new Float32Array(63);
	for (let i = 0; i < 21; i++) {
		out[i * 3] = (P[i].x - w.x) / size;
		out[i * 3 + 1] = (P[i].y - w.y) / size;
		out[i * 3 + 2] = (P[i].z - w.z) / size;
	}
	return out;
}

/** Raw pixel 63-vector for the upstream baseline replica (pixel-space, like its CSVs). */
export function rawPixelFeatures(lm: HandLandmark[], W: number, H: number): Float32Array {
	const P = pts(lm, W, H);
	const out = new Float32Array(63);
	for (let i = 0; i < 21; i++) {
		out[i * 3] = P[i].x;
		out[i * 3 + 1] = P[i].y;
		out[i * 3 + 2] = P[i].z;
	}
	return out;
}

export function jointAngleDeg(a: Pt, b: Pt, c: Pt): number {
	const v1x = a.x - b.x;
	const v1y = a.y - b.y;
	const v2x = c.x - b.x;
	const v2y = c.y - b.y;
	const d1 = Math.hypot(v1x, v1y) + 1e-9;
	const d2 = Math.hypot(v2x, v2y) + 1e-9;
	const cos = Math.max(-1, Math.min(1, (v1x * v2x + v1y * v2y) / (d1 * d2)));
	return (Math.acos(cos) * 180) / Math.PI;
}

export interface FingerState extends FingerInfo {
	extended: boolean;
	curl: number; // 0 = lurus, 1 = menekuk penuh
	tipAngle: number;
}

export function fingerStates(lm: HandLandmark[], W: number, H: number): FingerState[] {
	const P = pts(lm, W, H);
	const wrist = P[0];
	return FINGERS.map((f) => {
		const [mcp, pip, dip, tip] = f.joints.map((j) => P[j]);
		const dTip = Math.hypot(tip.x - wrist.x, tip.y - wrist.y);
		const dPip = Math.hypot(pip.x - wrist.x, pip.y - wrist.y);
		const ang = jointAngleDeg(mcp, pip, dip);
		const ang2 = jointAngleDeg(pip, dip, tip);
		const curl = Math.max(0, Math.min(1, 1 - (ang + ang2 - 60) / 220));
		return {
			...f,
			extended: dTip > dPip * 1.06 && curl < 0.55,
			curl,
			tipAngle: Math.round(ang2)
		};
	});
}

/** Draw the enlarged joint map: numbered sendi 0-20, colored bones, fingertip rings. */
export function drawSkeleton(
	ctx: CanvasRenderingContext2D,
	lm: HandLandmark[],
	cw: number,
	ch: number,
	opts: { highlight?: number[]; showNumbers?: boolean } = {}
) {
	const showNumbers = opts.showNumbers ?? true;
	let minX = Infinity;
	let minY = Infinity;
	let maxX = -Infinity;
	let maxY = -Infinity;
	for (const p of lm) {
		minX = Math.min(minX, p.x);
		minY = Math.min(minY, p.y);
		maxX = Math.max(maxX, p.x);
		maxY = Math.max(maxY, p.y);
	}
	const pad = 0.12;
	const bw = Math.max(1e-6, maxX - minX);
	const bh = Math.max(1e-6, maxY - minY);
	const s = Math.min(cw / (bw * (1 + pad * 2)), ch / (bh * (1 + pad * 2)));
	const ox = (cw - bw * s) / 2 - (minX - bw * pad) * s;
	const oy = (ch - bh * s) / 2 - (minY - bh * pad) * s;
	const X = (i: number) => lm[i].x * s + ox; // peta sendi tak-dimirror (sesuai ruang latih)
	const Y = (i: number) => lm[i].y * s + oy;

	const fingerOf = (bone: [number, number]) => FINGERS.find((f) => f.joints.includes(bone[1]));

	ctx.lineCap = 'round';
	for (const b of BONES) {
		const f = fingerOf(b);
		ctx.strokeStyle = f?.color ?? '#111';
		ctx.lineWidth = b[1] % 4 === 0 ? 9 : 7;
		ctx.beginPath();
		ctx.moveTo(X(b[0]), Y(b[0]));
		ctx.lineTo(X(b[1]), Y(b[1]));
		ctx.stroke();
		ctx.strokeStyle = '#111';
		ctx.lineWidth = 1.5;
		ctx.stroke();
	}
	for (let i = 0; i < 21; i++) {
		const hot = opts.highlight?.includes(i) ?? false;
		const isTip = i === 0 || [4, 8, 12, 16, 20].includes(i);
		ctx.beginPath();
		ctx.arc(X(i), Y(i), isTip ? 11 : 8.5, 0, Math.PI * 2);
		ctx.fillStyle = hot ? '#16a34a' : i === 0 ? '#111' : '#fff';
		ctx.fill();
		ctx.lineWidth = 2.5;
		ctx.strokeStyle = '#111';
		ctx.stroke();
		if (showNumbers) {
			ctx.fillStyle = hot ? '#fff' : i === 0 ? '#FFD02B' : '#111';
			ctx.font = '900 10px "Space Grotesk", sans-serif';
			ctx.textAlign = 'center';
			ctx.textBaseline = 'middle';
			ctx.fillText(String(i), X(i), Y(i));
		}
	}
}

/** Tiny sparkline of the 63-dim feature vector (what the MLP actually "sees"). */
export function drawFeatSpark(
	ctx: CanvasRenderingContext2D,
	feats: Float32Array | number[],
	cw: number,
	ch: number
) {
	ctx.clearRect(0, 0, cw, ch);
	const n = feats.length;
	const bw = cw / n;
	let max = 0.001;
	for (const v of feats) max = Math.max(max, Math.abs(v));
	const mid = ch / 2;
	ctx.fillStyle = '#111';
	ctx.fillRect(0, mid - 1, cw, 2);
	for (let i = 0; i < n; i++) {
		const v = feats[i] / max;
		const h = (v * (ch / 2 - 4)) | 0;
		ctx.fillStyle = i % 3 === 2 ? '#FF90E8' : i % 3 === 1 ? '#111' : '#b88a00';
		if (h >= 0) ctx.fillRect(i * bw, mid - h, Math.max(1, bw - 0.5), h);
		else ctx.fillRect(i * bw, mid, Math.max(1, bw - 0.5), -h);
	}
}
