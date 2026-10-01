// MediaPipe HandLandmarker wrapper + application-control gesture decoder.
// IMPORTANT: these gestures are UI controls, NOT SIBI signs. SIBI meaning
// comes from the CNN classifier; gestures below only drive the interface.

import type { HandLandmark } from './types';

export type ControlGesture =
	| 'none'
	| 'point' // ☝ move cursor / hover prediction
	| 'pinch' // 🤏 thumb+index close -> confirm / click
	| 'fist' // ✊ delete last char/word
	| 'peace' // ✌️ space / commit word
	| 'palm' // 🖐️ pause / resume recognition
	| 'thumbsup'; // 👍 confirm / speak sentence

export interface DecodedHand {
	landmarks: HandLandmark[];
	gesture: ControlGesture;
	pinch: boolean;
	pinchDist: number;
	indexTip: { x: number; y: number }; // normalized 0..1 (mirrored for display)
	box: { x: number; y: number; w: number; h: number }; // pixel box in video space
	handedness: string;
}

// MediaPipe landmark indices
const TIP = { thumb: 4, index: 8, middle: 12, ring: 16, pinky: 20 };
const PIP = { thumb: 3, index: 6, middle: 10, ring: 14, pinky: 18 };
const MCP = { index: 5, middle: 9, ring: 13, pinky: 17 };

function dist(a: HandLandmark, b: HandLandmark): number {
	const dx = a.x - b.x;
	const dy = a.y - b.y;
	return Math.hypot(dx, dy);
}

function fingerExtended(lm: HandLandmark[], tip: number, pip: number, mcp: number): boolean {
	// extended if tip is farther from wrist than pip, with margin
	const wrist = lm[0];
	return dist(lm[tip], wrist) > dist(lm[pip], wrist) * 1.08 && lm[tip].y < lm[pip].y + 0.06;
}

function thumbExtended(lm: HandLandmark[]): boolean {
	return dist(lm[TIP.thumb], lm[MCP.middle]) > dist(lm[PIP.thumb], lm[MCP.middle]) * 1.15;
}

export function decodeControlGesture(lm: HandLandmark[]): {
	gesture: ControlGesture;
	pinch: boolean;
	pinchDist: number;
} {
	const indexUp = fingerExtended(lm, TIP.index, PIP.index, MCP.index);
	const middleUp = fingerExtended(lm, TIP.middle, PIP.middle, MCP.middle);
	const ringUp = fingerExtended(lm, TIP.ring, PIP.ring, MCP.ring);
	const pinkyUp = fingerExtended(lm, TIP.pinky, PIP.pinky, MCP.pinky);
	const thumbUp = thumbExtended(lm);
	const pinchDist = dist(lm[TIP.thumb], lm[TIP.index]);
	const handSize = Math.max(1e-6, dist(lm[0], lm[MCP.middle]));
	const pinch = pinchDist / handSize < 0.42;

	const upCount = [indexUp, middleUp, ringUp, pinkyUp].filter(Boolean).length;

	let gesture: ControlGesture = 'none';
	if (!indexUp && !middleUp && !ringUp && !pinkyUp && !thumbUp) gesture = 'fist';
	else if (pinch && upCount <= 2) gesture = 'pinch';
	else if (indexUp && !middleUp && !ringUp && !pinkyUp) gesture = 'point';
	else if (indexUp && middleUp && !ringUp && !pinkyUp) gesture = 'peace';
	else if (upCount >= 4) gesture = 'palm';
	else if (thumbUp && !indexUp && !middleUp && !ringUp && !pinkyUp) gesture = 'thumbsup';

	return { gesture, pinch, pinchDist: pinchDist / handSize };
}

export function handBoundingBox(
	lm: HandLandmark[],
	videoW: number,
	videoH: number
): { x: number; y: number; w: number; h: number } {
	let minX = 1;
	let minY = 1;
	let maxX = 0;
	let maxY = 0;
	for (const p of lm) {
		minX = Math.min(minX, p.x);
		minY = Math.min(minY, p.y);
		maxX = Math.max(maxX, p.x);
		maxY = Math.max(maxY, p.y);
	}
	return {
		x: minX * videoW,
		y: minY * videoH,
		w: Math.max(1, (maxX - minX) * videoW),
		h: Math.max(1, (maxY - minY) * videoH)
	};
}

// --- HandLandmarker loader (lazy, CDN wasm) ----------------------------------

interface LandmarkerResult {
	landmarks?: HandLandmark[][];
	handedness?: { categoryName: string }[][];
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let landmarker: any = null;
let landmarkerLoading: Promise<unknown> | null = null;

export async function ensureHandLandmarker(): Promise<unknown> {
	if (landmarker) return landmarker;
	if (landmarkerLoading) return landmarkerLoading;
	landmarkerLoading = (async () => {
		const vision = await import('@mediapipe/tasks-vision');
		const { FilesetResolver, HandLandmarker } = vision;
		const fileset = await FilesetResolver.forVisionTasks(
			'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm'
		);
		landmarker = await HandLandmarker.createFromOptions(fileset, {
			baseOptions: {
				modelAssetPath:
					'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
				delegate: 'GPU'
			},
			runningMode: 'VIDEO',
			numHands: 2,
			minHandDetectionConfidence: 0.4,
			minHandPresenceConfidence: 0.5,
			minTrackingConfidence: 0.5
		});
		return landmarker;
	})();
	try {
		return await landmarkerLoading;
	} finally {
		landmarkerLoading = null;
	}
}

export function detectHands(
	video: HTMLVideoElement,
	nowMs: number
): { hands: DecodedHand[]; raw: LandmarkerResult | null } {
	if (!landmarker) return { hands: [], raw: null };
	let res: LandmarkerResult | null = null;
	try {
		res = landmarker.detectForVideo(video, nowMs) as LandmarkerResult;
	} catch {
		return { hands: [], raw: null };
	}
	const vw = video.videoWidth || 640;
	const vh = video.videoHeight || 480;
	const hands: DecodedHand[] = [];
	const list = res.landmarks ?? [];
	for (let i = 0; i < list.length; i++) {
		const lm = list[i];
		if (!lm || lm.length < 21) continue;
		const { gesture, pinch, pinchDist } = decodeControlGesture(lm);
		hands.push({
			landmarks: lm,
			gesture,
			pinch,
			pinchDist,
			indexTip: { x: 1 - lm[TIP.index].x, y: lm[TIP.index].y }, // mirrored X
			box: handBoundingBox(lm, vw, vh),
			handedness: res.handedness?.[i]?.[0]?.categoryName ?? 'Unknown'
		});
	}
	return { hands, raw: res };
}

export function drawHandOverlay(
	ctx: CanvasRenderingContext2D,
	hands: DecodedHand[],
	W: number,
	H: number
) {
	const CONN: [number, number][] = [
		[0, 1], [1, 2], [2, 3], [3, 4],
		[0, 5], [5, 6], [6, 7], [7, 8],
		[5, 9], [9, 10], [10, 11], [11, 12],
		[9, 13], [13, 14], [14, 15], [15, 16],
		[13, 17], [17, 18], [18, 19], [19, 20], [0, 17]
	];
	for (const h of hands) {
		// ROI box (mirrored to match displayed video)
		const bx = W - (h.box.x + h.box.w);
		ctx.lineWidth = 3;
		ctx.strokeStyle = h.gesture === 'pinch' ? '#16a34a' : '#111111';
		ctx.setLineDash([8, 5]);
		ctx.strokeRect(bx, h.box.y, h.box.w, h.box.h);
		ctx.setLineDash([]);
		ctx.lineWidth = 3.5;
		ctx.strokeStyle = '#111111';
		ctx.beginPath();
		for (const [a, b] of CONN) {
			const pa = h.landmarks[a];
			const pb = h.landmarks[b];
			ctx.moveTo((1 - pa.x) * W, pa.y * H);
			ctx.lineTo((1 - pb.x) * W, pb.y * H);
		}
		ctx.stroke();
		ctx.fillStyle = '#FFD02B';
		for (const p of h.landmarks) {
			ctx.beginPath();
			ctx.arc((1 - p.x) * W, p.y * H, 5.5, 0, Math.PI * 2);
			ctx.fill();
			ctx.lineWidth = 2;
			ctx.strokeStyle = '#111';
			ctx.stroke();
		}
	}
}
