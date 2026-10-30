// MediaPipe HandLandmarker wrapper + application-control gesture decoder.
// IMPORTANT: these gestures are UI controls, NOT SIBI signs. SIBI meaning
// comes from the CNN classifier; gestures below only drive the interface.

import type { HandLandmark } from './types';
import type {
	FilesetResolver,
	HandLandmarker,
	HandLandmarkerResult,
	NormalizedLandmark
} from '@mediapipe/tasks-vision';

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

/**
 * Detect the two handshapes used by the dynamic SIBI letters:
 *  - indexOnly (D shape) is the base for Z
 *  - pinkyOnly (I shape) is the base for J
 * The thumb is ignored on purpose: SIBI I / D handshapes vary between signers
 * and a stiff thumb must not break recognition.
 */
export function fingerPattern(lm: HandLandmark[]): {
	indexOnly: boolean;
	pinkyOnly: boolean;
} {
	const indexUp = fingerExtended(lm, TIP.index, PIP.index, MCP.index);
	const middleUp = fingerExtended(lm, TIP.middle, PIP.middle, MCP.middle);
	const ringUp = fingerExtended(lm, TIP.ring, PIP.ring, MCP.ring);
	const pinkyUp = fingerExtended(lm, TIP.pinky, PIP.pinky, MCP.pinky);
	return {
		indexOnly: indexUp && !middleUp && !ringUp && !pinkyUp,
		pinkyOnly: pinkyUp && !indexUp && !middleUp && !ringUp
	};
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

export type VisionStatus = 'idle' | 'loading' | 'ready' | 'error';

let visionStatus: VisionStatus = 'idle';
let visionError: string = '';

export function getVisionStatus(): VisionStatus {
	return visionStatus;
}

export function getVisionError(): string {
	return visionError;
}

function toHandLandmarks(list: NormalizedLandmark[][]): HandLandmark[][] {
	return list.map((hand) => hand.map((p) => ({ x: p.x, y: p.y, z: p.z })));
}

let landmarker: HandLandmarker | null = null;
let landmarkerLoading: Promise<HandLandmarker> | null = null;
// MediaPipe VIDEO mode rejects non-increasing timestamps; rAF timestamps are
// usually monotonic, but background tabs / duplicate frames can repeat them.
let lastDetectTs = 0;
let prevLm: HandLandmark[] | null = null;
let lostFrames = 0;

async function createLandmarker(
	FilesetResolverCtor: typeof FilesetResolver,
	HandLandmarkerCtor: typeof HandLandmarker,
	delegate: 'GPU' | 'CPU'
): Promise<HandLandmarker> {
	const fileset = await FilesetResolverCtor.forVisionTasks(
		'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm'
	);
	return await HandLandmarkerCtor.createFromOptions(fileset, {
		baseOptions: {
			modelAssetPath:
				'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
			delegate
		},
		runningMode: 'VIDEO',
		numHands: 1,
		minHandDetectionConfidence: 0.4,
		minHandPresenceConfidence: 0.6,
		minTrackingConfidence: 0.6
	});
}

export async function ensureHandLandmarker(): Promise<HandLandmarker> {
	if (landmarker) return landmarker;
	if (landmarkerLoading) return landmarkerLoading;
	visionStatus = 'loading';
	visionError = '';
	landmarkerLoading = (async () => {
		const vision = await import('@mediapipe/tasks-vision');
		const { FilesetResolver: ResolverCtor, HandLandmarker: Ctor } = vision;
		// GPU first for speed; CPU fallback for machines/browsers where the
		// GPU delegate fails (previously tracking silently never started).
		try {
			landmarker = await createLandmarker(ResolverCtor, Ctor, 'GPU');
		} catch (gpuErr) {
			try {
				landmarker = await createLandmarker(ResolverCtor, Ctor, 'CPU');
			} catch (cpuErr) {
				const msg =
					cpuErr instanceof Error ? cpuErr.message : gpuErr instanceof Error ? gpuErr.message : '';
				visionStatus = 'error';
				visionError = `Hand tracking gagal dimuat (GPU+CPU). Periksa koneksi internet. ${msg}`.trim();
				throw cpuErr;
			}
		}
		visionStatus = 'ready';
		return landmarker;
	})();
	try {
		return await landmarkerLoading;
	} catch (e) {
		// allow retry on next call
		landmarker = null;
		throw e;
	} finally {
		landmarkerLoading = null;
	}
}

/** Forget the cached landmarker so the next ensureHandLandmarker() retries. */
export function resetHandLandmarker(): void {
	try {
		landmarker?.close();
	} catch {
		// ignore close errors
	}
	landmarker = null;
	landmarkerLoading = null;
	lastDetectTs = 0;
	prevLm = null;
	if (visionStatus === 'error') visionStatus = 'idle';
}

export interface DetectOutput {
	hands: DecodedHand[];
	raw: HandLandmarkerResult | null;
}

export function detectHands(video: HTMLVideoElement, nowMs: number): DetectOutput {
	if (!landmarker) return { hands: [], raw: null };
	if (!video || video.readyState < 2 || (video.videoWidth || 0) === 0)
		return { hands: [], raw: null };
	// Guard monotonic timestamps required by VIDEO running mode.
	const ts = nowMs > lastDetectTs ? nowMs : lastDetectTs + 1;
	lastDetectTs = ts;
	let res: HandLandmarkerResult | null = null;
	try {
		res = landmarker.detectForVideo(video, ts);
	} catch {
		return { hands: [], raw: null };
	}
	const vw = video.videoWidth || 640;
	const vh = video.videoHeight || 480;
	const hands: DecodedHand[] = [];
	// Single-hand mode: keep only the highest-confidence detected hand, and
	// drop low-confidence detections so they can't pollute classification.
	const rawList = res?.landmarks ? toHandLandmarks(res.landmarks) : [];
	const scores = res?.handedness ?? [];
	const order = rawList
		.map((lm, i) => ({ i, score: scores[i]?.[0]?.score ?? 0 }))
		.sort((a, b) => b.score - a.score);
	const chosen = order.length > 0 && (order[0].score === 0 || order[0].score >= 0.5) ? order[0].i : -1;
	const lm = chosen >= 0 ? rawList[chosen] : null;
	if (lm && lm.length >= 21) {
		const { gesture, pinch, pinchDist } = decodeControlGesture(lm);
		hands.push({
			landmarks: lm,
			gesture,
			pinch,
			pinchDist,
			indexTip: { x: 1 - lm[TIP.index].x, y: lm[TIP.index].y },
			box: handBoundingBox(lm, vw, vh),
			handedness: res?.handedness?.[chosen]?.[0]?.categoryName ?? 'Unknown'
		});
	}
	// Light temporal smoothing of landmarks for stable classification across
	// environments (reduces jitter). Reset when hand is lost.
	if (hands.length === 0) {
		prevLm = null;
	} else if (prevLm && prevLm.length === hands[0].landmarks.length) {
		const a = 0.6;
		const smoothed = hands[0].landmarks.map((p, i) => ({
			x: prevLm![i].x * (1 - a) + p.x * a,
			y: prevLm![i].y * (1 - a) + p.y * a,
			z: (prevLm![i].z ?? 0) * (1 - a) + (p.z ?? 0) * a
		}));
		hands[0].landmarks = smoothed;
		hands[0].box = handBoundingBox(smoothed, vw, vh);
		prevLm = smoothed.map((p) => ({ ...p }));
	} else {
		prevLm = hands[0].landmarks.map((p) => ({ ...p }));
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
