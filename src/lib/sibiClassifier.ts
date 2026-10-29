// SIBI model engine: ONE primary trained TF.js graph model + 2 votes (ALL ON).
//
//  1. sibi-joint (PRIMARY, loaded at startup, ~236 KB): MLP on CANONICAL v2
//     21-joint vector: wrist-relative, hand-size normalized, rotated so the
//     middle finger points +Y (camera-roll invariant), mirrored so the thumb
//     base is always +X (left/right-hand invariant). Val acc 90.3%; mirror /
//     +-30deg / distance perturbations keep 90.3% (exact feature invariance),
//     simulated other-person finger ratios keep 86-90%.
//     Retrained here: python/retrain_joint.py (MUST match jointFeatures.ts).
//  2. sibi-joint-baseline (VOTE, loaded at startup): upstream AJustiago/SIBI-Recognition
//     Conv1D on raw pixel joints (MIT). Val acc 74.2%. On by default.
//  3. sibi-advanced (VOTE, loaded at startup): MobileNetV2 128px on YOUR SIBI photos
//     (5280 imgs, 24 classes x220, stratified 80/20 -> 4224/1056).
//     Test acc 84.8%, top-3 95.6%, macro-F1 84.8% (same split, seed 42).
//     Served at /models/sibi-advanced/model.json AND mirrored to
//     /models/sibi-image/model.json for backward compat.
//     Trained here: python/train_sibi_advanced.py. On by default.
//
// All three load at startup so every vote is live. Warmup failures mark the
// slot as error (never fake-ready). Use loadPrimaryOnly() for fast-first-paint.
// J and Z are excluded everywhere: they are dynamic SIBI signs (motion),
// not static handshapes. That is a documented limit, not a bug.

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import type { GraphModel } from '@tensorflow/tfjs';

export type ModelStatus = 'idle' | 'loading' | 'ready' | 'demo' | 'error';

export interface ModelSlot {
	key: 'joint' | 'baseline' | 'image';
	name: string;
	url: string;
	valAcc: string;
	credit: string;
	status: ModelStatus;
	model: GraphModel | null;
	labels: string[];
	/** Human-readable failure reason when status === 'error'. Shown in the UI. */
	lastError?: string;
}

export interface ScoredLetter {
	label: string;
	confidence: number;
}

export interface JointResult extends ScoredLetter {
	probs: number[];
	top3: ScoredLetter[];
	latencyMs: number;
}

export const JOINT_LABELS_FALLBACK = [
	'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
	'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
	'T', 'U', 'V', 'W', 'X', 'Y'
];

async function fetchLabels(url: string): Promise<string[]> {
	try {
		const r = await fetch(url);
		if (!r.ok) return JOINT_LABELS_FALLBACK;
		const j = (await r.json()) as unknown;
		return Array.isArray(j) && j.length > 0 ? (j as string[]) : JOINT_LABELS_FALLBACK;
	} catch {
		return JOINT_LABELS_FALLBACK;
	}
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function runGraph(model: GraphModel, input: tf.Tensor): Promise<Float32Array | null> {
	try {
		const r = (await model.executeAsync(input)) as tf.Tensor | tf.Tensor[];
		const ten = (Array.isArray(r) ? r[0] : r) as tf.Tensor;
		const data = (await ten.data()) as Float32Array;
		if (Array.isArray(r)) r.forEach((t: tf.Tensor) => t.dispose());
		else ten.dispose();
		return data;
	} catch {
		return null;
	}
}

function top(probs: ArrayLike<number>, labels: string[], k = 3): ScoredLetter[] {
	const idx = Array.from(probs)
		.map((p, i) => i)
		.sort((a, b) => (probs[b] as number) - (probs[a] as number))
		.slice(0, k);
	return idx.map((i) => ({ label: labels[i] ?? '?', confidence: probs[i] as number }));
}

export class SibiEngine {
	private inflight = new Map<ModelSlot['key'], Promise<void>>();
	slots: ModelSlot[] = [
		{
			key: 'joint', name: 'Sendi (utama)', url: '/models/sibi-joint/model.json',
			valAcc: '90,3%', credit: 'MLP kanonis v2, dilatih di sini (tangan/sudut/orang invarian)',
			status: 'idle', model: null, labels: []
		},
		{
			key: 'baseline', name: 'Sendi baseline', url: '/models/sibi-joint-baseline/model.json',
			valAcc: '74,2%', credit: 'AJustiago/SIBI-Recognition (MIT)',
			status: 'idle', model: null, labels: []
		},
		{
			key: 'image', name: 'Citra ROI', url: '/models/sibi-advanced/model.json',
			valAcc: '84,8%', credit: 'MobileNetV2 128px, trained here on 5280 SIBI photos (80/20)',
			status: 'idle', model: null, labels: []
		}
	];

	get readyCount(): number {
		return this.slots.filter((s) => s.status === 'ready').length;
	}

	get primaryReady(): boolean {
		return this.slots[0].status === 'ready';
	}

	async load(): Promise<void> {
		await tf.ready();
		try {
			await tf.setBackend('webgl');
		} catch {
			await tf.setBackend('cpu');
		}
		await tf.ready();
		// Startup: ALL models load in parallel so every vote is live.
		// Joint (~236 KB) resolves first; baseline (~4 MB) and image (~12 MB)
		// follow. Each slot reports its own status/error independently.
		await Promise.all([
			this.loadSlot('joint'),
			this.loadSlot('baseline'),
			this.loadSlot('image')
		]);
	}

	/** Load only the primary model (fast first paint, ~200 KB). */
	async loadPrimaryOnly(): Promise<void> {
		await tf.ready();
		try {
			await tf.setBackend('webgl');
		} catch {
			await tf.setBackend('cpu');
		}
		await tf.ready();
		await this.loadSlot('joint');
	}

	/** Load one slot on demand ('joint' | 'baseline' | 'image'). Safe to re-call. */
	async loadSlot(key: ModelSlot['key']): Promise<void> {
		const s = this.slots.find((x) => x.key === key);
		if (!s || s.status === 'ready') return;
		const running = this.inflight.get(key);
		if (running) return running;
		const p = this.doLoadSlot(s);
		this.inflight.set(key, p);
		try {
			await p;
		} finally {
			this.inflight.delete(key);
		}
	}

	private async doLoadSlot(s: ModelSlot): Promise<void> {
		s.status = 'loading';
		s.lastError = undefined;
		try {
			const labelsUrl = s.url.replace(/model\.json$/, 'labels.json');
			const [m, labels] = await Promise.all([
				tf.loadGraphModel(s.url),
				fetchLabels(labelsUrl)
			]);
			s.model = m;
			s.labels = labels.length >= 24 ? labels.slice(0, 24) : [...JOINT_LABELS_FALLBACK];
			try {
				const warm =
					s.key === 'joint'
						? tf.zeros([1, 63])
						: s.key === 'baseline'
							? tf.zeros([1, 63, 1])
							: tf.zeros([1, 128, 128, 3]);
				const r = (await m.executeAsync(warm)) as tf.Tensor | tf.Tensor[];
				if (Array.isArray(r)) r.forEach((t: tf.Tensor) => t.dispose());
				else (r as tf.Tensor).dispose();
				warm.dispose();
			} catch (warmErr) {
				// Warmup failure means inference will fail too: do NOT report
				// ready. Surface the error so the UI shows Gagal, not Siap.
				const msg = warmErr instanceof Error ? warmErr.message : String(warmErr);
				try {
					m.dispose();
				} catch {
					// ignore dispose errors
				}
				s.model = null;
				s.status = 'error';
				s.lastError = `Warmup gagal (${msg})`;
				return;
			}
			s.status = 'ready';
		} catch (err) {
			s.status = 'error';
			s.model = null;
			s.lastError = err instanceof Error ? err.message : String(err);
		}
	}

	/** Retry a failed slot (clears error state first). */
	async retrySlot(key: ModelSlot['key']): Promise<void> {
		const s = this.slots.find((x) => x.key === key);
		if (!s) return;
		if (s.status === 'loading' || s.status === 'ready') return;
		s.status = 'idle';
		s.lastError = undefined;
		await this.loadSlot(key);
	}

	/** Primary: 63-vector normalized joints -> letter + full distribution. */
	async predictJoint(feats: Float32Array): Promise<JointResult | null> {
		const s = this.slots[0];
		if (s.status !== 'ready' || !s.model) return null;
		if (!feats || feats.length !== 63) return null;
		for (let i = 0; i < feats.length; i++) if (!Number.isFinite(feats[i])) return null;
		const t0 = performance.now();
		const input = tf.tensor2d(feats, [1, 63]);
		const probs = await runGraph(s.model, input);
		input.dispose();
		if (!probs) return null;
		let best = 0;
		for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
		return {
			label: s.labels[best] ?? '?',
			confidence: probs[best],
			probs: Array.from(probs),
			top3: top(probs, s.labels, 3),
			latencyMs: performance.now() - t0
		};
	}

	/** Baseline replica: raw pixel joints (63,1) -> letter (outputs 24,25 dead). */
	async predictBaseline(px: Float32Array): Promise<ScoredLetter | null> {
		const s = this.slots[1];
		if (s.status !== 'ready' || !s.model) return null;
		if (!px || px.length !== 63) return null;
		const input = tf.tensor3d(px, [1, 63, 1]);
		const probs = await runGraph(s.model, input);
		input.dispose();
		if (!probs) return null;
		let best = 0;
		for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
		if (best >= 24) return { label: '?', confidence: probs[best] };
		return { label: s.labels[best] ?? '?', confidence: probs[best] };
	}

	/** Image cross-check: 128x128x3 [-1,1] ROI tensor -> letter. */
	async predictImage(tensorData: Float32Array): Promise<ScoredLetter | null> {
		const s = this.slots[2];
		if (s.status !== 'ready' || !s.model) return null;
		if (!tensorData || tensorData.length !== 128 * 128 * 3) return null;
		const input = tf.tensor4d(tensorData, [1, 128, 128, 3]);
		const probs = await runGraph(s.model, input);
		input.dispose();
		if (!probs) return null;
		let best = 0;
		for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
		return { label: s.labels[best] ?? '?', confidence: probs[best] };
	}

	dispose() {
		for (const s of this.slots) {
			try {
				s.model?.dispose();
			} catch {
				// noop
			}
			s.model = null;
			s.status = 'idle';
		}
	}
}

// --- Temporal smoothing ------------------------------------------------------
// Hold a sign steady: require `minFrames` agreements inside a sliding window,
// a minimum confidence, and a cooldown so one physical sign = one character.

export interface SmootherOptions {
	windowSize?: number;
	minFrames?: number;
	minConfidence?: number;
	cooldownMs?: number;
	holdMs?: number;
}

export class PredictionSmoother {
	private window: { label: string; conf: number; t: number }[] = [];
	private lastCommitAt = 0;
	private currentLabel = '';
	private labelStartAt = 0;
	private pendingLabel = '';
	private pendingCount = 0;
	lastEmitted: { label: string; conf: number } | null = null;
	progress = 0;

	constructor(private opts: SmootherOptions = {}) {}

	reset() {
		this.window = [];
		this.progress = 0;
		this.lastEmitted = null;
		this.lastCommitAt = 0;
		this.currentLabel = '';
		this.labelStartAt = 0;
		this.pendingLabel = '';
		this.pendingCount = 0;
	}

	/** Feed one raw frame prediction. Returns a committed char or null. */
	push(label: string, conf: number, now = performance.now()): string | null {
		const windowSize = this.opts.windowSize ?? 15;
		const minConf = this.opts.minConfidence ?? 0.6;
		const cooldown = this.opts.cooldownMs ?? 1200;
		const holdMs = this.opts.holdMs ?? 900;
		this.window.push({ label, conf, t: now });
		if (this.window.length > windowSize) this.window.shift();

		const counts = new Map<string, { n: number; sum: number }>();
		for (const w of this.window) {
			const e = counts.get(w.label) ?? { n: 0, sum: 0 };
			e.n += 1;
			e.sum += w.conf;
			counts.set(w.label, e);
		}
		let topLabel = '';
		let topN = 0;
		let topAvg = 0;
		for (const [k, v] of counts) {
			if (v.n > topN) {
				topLabel = k;
				topN = v.n;
				topAvg = v.sum / v.n;
			}
		}

		// Track how long the current majority label has been held, with a
		// small hysteresis so single-frame flickers don't reset the timer.
		if (topLabel) {
			if (topLabel === this.currentLabel) {
				this.pendingLabel = '';
				this.pendingCount = 0;
			} else {
				if (topLabel === this.pendingLabel) this.pendingCount++;
				else {
					this.pendingLabel = topLabel;
					this.pendingCount = 1;
				}
				if (this.pendingCount >= 1) {
					this.currentLabel = topLabel;
					this.labelStartAt = now;
					this.pendingLabel = '';
					this.pendingCount = 0;
				}
			}
		}
		const heldMs = this.currentLabel ? now - this.labelStartAt : 0;
		this.progress = Math.min(1, heldMs / holdMs);

		// Need the same label held for holdMs, enough confidence, and cooldown.
		const tail = this.window.slice(-3);
		const tailOk = tail.length >= 1 && tail.every((w) => w.label === this.currentLabel || w.label === topLabel);
		if (
			this.currentLabel &&
			topAvg >= minConf &&
			heldMs >= holdMs &&
			tailOk &&
			now - this.lastCommitAt >= cooldown
		) {
			this.lastCommitAt = now;
			this.lastEmitted = { label: this.currentLabel, conf: topAvg };
			this.window = [];
			this.progress = 0;
			this.currentLabel = '';
			this.labelStartAt = 0;
			return this.currentLabel;
		}
		return null;
	}
}
