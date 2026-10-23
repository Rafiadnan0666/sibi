<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { SIBI_LABELS_24 } from '$lib/sibiLabels';
	import { suggest, toSentence, autoCorrectWord, type ScoredWord } from '$lib/languageModel';
	import { cropROI, preprocessToTensor, drawPreview } from '$lib/preprocessing';
	import {
		landmarksToFeatures,
		rawPixelFeatures,
		drawSkeleton,
		drawFeatSpark
	} from '$lib/jointFeatures';
	import {
		ensureHandLandmarker,
		resetHandLandmarker,
		getVisionError,
		detectHands,
		drawHandOverlay,
		type DecodedHand,
		type ControlGesture
	} from '$lib/handTracker';
	import {
		loadHistory,
		saveHistoryEntry,
		clearHistory,
		deleteHistoryEntry,
		loadSetting,
		saveSetting,
		type HistoryEntry
	} from '$lib/storage';
	import { speakIndonesian, stopSpeaking } from '$lib/tts';
	import {
		type SibiEngine,
		type PredictionSmoother,
		type ModelSlot,
		type ScoredLetter
	} from '$lib/sibiClassifier';

	// Typed engine + smoother so TS can track every call site (no any).
	let engine: SibiEngine | null = $state(null);
	let smoother: PredictionSmoother | null = $state(null);
	let modelSlots: ModelSlot[] = $state([]);
	let flashTimer: number = 0;

	// Text state: signs become chars, chars become words, words become a sentence.
	// Typing works three ways: camera signs, the manual input box, or physical
	// keyboard (A-Z append, Backspace delete, Space word, Enter speak).
	let prefix: string = $state('');
	let words: string[] = $state([]);
	let recentChars: string[] = $state([]);
	let suggestion: { suggestions: ScoredWord[]; mode: 'lengkapi kata' | 'kata berikutnya' } =
		$derived(suggest(words, prefix, 3));
	let sentence: string = $derived(toSentence(prefix ? [...words, prefix] : words));
	let stability: number = $state(0);

	// Recognition state: 1 primary joint model + 2 optional lazy votes.
	interface Top3 {
		label: string;
		confidence: number;
	}
	let jointTop3: Top3[] = $state([]);
	let jointLabel: string = $state('-');
	let jointConf: number = $state(0);
	let baselineRes: ScoredLetter | null = $state(null);
	let imageRes: ScoredLetter | null = $state(null);
	let fusedConf: number = $state(0);
	let inferMs: number = $state(0);
	let featVec: number[] = $state([]);

	// Camera and vision state.
	let videoEl: HTMLVideoElement | null = $state(null);
	let overlayEl: HTMLCanvasElement | null = $state(null);
	let previewEl: HTMLCanvasElement | null = $state(null);
	let jointCanvas: HTMLCanvasElement | null = $state(null);
	let sparkCanvas: HTMLCanvasElement | null = $state(null);
	let stream: MediaStream | null = $state(null);
	let camOn: boolean = $state(false);
	let camError: string = $state('');
	let noSignal: boolean = $state(false);
	let noSignalSince = 0;
	let devices: { id: string; label: string }[] = $state([]);
	let deviceId: string = $state('');
	let visionReady: boolean = $state(false);
	let visionError: string = $state('');
	let hands: DecodedHand[] = $state([]);
	let gesture: ControlGesture = $state('none');
	let lastGesture: ControlGesture = $state('none');
	let gestureHold = 0;
	let fistHoldAt = 0;
	let peaceHoldAt = 0;
	let holdProgress = $state(0);
	let holdLabel = $state('');
	let lastLm: DecodedHand['landmarks'] | null = $state(null);
	let lastMotion = 0;
	let fps: number = $state(0);
	let handsFound: boolean = $state(false);
	let handHint: string = $state('');
	let lastHintAt = 0;
	let debugOpen: boolean = $state(false);

	// Settings. All three models are ON by default; unchecking only pauses
	// that vote for the session (model stays cached for instant re-enable).
	let minConf: number = $state(0.65);
	let cooldownMs: number = $state(1500);
	let grayscale: boolean = $state(false);
	let normMode: 'neg-one-one' | 'zero-one' = $state('neg-one-one');
	let useBaseline: boolean = $state(true);
	let useImage: boolean = $state(true);
	let autoCorrect: boolean = $state(true);
	let dragFrom = $state(-1);

	// Guard against confusing alphabets: top1 must beat top2 by this margin,
	// otherwise the frame is ignored (no commit, no smoother pollution).
	const MARGIN_MIN = 0.1;

	let history: HistoryEntry[] = $state([]);
	let notice: string = $state('');
	let lastGestureAt: Record<string, number> = {};
	let lastJointAt = 0;
	let lastBaseAt = 0;
	let lastImgAt = 0;
	let lastUiAt = 0;
	let lastFrameAt = 0;
	let raf = 0;

	// Sample collector: record YOUR OWN hand as labeled training data.
	// Pick a letter, hold the sign, press record. Download CSV, then merge it:
	//   python python/retrain_joint.py --extra ./sibi-extra.csv
	// This is the only honest way to push accuracy up: more real hands.
	let collectLetter: string = $state('A');
	let collecting: boolean = $state(false);
	let collected: { label: string; feat: number[] }[] = $state([]);
	let lastCollectAt = 0;

	function toggleCollect() {
		collecting = !collecting;
		if (collecting) flash(`Merekam ${collectLetter}: tahan isyaratnya`);
		else flash(`Berhenti. Terkumpul ${collected.length} sampel.`);
	}

	function clearSamples() {
		collected = [];
		collecting = false;
		flash('Sampel dibuang');
	}

	function downloadSamples() {
		if (!collected.length) {
			flash('Belum ada sampel');
			return;
		}
		const head = ['class_type', ...Array.from({ length: 63 }, (_, i) => `f${i}`)];
		const lines = [head.join(',')];
		for (const s of collected) lines.push([s.label, ...s.feat].join(','));
		const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
		const a = document.createElement('a');
		a.href = URL.createObjectURL(blob);
		a.download = 'sibi-extra.csv';
		a.click();
		URL.revokeObjectURL(a.href);
		flash(`${collected.length} sampel diunduh`);
	}

	const readyCount = $derived(modelSlots.filter((s) => s.status === 'ready').length);
	const primaryReady = $derived(modelSlots[0]?.status === 'ready');

	const GESTURE_META: Record<string, { icon: string; label: string; fn: string }> = {
		fist: { icon: 'FIST', label: 'Fist: hapus', fn: 'Hapus huruf atau kata terakhir' },
		peace: { icon: 'PEACE', label: 'Peace: spasi', fn: 'Kunci huruf jadi kata' },
		thumbsup: { icon: 'OK', label: 'Thumbs up: ucapkan', fn: 'Bacakan kalimat' },
		palm: { icon: 'PALM', label: 'Palm: jeda', fn: 'Jeda pengenalan' }
	};

	// Text ops (shared by camera, buttons, manual input, and keyboard).
	function commitChar(c: string) {
		const ch = c.toUpperCase();
		if (!/^[A-Z]$/.test(ch)) return;
		if (prefix.length >= 24) return;
		prefix += ch;
	}

	function deleteLast() {
		if (prefix.length > 0) prefix = prefix.slice(0, -1);
		else if (words.length > 0) words = words.slice(0, -1);
	}

	function removeWord(i: number) {
		words = words.filter((_, j) => j !== i);
	}

	function commitSpace() {
		if (prefix.trim()) {
			const raw = prefix.trim().toUpperCase();
			let final = raw;
			if (autoCorrect) {
				const corrected = autoCorrectWord(raw);
				if (corrected !== raw) {
					final = corrected;
					flash(`Dikoreksi: ${raw} -> ${corrected}`);
				}
			}
			words = [...words, final];
			prefix = '';
		}
	}

	function onWordDragStart(i: number) {
		dragFrom = i;
	}
	function onWordDragOver(e: DragEvent) {
		e.preventDefault();
	}
	function onWordDrop(i: number) {
		if (dragFrom < 0 || dragFrom === i) {
			dragFrom = -1;
			return;
		}
		const arr = [...words];
		const [moved] = arr.splice(dragFrom, 1);
		arr.splice(i, 0, moved);
		words = arr;
		dragFrom = -1;
	}

	function moveWord(i: number, dir: number) {
		const j = i + dir;
		if (j < 0 || j >= words.length) return;
		const arr = [...words];
		const [moved] = arr.splice(i, 1);
		arr.splice(j, 0, moved);
		words = arr;
	}

	function applySuggestion(s: ScoredWord) {
		words = [...words, s.word.toUpperCase()];
		prefix = '';
		flash(`Dipilih: ${s.word}`);
	}

	function clearAll() {
		prefix = '';
		words = [];
		recentChars = [];
		smoother?.reset();
	}

	function speak() {
		if (!sentence.trim()) {
			flash('Belum ada kalimat');
			return;
		}
		speakIndonesian(sentence);
	}

	async function saveCurrent() {
		if (!sentence.trim()) {
			flash('Belum ada kalimat');
			return;
		}
		history = await saveHistoryEntry(sentence, prefix ? [...words, prefix] : [...words]);
		flash('Tersimpan ke riwayat');
	}

	function flash(msg: string) {
		notice = msg;
		window.clearTimeout(flashTimer);
		flashTimer = window.setTimeout(() => (notice = ''), 2200);
	}

	/** Manual typing box: only A-Z survive, max 24 chars. */
	function onManual(e: Event) {
		const v = (e.target as HTMLInputElement).value
			.toUpperCase()
			.replace(/[^A-Z]/g, '')
			.slice(0, 24);
		prefix = v;
	}

	/** Physical keyboard: A-Z append, Backspace delete, Space word, Enter speak. */
	function onKey(e: KeyboardEvent) {
		const t = e.target as HTMLElement | null;
		if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT'))
			return;
		if (e.ctrlKey || e.metaKey || e.altKey) return;
		if (/^[a-zA-Z]$/.test(e.key)) commitChar(e.key);
		else if (e.key === 'Backspace') {
			deleteLast();
			e.preventDefault();
		} else if (e.key === ' ') {
			commitSpace();
			e.preventDefault();
		} else if (e.key === 'Enter') speak();
	}

	function gestureCooldown(key: string, ms: number): boolean {
		const now = performance.now();
		if (now - (lastGestureAt[key] ?? 0) < ms) return false;
		lastGestureAt[key] = now;
		return true;
	}

	// Camera.
	async function refreshDevices() {
		try {
			const all = await navigator.mediaDevices.enumerateDevices();
			devices = all
				.filter((d) => d.kind === 'videoinput')
				.map((d, i) => ({ id: d.deviceId, label: d.label || `Kamera ${i + 1}` }));
			if (!deviceId && devices.length > 0) deviceId = devices[0].id;
		} catch {
			// enumeration unavailable: selector stays hidden
		}
	}

	function stopTracks() {
		stream?.getTracks().forEach((t) => t.stop());
		stream = null;
		if (videoEl) videoEl.srcObject = null;
	}

	async function openStream(): Promise<MediaStream> {
		const base = { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' };
		if (deviceId) {
			try {
				return await navigator.mediaDevices.getUserMedia({
					video: { ...base, deviceId: { exact: deviceId } },
					audio: false
				});
			} catch {
				// device unplugged: fall back to default camera
			}
		}
		return await navigator.mediaDevices.getUserMedia({ video: base, audio: false });
	}

	async function startCamera() {
		camError = '';
		noSignal = false;
		noSignalSince = 0;
		try {
			stopTracks();
			stream = await openStream();
			// The <video> element is permanent and never rebuilt by {#if},
			// so the stream always lands on the visible player.
			camOn = true;
			await tick();
			if (videoEl) {
				videoEl.srcObject = stream;
				videoEl.muted = true;
				try {
					await videoEl.play();
				} catch {
					// play() can be refused before interaction; the button counts as one
				}
			}
			await refreshDevices();
			lastFrameAt = performance.now();
			cancelAnimationFrame(raf);
			raf = requestAnimationFrame(loop);
		} catch (e) {
			camError =
				'Kamera tidak bisa dibuka. Izinkan kamera di browser, tutup aplikasi lain yang memakai kamera, lalu coba lagi. ' +
				(e instanceof Error ? e.message : '');
			camOn = false;
		}
	}

	async function switchCamera(id: string) {
		deviceId = id;
		if (camOn) await startCamera();
	}

	function stopCamera() {
		cancelAnimationFrame(raf);
		stopTracks();
		camOn = false;
		noSignal = false;
		noSignalSince = 0;
		hands = [];
		handsFound = false;
		gesture = 'none';
		lastLm = null;
		lastGesture = 'none';
		gestureHold = 0;
	}

	function loop(t: number) {
		if (!camOn) return;
		raf = requestAnimationFrame(loop);
		const dt = t - lastFrameAt;
		lastFrameAt = t;
		if (dt > 0) fps = Math.round(fps * 0.9 + (1000 / dt) * 0.1);
		if (!videoEl || videoEl.readyState < 2) return;

		// Watchdog: live stream but black frames (camera held by another app).
		if ((videoEl.videoWidth || 0) === 0) {
			if (!noSignalSince) noSignalSince = t;
			if (t - noSignalSince > 4000 && !noSignal) {
				noSignal = true;
				flash('Tidak ada gambar dari kamera. Coba ganti perangkat di bawah.');
			}
		} else if (noSignalSince || noSignal) {
			noSignalSince = 0;
			noSignal = false;
		}

		const vw = videoEl.videoWidth || 640;
		const vh = videoEl.videoHeight || 480;
		const { hands: found } = detectHands(videoEl, t);
		hands = found;
		handsFound = found.length > 0;
		const primary = found[0];
		gesture = primary?.gesture ?? 'none';

		// Framing hints so the user can position their hand where tracking works.
		if (t - lastHintAt > 500) {
			lastHintAt = t;
			if (!handsFound) {
				handHint = 'Tahan tangan di tengah bingkai. Cahaya cukup, latar polos.';
			} else if (primary) {
				const area = (primary.box.w * primary.box.h) / (vw * vh);
				handHint = area < 0.04 ? 'Terlalu jauh atau kecil. Dekatkan tangan ke kamera.' : '';
			}
		}

		if (overlayEl) {
			overlayEl.width = vw;
			overlayEl.height = vh;
			const ctx = overlayEl.getContext('2d')!;
			ctx.clearRect(0, 0, vw, vh);
			drawHandOverlay(ctx, found, vw, vh);
		}

		if (primary) {
			if (primary.gesture === lastGesture) gestureHold++;
			else {
				lastGesture = primary.gesture;
				gestureHold = 1;
			}
			// Require the gesture to be held before firing. Fist and peace need
			// a 2 second hold so transitional shapes can't trigger actions.
			const HOLD_MS = 2000;
			const now = performance.now();
			if (primary.gesture === 'fist') {
				if (!fistHoldAt) fistHoldAt = now;
				const held = now - fistHoldAt;
				holdProgress = Math.min(1, held / HOLD_MS);
				holdLabel = 'fist';
				if (held >= HOLD_MS && gestureCooldown('fist', 2200)) {
					deleteLast();
					fistHoldAt = 0;
					holdProgress = 0;
					holdLabel = '';
				}
			} else if (primary.gesture === 'peace') {
				if (!peaceHoldAt) peaceHoldAt = now;
				const held = now - peaceHoldAt;
				holdProgress = Math.min(1, held / HOLD_MS);
				holdLabel = 'peace';
				if (held >= HOLD_MS && gestureCooldown('peace', 2200)) {
					commitSpace();
					peaceHoldAt = 0;
					holdProgress = 0;
					holdLabel = '';
				}
			} else if (primary.gesture === 'thumbsup') {
				if (gestureCooldown('thumbsup', 2500)) speak();
				holdProgress = 0;
				holdLabel = '';
			} else {
				fistHoldAt = 0;
				peaceHoldAt = 0;
				holdProgress = 0;
				holdLabel = '';
			}
		} else {
			fistHoldAt = 0;
			peaceHoldAt = 0;
			holdProgress = 0;
			holdLabel = '';
		}

		void classifyFrame(primary, t, vw, vh);
	}

	async function classifyFrame(primary: DecodedHand | undefined, t: number, vw: number, vh: number) {
		if (!engine || !smoother || !primary || !videoEl) return;
		// Palm = pause recognition so the user can hold without committing.
		if (primary.gesture === 'palm') return;
		// Reject tiny / poorly framed hands so they can't become garbage letters.
		const area = (primary.box.w * primary.box.h) / (vw * vh);
		if (area < 0.06 || area > 0.8) return;
		// Motion gate: skip transitional frames (hand moving) to avoid
		// committing a letter mid-gesture.
		if (lastLm && lastLm.length === primary.landmarks.length) {
			let s = 0;
			for (let i = 0; i < primary.landmarks.length; i++) {
				s += Math.hypot(primary.landmarks[i].x - lastLm[i].x, primary.landmarks[i].y - lastLm[i].y);
			}
			lastMotion = s / primary.landmarks.length;
			if (lastMotion > 0.02) {
				lastLm = primary.landmarks;
				return;
			}
		}
		lastLm = primary.landmarks;
		// Lighting gate: very dark/bright frames give unstable detection.
		try {
			const c = document.createElement('canvas');
			c.width = 24; c.height = 24;
			const ctx = c.getContext('2d', { willReadFrequently: true })!;
			ctx.drawImage(videoEl, primary.box.x, primary.box.y, primary.box.w, primary.box.h, 0, 0, 24, 24);
			const d = ctx.getImageData(0, 0, 24, 24).data;
			let lum = 0;
			for (let i = 0; i < d.length; i += 4) lum += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
			lum /= d.length / 4;
			if (lum < 45 || lum > 215) {
				handHint = lum < 45 ? 'Terlalu gelap. Tambah cahaya.' : 'Terlalu terang / overexposed.';
				return;
			}
		} catch {
			// ignore brightness check failures
		}
		// UI gestures take priority so they never pollute the letter stream.
		if (primary.gesture === 'fist' || primary.gesture === 'peace') return;
		if (t - lastJointAt >= 110 && engine.slots[0].status === 'ready') {
			lastJointAt = t;
			try {
				const feats = landmarksToFeatures(primary.landmarks, vw, vh);
				// Sample collector: same canonical vector the model sees.
				if (collecting && t - lastCollectAt > 150) {
					lastCollectAt = t;
					collected = [...collected, { label: collectLetter, feat: Array.from(feats) }];
					if (collected.length >= 2000) {
						collecting = false;
						flash('2000 sampel cukup. Unduh lalu latih ulang.');
					}
				}
				const out = await engine.predictJoint(feats);
				if (out) {
					inferMs = Math.round(out.latencyMs);
					const baseAgree = baselineRes?.label === out.label;
					const imgAgree = imageRes?.label === out.label;
					let fused = out.confidence + (baseAgree ? 0.07 : 0) + (imgAgree ? 0.07 : 0);
					fused = Math.min(0.99, fused);
					fusedConf = fused;
					if (t - lastUiAt > 150) {
						lastUiAt = t;
						jointLabel = out.label;
						jointConf = out.confidence;
						jointTop3 = out.top3;
						featVec = Array.from(feats);
					}
					stability = smoother.progress;
					const margin =
						out.top3.length > 1 ? out.top3[0].confidence - out.top3[1].confidence : 1;
					// Confusing-alphabet guard: 50/50 frames never commit.
					if (fused >= minConf && margin >= MARGIN_MIN) {
						const committed = smoother.push(out.label, fused, t);
						if (committed) {
							commitChar(committed);
							recentChars = [...recentChars, committed].slice(-10);
							flash(`Terkunci: ${committed}`);
						}
						stability = smoother.progress;
					}
				}
			} catch {
				// one bad frame must never kill the loop
			}
		}

		// Path 2 (~4fps): baseline replica votes too, only if the user enabled it.
		if (useBaseline && t - lastBaseAt >= 250 && engine.slots[1].status === 'ready') {
			lastBaseAt = t;
			try {
				const px = rawPixelFeatures(primary.landmarks, vw, vh);
				baselineRes = await engine.predictBaseline(px);
			} catch {
				// ignore
			}
		}

		// Path 3 (~5fps): image CNN checks the ROI crop, only if enabled.
		if (useImage && t - lastImgAt >= 200 && engine.slots[2].status === 'ready') {
			lastImgAt = t;
			try {
				const roi = cropROI(videoEl, primary.box);
				const baseOpts = { size: 128, normalize: normMode, grayscale };
				const tensor = preprocessToTensor(roi, baseOpts);
				if (debugOpen && previewEl) drawPreview(tensor, previewEl, normMode);
				let res = await engine.predictImage(tensor.data);
				// Hand invariance for the image vote: try the mirrored ROI too and
				// keep whichever orientation the model is more confident about.
				try {
					const tensorM = preprocessToTensor(roi, { ...baseOpts, mirror: true });
					const resM = await engine.predictImage(tensorM.data);
					if (resM && (!res || resM.confidence > res.confidence)) res = resM;
				} catch {
					// ignore mirror attempt failures
				}
				imageRes = res;
			} catch {
				// ignore
			}
		}

		// Debug canvases draw only while the debug panel is open (speed).
		if (debugOpen && jointCanvas && primary) {
			const ctx = jointCanvas.getContext('2d')!;
			ctx.clearRect(0, 0, jointCanvas.width, jointCanvas.height);
			drawSkeleton(ctx, primary.landmarks, jointCanvas.width, jointCanvas.height, {
				highlight: [0, 4, 8, 12, 16, 20]
			});
		}
		if (debugOpen && sparkCanvas && featVec.length === 63) {
			drawFeatSpark(sparkCanvas.getContext('2d')!, featVec, sparkCanvas.width, sparkCanvas.height);
		}
	}

	/** Vote checkboxes pause/resume that vote. First enable lazy-loads it. */
	async function onToggleVote(kind: 'baseline' | 'image') {
		const on = kind === 'baseline' ? useBaseline : useImage;
		await saveSetting(kind === 'baseline' ? 'useBaseline' : 'useImage', on);
		if (on && engine) {
			const slot = engine.slots.find((s) => s.key === kind);
			if (slot && slot.status !== 'ready' && (slot.status as string) !== 'loading') {
				// Show Memuat... immediately (engine mutates nested status).
				slot.status = 'loading';
				modelSlots = [...engine.slots];
				await engine.loadSlot(kind);
				modelSlots = [...engine.slots];
				const after = engine.slots.find((s) => s.key === kind);
				if (after?.status === 'error') {
					flash(`Vote ${kind} gagal: ${after.lastError ?? 'cek koneksi'} - coba lagi di bawah.`);
				}
			}
		}
	}

	/** Turn every model on and (re)load any that failed. */
	async function enableAllModels() {
		useBaseline = true;
		useImage = true;
		autoCorrect = true;
		await saveSetting('useBaseline', true);
		await saveSetting('useImage', true);
		await saveSetting('autoCorrect', true);
		if (!engine) return;
		for (const s of engine.slots) {
			if (s.status !== 'ready' && s.status !== 'loading') s.status = 'loading';
		}
		modelSlots = [...engine.slots];
		await Promise.all([
			engine.loadSlot('joint').catch(() => {}),
			engine.loadSlot('baseline').catch(() => {}),
			engine.loadSlot('image').catch(() => {})
		]);
		modelSlots = [...engine.slots];
		const ready = engine.readyCount;
		flash(ready === 3 ? 'Semua 3 model siap.' : `${ready}/3 siap. Yang gagal bisa coba lagi.`);
	}

	async function retryModel(key: 'joint' | 'baseline' | 'image') {
		if (!engine) return;
		modelSlots = [...engine.slots];
		await engine.retrySlot(key);
		modelSlots = [...engine.slots];
	}

	async function retryVision() {
		visionError = '';
		resetHandLandmarker();
		try {
			await ensureHandLandmarker();
			visionReady = true;
			flash('Tracking siap. Nyalakan kamera.');
		} catch {
			visionError = getVisionError() || 'Hand tracking gagal dimuat. Periksa koneksi internet.';
		}
	}

	// Init.
	onMount(() => {
		let cancelled = false;
		window.addEventListener('keydown', onKey);
		(async () => {
			history = await loadHistory();
			minConf = await loadSetting('minConf', 0.65);
			cooldownMs = await loadSetting('cooldownMs', 1500);
			grayscale = await loadSetting('grayscale', false);
			useBaseline = await loadSetting('useBaseline', true);
			useImage = await loadSetting('useImage', true);
			autoCorrect = await loadSetting('autoCorrect', true);
			void refreshDevices();
			try {
				if (navigator.mediaDevices?.addEventListener) {
					navigator.mediaDevices.addEventListener('devicechange', () => void refreshDevices());
				}
			} catch {
				// ignore
			}
			try {
				if (cancelled) return;
				// Dynamic import keeps TF.js out of the SSR/prerender bundle and
				// code-splits it; module types still let TS track engine fully.
				const mod = await import('$lib/sibiClassifier');
				engine = new mod.SibiEngine();
				smoother = new mod.PredictionSmoother({ minConfidence: minConf, cooldownMs, holdMs: 2000 });
				// Show loading state immediately; engine mutates nested statuses.
				for (const s of engine.slots) s.status = 'loading';
				modelSlots = [...engine.slots];
				await engine.load();
				if (cancelled) return;
				modelSlots = [...engine.slots];
				const ready = engine.readyCount;
				flash(
					engine.primaryReady
						? `Model siap (${ready}/3). Tahan isyarat sampai terkunci.`
						: 'Model gagal dimuat. Ketik manual di bawah tetap bisa dipakai.'
				);
			} catch {
				if (!cancelled) flash('Backend ML tak tersedia. Ketik manual di bawah tetap bisa dipakai.');
			}
			try {
				await ensureHandLandmarker();
				if (!cancelled) visionReady = true;
			} catch {
				if (!cancelled) visionError = getVisionError() || 'Hand tracking gagal dimuat. Periksa koneksi internet.';
			}
		})();
		try {
			if ('speechSynthesis' in window) window.speechSynthesis.getVoices();
		} catch {
			// voice is optional
		}
		return () => {
			cancelled = true;
			window.removeEventListener('keydown', onKey);
			cancelAnimationFrame(raf);
			stream?.getTracks().forEach((tr) => tr.stop());
			stopSpeaking();
		};
	});

	async function restoreDefaults() {
		minConf = 0.65;
		cooldownMs = 1500;
		grayscale = false;
		normMode = 'neg-one-one';
		useBaseline = true;
		useImage = true;
		await saveSetting('minConf', minConf);
		await saveSetting('cooldownMs', cooldownMs);
		await saveSetting('grayscale', grayscale);
		await saveSetting('useBaseline', true);
		await saveSetting('useImage', true);
		smoother = new (await import('$lib/sibiClassifier')).PredictionSmoother({ minConfidence: minConf, cooldownMs, holdMs: 2000 });
		lastLm = null;
		flash('Pengaturan dipulihkan ke default');
	}

	async function onConfChange() {
		const mod = await import('$lib/sibiClassifier');
		smoother = new mod.PredictionSmoother({ minConfidence: minConf, cooldownMs, holdMs: 2000 });
		await saveSetting('minConf', minConf);
		await saveSetting('cooldownMs', cooldownMs);
		await saveSetting('grayscale', grayscale);
	}
</script>

<svelte:head>
	<title>SIBI Translator: SIBI ke Bahasa Indonesia</title>
	<meta
		name="description"
		content="Asisten komunikasi SIBI. Pengenalan alfabet SIBI real time dari sendi tangan, prediksi Bahasa Indonesia, output suara."
	/>
	<link rel="preconnect" href="https://fonts.googleapis.com" />
	<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin="anonymous" />
	<link
		href="https://fonts.googleapis.com/css2?family=Archivo+Black&family=Space+Grotesk:wght@400;500;600;700&display=swap"
		rel="stylesheet"
	/>
</svelte:head>

<!-- HEADER -->
<header class="sticky top-0 z-40 border-b-4 border-black bg-[#FFD02B]">
	<div class="mx-auto flex max-w-6xl flex-wrap items-center gap-3 px-4 py-3">
		<div class="brutal-sm flex h-11 w-11 items-center justify-center bg-white text-2xl"></div>
		<div class="min-w-0">
			<h1 class="font-display text-lg leading-none tracking-tight uppercase sm:text-2xl">
				SIBI Translator
			</h1>
			<p class="text-xs font-semibold tracking-wide uppercase sm:text-sm">
				SIBI ke Bahasa Indonesia | 3 model aktif
			</p>
		</div>
		<div class="ml-auto flex items-center gap-2">
			<span
				class="sticker rounded-full px-3 py-1 text-xs font-bold uppercase {readyCount === 3
					? 'bg-green-300'
					: primaryReady
						? 'bg-[#FFD02B]'
						: 'bg-[#FF90E8]'}"
			>
				{readyCount === 3 ? '3 model siap' : primaryReady ? `Sendi siap | ${readyCount}/3` : 'Model...'}
			</span>
			<span
				class="sticker hidden rounded-full bg-white px-3 py-1 text-xs font-bold uppercase sm:inline"
			>
				{visionReady ? 'Tracking siap' : 'Tracking...'}
			</span>
		</div>
	</div>
</header>

<main class="mx-auto max-w-6xl space-y-6 px-4 py-6">
	{#if notice}
		<div class="brutal rounded-xl bg-black px-4 py-2 text-center text-sm font-bold text-[#FFD02B]">
			{notice}
		</div>
	{/if}

<!-- CAMERA -->
<section class="brutal-lg overflow-hidden rounded-2xl bg-white">
	<div class="relative bg-[#111]">
		<video
			bind:this={videoEl}
			class="mirror aspect-video max-h-[80vh] w-full object-cover {camOn ? '' : 'hidden'}"
			playsinline
			muted
			autoplay
		></video>
		<canvas
			bind:this={overlayEl}
			class="pointer-events-none absolute inset-0 h-full w-full {camOn ? '' : 'hidden'}"
		></canvas>
		{#if camOn}
			{#if handsFound}
				<div class="absolute top-3 right-3 z-10 rounded-2xl border-4 border-black bg-[#FFD02B] px-4 py-2 text-center shadow-[6px_6px_0_#000]">
					<p class="font-display text-6xl leading-none sm:text-7xl">{jointLabel}</p>
					<p class="mt-1 text-xs font-black uppercase">
						{Math.round(jointConf * 100)}% · fusi {Math.round(fusedConf * 100)}%
					</p>
					<div class="mt-1 h-2 w-28 overflow-hidden rounded-full border-2 border-black bg-white sm:w-36">
						<div class="h-full bg-green-500" style="width:{Math.round(stability * 100)}%"></div>
					</div>
					<p class="mt-0.5 text-[10px] font-bold uppercase opacity-70">stabil {Math.round(stability * 100)}%</p>
				</div>
			{/if}
			{#if !handsFound && !noSignal}
				<div class="pointer-events-none absolute inset-0 grid place-items-center">
					<div class="grid h-[62%] w-[40%] place-items-center rounded-2xl border-4 border-dashed border-[#FFD02B]/70">
						<span class="rounded-md bg-black/70 px-2 py-0.5 text-[11px] font-bold text-[#FFD02B]">tangan di sini</span>
					</div>
				</div>
			{/if}
			{#if handHint && !noSignal}
				<div class="absolute inset-x-0 top-0 border-b-4 border-black bg-[#FFD02B] px-4 py-1.5 text-center text-xs font-black uppercase">
					{handHint}
				</div>
			{/if}
			{#if holdLabel}
				<div class="absolute inset-x-0 top-0 z-20 border-b-4 border-black bg-white px-4 py-1.5">
					<div class="flex items-center gap-2 text-xs font-black uppercase">
						<span>{holdLabel === 'fist' ? 'Hapus' : 'Kunci kata'}</span>
						<div class="h-2 flex-1 overflow-hidden rounded-full border-2 border-black bg-[#FFF6D6]">
							<div class="h-full bg-[#FF90E8]" style="width:{Math.round(holdProgress * 100)}%"></div>
						</div>
						<span>{Math.round(holdProgress * 100)}%</span>
					</div>
				</div>
			{/if}
			{#if noSignal}
				<div class="absolute inset-x-0 top-0 border-b-4 border-black bg-[#FF90E8] px-4 py-2 text-center">
					<p class="text-xs font-black uppercase">Stream kosong: tidak ada gambar</p>
					<p class="text-[11px] font-bold">Kamera mungkin dipakai aplikasi lain atau salah perangkat.</p>
				</div>
			{/if}
			<div class="absolute bottom-12 left-2 right-2 z-10 flex flex-wrap gap-1.5">
				{#if words.length}
					{#each words as w, i}
						<span class="sticker flex items-center gap-1 rounded-lg bg-white/95 px-2 py-1 text-xs font-black uppercase {dragFrom === i ? 'opacity-50' : ''}">
							<button onclick={() => moveWord(i, -1)} disabled={i === 0} class="rounded px-1 text-xs font-black disabled:opacity-30" title="Geser kiri">L</button>
							<span>{w}</span>
							<button onclick={() => removeWord(i)} class="rounded px-1 text-xs font-black opacity-60 hover:opacity-100" title="Hapus">x</button>
							<button onclick={() => moveWord(i, 1)} disabled={i === words.length - 1} class="rounded px-1 text-xs font-black disabled:opacity-30" title="Geser kanan">R</button>
						</span>
					{/each}
				{/if}
				{#if prefix}
					<span class="sticker rounded-lg bg-[#FFD02B] px-2 py-1 text-xs font-black uppercase">{prefix}_</span>
				{/if}
			</div>
			<div class="absolute bottom-2 left-2 flex flex-wrap gap-1.5 text-[11px] font-bold">
				<span class="sticker rounded-md bg-white px-2 py-0.5">FPS {fps}</span>
				<span class="sticker rounded-md bg-white px-2 py-0.5">sendi {inferMs} ms</span>
				<span class="sticker rounded-md px-2 py-0.5 {handsFound ? 'bg-green-300' : 'bg-[#FF90E8]'}">
					{handsFound ? 'tangan OK' : 'cari tangan...'}
				</span>
				{#if gesture !== 'none' && GESTURE_META[gesture]}
					<span class="sticker rounded-md bg-[#FFD02B] px-2 py-0.5">
						{GESTURE_META[gesture].icon} {GESTURE_META[gesture].label}
					</span>
				{/if}
			</div>
		{:else}
			<div class="grid place-items-center px-6 py-16 text-center text-white">
				<p class="text-5xl">Kamera</p>
				<p class="font-display mt-3 text-xl uppercase">Kamera mati</p>
				<p class="mx-auto mt-1 max-w-md text-sm opacity-80">
					Nyalakan kamera, lalu peragakan isyarat alfabet SIBI statis. Tahan tiap isyarat sampai bilah stabil penuh.
				</p>
				{#if devices.length > 1}
					<label class="mx-auto mt-3 flex max-w-md items-center gap-2 text-xs font-bold">
						<span class="shrink-0 uppercase">Kamera:</span>
						<select
							value={deviceId}
							onchange={(e) => void switchCamera((e.target as HTMLSelectElement).value)}
							class="min-w-0 flex-1 rounded-lg border-2 border-white bg-black px-2 py-1 text-white"
						>
							{#each devices as d}
								<option value={d.id}>{d.label}</option>
							{/each}
						</select>
					</label>
				{/if}
				{#if camError}
					<p class="mx-auto mt-3 max-w-md rounded-lg border-2 border-red-400 bg-red-950 px-3 py-2 text-xs font-bold text-red-200">
						{camError}
					</p>
				{/if}
				{#if visionError}
					<div class="mx-auto mt-3 max-w-md rounded-lg border-2 border-yellow-400 bg-yellow-950 px-3 py-2 text-xs font-bold text-yellow-200">
						<p>{visionError}</p>
						<button onclick={() => void retryVision()} class="brutal-btn mt-2 rounded-lg bg-[#FFD02B] px-3 py-1 text-xs font-black uppercase text-black">
							Coba tracking lagi
						</button>
					</div>
				{/if}
			</div>
		{/if}
	</div>
	<div class="flex items-center gap-2 border-t-4 border-black bg-white px-4 py-2">
		{#if !camOn}
			<button class="brutal-btn rounded-lg bg-black px-5 py-2 text-sm font-bold text-white" onclick={startCamera}>
				Nyalakan kamera
			</button>
		{:else}
			<button class="brutal-btn rounded-lg bg-white px-5 py-2 text-sm font-bold" onclick={stopCamera}>
				Matikan
			</button>
		{/if}
		{#if camOn && devices.length > 1}
			<select
				value={deviceId}
				onchange={(e) => void switchCamera((e.target as HTMLSelectElement).value)}
				class="min-w-0 flex-1 rounded-lg border-2 border-black bg-[#FFF6D6] px-2 py-1 text-xs font-bold"
			>
				{#each devices as d}
					<option value={d.id}>{d.label}</option>
				{/each}
			</select>
		{/if}
	</div>
</section>

<!-- OUTPUT -->
<section class="brutal-lg overflow-hidden rounded-2xl bg-white">
	<div class="border-b-4 border-black bg-black px-4 py-3">
		<h2 class="font-display text-base uppercase text-[#FFD02B] sm:text-lg">Kalimat Indonesia</h2>
	</div>
	<div class="space-y-4 p-4">
		<div class="brutal-sm rounded-xl bg-[#FFD02B] p-3">
			<p class="text-[11px] font-bold tracking-widest uppercase opacity-70">Hasil kalimat</p>
			<p class="font-display min-h-[2.5rem] text-xl leading-snug sm:text-2xl">
				{sentence || 'Belum ada. Isyarat dulu di kamera, atau ketik di bawah.'}
			</p>
		</div>
		<div>
			<p class="mb-1.5 text-[11px] font-bold tracking-widest uppercase opacity-60">Ketik manual</p>
			<input
				value={prefix}
				oninput={onManual}
				placeholder="Ketik huruf di sini (A-Z)"
				maxlength={24}
				autocomplete="off"
				class="brutal-sm w-full rounded-xl bg-white px-3 py-2 text-lg font-black tracking-widest uppercase placeholder:text-sm placeholder:font-semibold placeholder:normal-case placeholder:opacity-40"
			/>
			<p class="mt-1 text-[11px] font-semibold opacity-60">
				Keyboard: A-Z tambah huruf, Backspace hapus, Spasi kunci kata, Enter ucapkan.
			</p>
		</div>
		<div>
			<p class="mb-2 text-[11px] font-bold tracking-widest uppercase opacity-60">
				Prediksi: {suggestion.mode} (klik untuk pilih)
			</p>
			<div class="grid grid-cols-3 gap-2">
				{#each suggestion.suggestions as s, i}
					<button
						onclick={() => applySuggestion(s)}
						class="brutal-btn rounded-xl bg-white px-2 py-2.5 text-sm font-black uppercase"
					>
						<span class="mr-1 rounded bg-black px-1.5 py-0.5 text-[11px] text-[#FFD02B]">{i + 1}</span>{s.word}
						<span class="block text-[10px] font-semibold normal-case opacity-60">{s.why}</span>
					</button>
				{/each}
			</div>
		</div>
		<div class="grid grid-cols-4 gap-2">
			<button onclick={commitSpace} class="brutal-btn rounded-xl bg-white px-1 py-2 text-xs font-black uppercase">
				Spasi
			</button>
			<button onclick={deleteLast} class="brutal-btn rounded-xl bg-[#FF90E8] px-1 py-2 text-xs font-black uppercase">
				Hapus
			</button>
			<button onclick={speak} class="brutal-btn rounded-xl bg-black px-1 py-2 text-xs font-black uppercase text-[#FFD02B]">
				Ucapkan
			</button>
			<button onclick={() => void saveCurrent()} class="brutal-btn rounded-xl bg-[#7DF9FF] px-1 py-2 text-xs font-black uppercase">
				Simpan
			</button>
		</div>
		<button onclick={clearAll} class="brutal-btn w-full rounded-xl bg-white px-3 py-2 text-xs font-bold uppercase">
			Bersihkan kalimat
		</button>
	</div>
</section>
	<!-- GESTURES -->
	<section class="brutal-lg rounded-2xl bg-white p-4">
		<h2 class="font-display text-base uppercase sm:text-lg">Kontrol kamera <span class="text-xs font-bold normal-case opacity-60">(pengatur tombol, bukan isyarat SIBI)</span></h2>
		<div class="mt-3 grid grid-cols-3 gap-2">
			{#each Object.entries(GESTURE_META) as [key, m]}
				<div class="brutal-sm rounded-xl p-2 text-center text-xs font-bold {gesture === key ? 'bg-green-300' : 'bg-[#FFF6D6]'}">
					<p class="text-2xl">{m.icon}</p>
					<p class="uppercase">{m.label}</p>
					<p class="font-semibold normal-case opacity-60">{m.fn}</p>
				</div>
			{/each}
		</div>
		<p class="mt-2 text-xs font-semibold opacity-70">
			Kartu yang menyala berarti gestur itu sedang terdeteksi. Huruf tidak masuk saat fist
			atau peace ditahan, supaya aliran huruf tetap bersih.
		</p>
	</section>

	<!-- HISTORY -->
	<section class="brutal-lg rounded-2xl bg-white p-4">
		<div class="flex items-center gap-3">
			<h2 class="font-display text-base uppercase sm:text-lg">Riwayat (tersimpan lokal)</h2>
			{#if history.length}
				<button onclick={() => void clearHistory().then(() => (history = []))} class="brutal-btn ml-auto rounded-lg bg-white px-3 py-1 text-xs font-black uppercase">
					Hapus semua
				</button>
			{/if}
		</div>
		{#if history.length === 0}
			<p class="mt-2 text-sm font-semibold opacity-60">Belum ada kalimat tersimpan. Ucapkan dulu, lalu Simpan.</p>
		{:else}
			<ul class="mt-3 grid gap-2 sm:grid-cols-2">
				{#each history as h}
					<li class="brutal-sm flex items-center gap-2 rounded-xl bg-[#FFF6D6] px-3 py-2">
						<button onclick={() => speakIndonesian(h.text)} class="min-w-0 flex-1 truncate text-left text-sm font-bold" title={h.text}>
							{h.text}
						</button>
						<span class="shrink-0 text-[10px] font-bold opacity-50">{new Date(h.createdAt).toLocaleString('id-ID')}</span>
						<button onclick={() => void deleteHistoryEntry(h.id).then((n: HistoryEntry[]) => (history = n))} class="brutal-btn shrink-0 rounded-md bg-white px-2 py-0.5 text-xs font-black" aria-label="hapus">x</button>
					</li>
				{/each}
			</ul>
		{/if}
	</section>

	<!-- MODEL STATUS -->
	<section class="brutal-lg rounded-2xl bg-white p-4">
		<div class="flex flex-wrap items-center gap-3">
			<h2 class="font-display text-base uppercase sm:text-lg">Model ({readyCount}/3 siap)</h2>
			<button onclick={() => void enableAllModels()} class="brutal-btn ml-auto rounded-lg bg-black px-3 py-1 text-xs font-black uppercase text-[#FFD02B]">
				Nyalakan semua
			</button>
		</div>
		{#if visionError}
			<div class="mt-2 rounded-xl border-2 border-yellow-500 bg-yellow-100 p-3 text-xs font-bold">
				<p>Tracking: {visionError}</p>
				<button onclick={() => void retryVision()} class="brutal-btn mt-2 rounded-lg bg-[#FFD02B] px-3 py-1 text-xs font-black uppercase">
					Coba tracking lagi
				</button>
			</div>
		{/if}
		<div class="mt-2 grid gap-2 sm:grid-cols-3">
			{#each modelSlots as s}
				<div class="brutal-sm rounded-xl p-3 text-xs font-bold {s.status === 'ready' ? 'bg-green-300' : s.status === 'loading' ? 'bg-[#FFD02B]' : s.status === 'error' ? 'bg-red-200' : 'bg-[#FFF6D6]'}">
					<p class="font-display text-sm uppercase">{s.name} | val {s.valAcc}</p>
					<p class="opacity-70">{s.credit}</p>
					<p class="mt-1 uppercase">
						{s.status === 'ready' ? 'Siap' : s.status === 'loading' ? 'Memuat...' : s.status === 'error' ? 'Gagal' : 'Mati'}
					</p>
					{#if s.status === 'error'}
						<p class="mt-1 font-semibold normal-case opacity-80">{s.lastError ?? 'Gagal dimuat.'}</p>
						<button onclick={() => void retryModel(s.key)} class="brutal-btn mt-2 rounded-lg bg-black px-3 py-1 text-xs font-black uppercase text-white">
							Coba lagi
						</button>
					{/if}
				</div>
			{:else}
				<p class="text-xs font-bold sm:col-span-3">Memuat semua model...</p>
			{/each}
		</div>
	</section>

	<!-- RESEARCH / DATASET -->
	<section class="brutal-lg rounded-2xl bg-black p-4 text-white sm:p-6">
		<h2 class="font-display text-base uppercase text-[#FFD02B] sm:text-lg">Data, model, dan angka evaluasi</h2>
		<div class="mt-3 grid gap-3 text-sm sm:grid-cols-3">
			<div class="rounded-xl border-2 border-[#FFD02B] bg-[#1c1c1c] p-3">
				<p class="font-black uppercase text-[#FFD02B]">Data latih terbuka</p>
				<p class="mt-1 font-semibold opacity-90">
					272 sampel sendi (24 huruf statis) dari
					<code>AJustiago/SIBI-Recognition</code> (lisensi MIT, kamus SIBI resmi).
					Setara alfabet dataset Kaggle <code>alvinbintang/sibi-dataset</code> yang butuh token login.
					Punya <code>kaggle.json</code>? Lihat perintah latih ulang di bawah.
				</p>
				<a class="mt-2 inline-block rounded-md bg-[#FFD02B] px-3 py-1 text-xs font-black text-black" href="https://www.kaggle.com/datasets/alvinbintang/sibi-dataset" target="_blank" rel="noreferrer">
					Buka dataset Kaggle
				</a>
			</div>
			<div class="rounded-xl border-2 border-[#FFD02B] bg-[#1c1c1c] p-3">
				<p class="font-black uppercase text-[#FFD02B]">Model sendi v2 di browser</p>
				<ul class="mt-1 list-disc pl-5 font-semibold opacity-90">
					<li>MLP kanonis 63 fitur: <b>val 90,3%</b></li>
					<li>Tangan lain (mirror): tetap 90,3%</li>
					<li>Kamera miring +/-30deg: tetap 90,3%</li>
					<li>Jarak dekat/jauh: tetap 90,3%</li>
					<li>Simulasi orang lain: 86-90%</li>
				</ul>
				<p class="mt-1 font-semibold opacity-90">
					Bingkai ragu 50/50 (mis. M vs N) otomatis ditolak. J dan Z
					dikecualikan karena butuh gerakan.
				</p>
			</div>
			<div class="rounded-xl border-2 border-[#FFD02B] bg-[#1c1c1c] p-3">
				<p class="font-black uppercase text-[#FFD02B]">Yang diukur</p>
				<ul class="mt-1 list-disc pl-5 font-semibold opacity-90">
					<li>Akurasi, presisi, recall, F1, confusion matrix (skrip latih)</li>
					<li>Ketahanan: tangan, sudut, jarak, simulasi orang (probe di metrics)</li>
					<li>FPS, latensi, stabilitas prediksi (live di UI)</li>
				</ul>
			</div>
		</div>
		<div class="mt-3 rounded-xl border-2 border-dashed border-[#FFD02B] p-3 text-xs font-semibold opacity-90">
			Reproduksi: <code>python python/retrain_joint.py</code> (+/-3 mnt CPU), lalu
			<code>tensorflowjs_converter</code> ke <code>static/models/sibi-joint</code>.
			Latih ulang di data Kaggle: <code>python python/train_sibi.py --data ./SIBI</code>
			(CNN citra A-Z). Rencana lanjut: dataset kata dinamis dan arah sebaliknya
			Indonesia ke SIBI.
		</div>
	</section>

	<footer class="pb-8 text-center text-xs font-bold uppercase opacity-60">
		SIBI Translator | SvelteKit + MediaPipe + TF.js + Web Speech API
	</footer>
</main>
