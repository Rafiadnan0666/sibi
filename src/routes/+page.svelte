<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { SIBI_LABELS_24 } from '$lib/sibiLabels';
	import { suggest, toSentence, type ScoredWord } from '$lib/languageModel';
	import { cropROI, preprocessToTensor, drawPreview } from '$lib/preprocessing';
	import {
		landmarksToFeatures,
		rawPixelFeatures,
		fingerStates,
		drawSkeleton,
		drawFeatSpark,
		type FingerState
	} from '$lib/jointFeatures';
	import {
		ensureHandLandmarker,
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
	import type { ModelSlot, ScoredLetter } from '$lib/sibiClassifier';

	// TF.js needs browser APIs, so it loads late. This keeps prerender safe.
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	let engine: any = null;
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	let smoother: any = null;
	let modelSlots: ModelSlot[] = $state([]);

	// Text state: signs become chars, chars become words, words become a sentence.
	let prefix: string = $state('');
	let words: string[] = $state([]);
	let recentChars: string[] = $state([]);
	let suggestion: { suggestions: ScoredWord[]; mode: 'lengkapi kata' | 'kata berikutnya' } =
		$derived(suggest(words, prefix, 3));
	let sentence: string = $derived(toSentence(prefix ? [...words, prefix] : words));
	let stability: number = $state(0);

	// Recognition state: 3 trained models plus fusion.
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
	let fingers: FingerState[] = $state([]);
	let featVec: number[] = $state([]);

	// Camera and vision state.
	let videoEl: HTMLVideoElement | null = $state(null);
	let overlayEl: HTMLCanvasElement | null = $state(null);
	let previewEl: HTMLCanvasElement | null = $state(null);
	let jointCanvas: HTMLCanvasElement | null = $state(null);
	let sparkCanvas: HTMLCanvasElement | null = $state(null);
	let stageEl: HTMLDivElement | null = $state(null);
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
	let fps: number = $state(0);
	let handsFound: boolean = $state(false);
	let cursor: { x: number; y: number } = $state({ x: 50, y: 50 });
	let hoverId: string = $state('');
	let flashId: string = $state('');
	let handHint: string = $state('');
	let lastHintAt = 0;

	// Settings.
	let minConf: number = $state(0.55);
	let cooldownMs: number = $state(1200);
	let grayscale: boolean = $state(false);
	let normMode: 'neg-one-one' | 'zero-one' = $state('neg-one-one');
	let useBaseline: boolean = $state(true);
	let useImage: boolean = $state(true);

	let history: HistoryEntry[] = $state([]);
	let notice: string = $state('');
	let lastPinchAt = 0;
	let lastGestureAt: Record<string, number> = {};
	let lastJointAt = 0;
	let lastBaseAt = 0;
	let lastImgAt = 0;
	let lastUiAt = 0;
	let lastFrameAt = 0;
	let raf = 0;

	const readyCount = $derived(modelSlots.filter((s) => s.status === 'ready').length);
	const primaryReady = $derived(modelSlots[0]?.status === 'ready');

	const GESTURE_META: Record<string, { icon: string; label: string; fn: string }> = {
		point: { icon: '☝', label: 'Point: gerakkan kursor', fn: 'Arahkan ke tombol' },
		pinch: { icon: '🤏', label: 'Pinch: pilih / klik', fn: 'Jepit jempol + telunjuk' },
		fist: { icon: '✊', label: 'Fist: hapus', fn: 'Hapus huruf atau kata terakhir' },
		peace: { icon: '✌️', label: 'Peace: spasi', fn: 'Kunci huruf jadi kata' },
		thumbsup: { icon: '👍', label: 'Thumbs up: ucapkan', fn: 'Bacakan kalimat' }
	};

	// Text ops.
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
			words = [...words, prefix.trim().toUpperCase()];
			prefix = '';
		}
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
		smoother?.reset?.();
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
		window.clearTimeout((flash as unknown as { _t?: number })._t);
		(flash as unknown as { _t?: number })._t = window.setTimeout(() => (notice = ''), 2200);
	}

	function flashButton(id: string) {
		flashId = id;
		window.setTimeout(() => {
			if (flashId === id) flashId = '';
		}, 350);
	}

	const hoverActions: Record<string, () => void> = {
		sug0: () => suggestion.suggestions[0] && applySuggestion(suggestion.suggestions[0]),
		sug1: () => suggestion.suggestions[1] && applySuggestion(suggestion.suggestions[1]),
		sug2: () => suggestion.suggestions[2] && applySuggestion(suggestion.suggestions[2]),
		space: () => commitSpace(),
		delete: () => deleteLast(),
		speak: () => speak(),
		save: () => void saveCurrent()
	};

	function triggerHover(id: string) {
		const fn = hoverActions[id];
		if (!fn) return;
		flashButton(id);
		fn();
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

		if (primary && stageEl) {
			cursor = {
				x: Math.min(98, Math.max(2, primary.indexTip.x * 100)),
				y: Math.min(96, Math.max(4, primary.indexTip.y * 100))
			};
			updateHover();
			if (primary.pinch && t - lastPinchAt > 900 && hoverId && gestureCooldown('pinch', 900)) {
				lastPinchAt = t;
				triggerHover(hoverId);
			}
		}

		if (primary) {
			if (primary.gesture === 'fist' && gestureCooldown('fist', 1400)) deleteLast();
			else if (primary.gesture === 'peace' && gestureCooldown('peace', 1400)) commitSpace();
			else if (primary.gesture === 'thumbsup' && gestureCooldown('thumbsup', 2000)) speak();
		}

		void classifyFrame(primary, t, vw, vh);
	}

	function updateHover() {
		if (!stageEl) return;
		const px = (cursor.x / 100) * stageEl.clientWidth;
		const py = (cursor.y / 100) * stageEl.clientHeight;
		const els = stageEl.querySelectorAll<HTMLElement>('[data-hover]');
		let hit = '';
		for (const el of els) {
			const r = el.getBoundingClientRect();
			const s = stageEl.getBoundingClientRect();
			const lx = r.left - s.left;
			const ly = r.top - s.top;
			if (px >= lx && px <= lx + r.width && py >= ly && py <= ly + r.height) {
				hit = el.getAttribute('data-hover') ?? '';
				break;
			}
		}
		hoverId = hit;
	}

	async function classifyFrame(primary: DecodedHand | undefined, t: number, vw: number, vh: number) {
		if (!engine || !primary || !videoEl) return;
		// UI gestures take priority so they never pollute the letter stream.
		if (primary.gesture === 'fist' || primary.gesture === 'peace') return;

		// Path 1 (every ~90ms): joints go to the main MLP.
		if (t - lastJointAt >= 90 && engine.slots[0].status === 'ready') {
			lastJointAt = t;
			try {
				const feats = landmarksToFeatures(primary.landmarks, vw, vh);
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
						fingers = fingerStates(primary.landmarks, vw, vh);
					}
					stability = smoother.progress ?? 0;
					if (fused >= minConf) {
						const committed = smoother.push(out.label, fused, t);
						if (committed) {
							commitChar(committed);
							recentChars = [...recentChars, committed].slice(-10);
							flash(`Terkunci: ${committed}`);
						}
						stability = smoother.progress ?? 0;
					}
				}
			} catch {
				// one bad frame must never kill the loop
			}
		}

		// Path 2 (~4fps): baseline replica votes too.
		if (useBaseline && t - lastBaseAt >= 250 && engine.slots[1].status === 'ready') {
			lastBaseAt = t;
			try {
				const px = rawPixelFeatures(primary.landmarks, vw, vh);
				baselineRes = await engine.predictBaseline(px);
			} catch {
				// ignore
			}
		}

		// Path 3 (~5fps): image CNN checks the ROI crop.
		if (useImage && t - lastImgAt >= 200 && engine.slots[2].status === 'ready') {
			lastImgAt = t;
			try {
				const roi = cropROI(videoEl, primary.box);
				const tensor = preprocessToTensor(roi, { size: 128, normalize: normMode, grayscale });
				if (previewEl) drawPreview(tensor, previewEl, normMode);
				imageRes = await engine.predictImage(tensor.data);
			} catch {
				// ignore
			}
		}

		if (jointCanvas && primary) {
			const ctx = jointCanvas.getContext('2d')!;
			ctx.clearRect(0, 0, jointCanvas.width, jointCanvas.height);
			drawSkeleton(ctx, primary.landmarks, jointCanvas.width, jointCanvas.height, {
				highlight: [0, 4, 8, 12, 16, 20]
			});
		}
		if (sparkCanvas && featVec.length === 63) {
			drawFeatSpark(sparkCanvas.getContext('2d')!, featVec, sparkCanvas.width, sparkCanvas.height);
		}
	}

	// Init.
	onMount(() => {
		let cancelled = false;
		(async () => {
			history = await loadHistory();
			minConf = await loadSetting('minConf', 0.55);
			cooldownMs = await loadSetting('cooldownMs', 1200);
			grayscale = await loadSetting('grayscale', false);
			void refreshDevices();
			try {
				if (navigator.mediaDevices?.addEventListener) {
					navigator.mediaDevices.addEventListener('devicechange', () => void refreshDevices());
				}
			} catch {
				// ignore
			}
			try {
				const mod = await import('$lib/sibiClassifier');
				if (cancelled) return;
				engine = new mod.SibiEngine();
				smoother = new mod.PredictionSmoother({ minConfidence: minConf, cooldownMs });
				modelSlots = engine.slots;
				await engine.load();
				if (cancelled) return;
				modelSlots = [...engine.slots];
				const n = engine.readyCount;
				flash(
					n === 3
						? '3 model siap: sendi 87 persen, baseline, citra'
						: n > 0
							? `${n} dari 3 model termuat. Sisanya pakai pad simulasi.`
							: 'Model gagal dimuat. Pakai pad alfabet di bawah.'
				);
			} catch {
				if (!cancelled) flash('Backend ML tak tersedia. Pakai pad alfabet di bawah.');
			}
			try {
				await ensureHandLandmarker();
				if (!cancelled) visionReady = true;
			} catch {
				if (!cancelled) visionError = 'Hand tracking gagal dimuat. Periksa koneksi internet.';
			}
		})();
		try {
			if ('speechSynthesis' in window) window.speechSynthesis.getVoices();
		} catch {
			// voice is optional
		}
		return () => {
			cancelled = true;
			cancelAnimationFrame(raf);
			stream?.getTracks().forEach((tr) => tr.stop());
			stopSpeaking();
		};
	});

	async function onConfChange() {
		const mod = await import('$lib/sibiClassifier');
		smoother = new mod.PredictionSmoother({ minConfidence: minConf, cooldownMs });
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
		<div class="brutal-sm flex h-11 w-11 items-center justify-center bg-white text-2xl">🤟</div>
		<div class="min-w-0">
			<h1 class="font-display text-lg leading-none tracking-tight uppercase sm:text-2xl">
				SIBI Translator
			</h1>
			<p class="text-xs font-semibold tracking-wide uppercase sm:text-sm">
				SIBI ke Bahasa Indonesia · 3 model terlatih
			</p>
		</div>
		<div class="ml-auto flex items-center gap-2">
			<span
				class="sticker rounded-full px-3 py-1 text-xs font-bold uppercase {readyCount === 3
					? 'bg-green-300'
					: readyCount > 0
						? 'bg-white'
						: 'bg-[#FF90E8]'}"
			>
				{readyCount}/3 model{primaryReady ? ' · sendi siap' : ''}
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

	<!-- HOW TO USE -->
	<div class="brutal-lg overflow-hidden rounded-2xl bg-white">
		<div class="border-b-4 border-black bg-black px-4 py-2 text-xs font-bold tracking-widest text-[#FFD02B] uppercase">
			Cara pakai: isyarat SIBI adalah inputnya
		</div>
		<div class="grid gap-2 p-4 sm:grid-cols-4">
			{#each [['1', 'Tunjukkan isyarat', 'Satu huruf SIBI statis di depan kamera'], ['2', 'Tahan sampai terkunci', 'Bilah stabil penuh berarti 1 huruf masuk'], ['3', 'Pilih prediksi atau spasi', 'Arahkan jari, jepit untuk pilih. Peace untuk spasi'], ['4', 'Ucapkan atau simpan', 'Thumbs up membacakan kalimat']] as [n, title, desc]}
				<div class="brutal-sm flex gap-3 rounded-xl bg-[#FFF6D6] p-3">
					<span class="font-display flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border-2 border-black bg-[#FFD02B] text-lg">{n}</span>
					<span>
						<span class="block text-sm font-black uppercase">{title}</span>
						<span class="block text-xs font-semibold opacity-70">{desc}</span>
					</span>
				</div>
			{/each}
		</div>
	</div>

	<!-- MODEL SLOTS -->
	<div class="grid gap-3 sm:grid-cols-3">
		{#each modelSlots as s}
			<div class="brutal-sm rounded-xl p-3 {s.status === 'ready' ? 'bg-green-300' : s.status === 'loading' ? 'bg-[#FFD02B]' : 'bg-white'}">
				<p class="text-[11px] font-bold tracking-widest uppercase opacity-60">
					{s.key === 'joint' ? 'Utama' : s.key === 'baseline' ? 'Vote kedua' : 'Cek visual'}
				</p>
				<p class="font-display text-sm uppercase">{s.name} · val {s.valAcc}</p>
				<p class="text-[11px] font-semibold opacity-70">{s.credit}</p>
				<p class="mt-1 text-[11px] font-black uppercase">
					{s.status === 'ready' ? 'Termuat' : s.status === 'loading' ? 'Memuat...' : s.status === 'error' ? 'Gagal' : 'Standby'}
				</p>
			</div>
		{:else}
			<div class="brutal-sm rounded-xl bg-white p-3 text-xs font-bold sm:col-span-3">
				Memuat 3 model TensorFlow.js...
			</div>
		{/each}
	</div>

	<!-- CAMERA: full width, big -->
	<section class="brutal-lg overflow-hidden rounded-2xl bg-white">
		<div class="flex flex-wrap items-center gap-2 border-b-4 border-black bg-[#FFD02B] px-4 py-3">
			<h2 class="font-display text-base uppercase sm:text-lg">Kamera: peragakan isyarat di sini</h2>
			<div class="ml-auto flex gap-2">
				{#if !camOn}
					<button class="brutal-btn rounded-lg bg-black px-5 py-2 text-sm font-bold text-white" onclick={startCamera}>
						Nyalakan kamera
					</button>
				{:else}
					<button class="brutal-btn rounded-lg bg-white px-5 py-2 text-sm font-bold" onclick={stopCamera}>
						Matikan
					</button>
				{/if}
			</div>
		</div>

		<div bind:this={stageEl} class="relative bg-[#111]">
			<!-- One permanent <video> element. It is never rebuilt by {#if},
			     so the stream always lands on the visible player. -->
			<video
				bind:this={videoEl}
				class="mirror aspect-video max-h-[72vh] w-full object-cover {camOn ? '' : 'hidden'}"
				playsinline
				muted
				autoplay
			></video>
			<canvas
				bind:this={overlayEl}
				class="pointer-events-none absolute inset-0 h-full w-full {camOn ? '' : 'hidden'}"
			></canvas>
			{#if camOn}
				<!-- Giant live letter badge -->
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
				<div
					class="pointer-events-none absolute z-10 flex h-8 w-8 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border-[3px] text-sm {hoverId
						? 'border-black bg-green-300'
						: 'border-[#FFD02B] bg-black text-[#FFD02B]'}"
					style="left:{cursor.x}%; top:{cursor.y}%"
				>
					☝
				</div>
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
				{#if noSignal}
					<div class="absolute inset-x-0 top-0 border-b-4 border-black bg-[#FF90E8] px-4 py-2 text-center">
						<p class="text-xs font-black uppercase">Stream kosong: tidak ada gambar</p>
						<p class="text-[11px] font-bold">Kamera mungkin dipakai aplikasi lain atau salah perangkat. Tutup aplikasi itu, atau pilih kamera lain di bawah, lalu tekan Matikan dan Nyalakan lagi.</p>
					</div>
				{/if}
				<div class="absolute bottom-2 left-2 flex flex-wrap gap-1.5 text-[11px] font-bold">
					<span class="sticker rounded-md bg-white px-2 py-0.5">FPS {fps}</span>
					<span class="sticker rounded-md bg-white px-2 py-0.5">sendi ke huruf {inferMs} ms</span>
					<span class="sticker rounded-md px-2 py-0.5 {handsFound ? 'bg-green-300' : 'bg-[#FF90E8]'}">
						{handsFound ? `${hands.length} tangan` : 'cari tangan...'}
					</span>
					{#if gesture !== 'none'}
						<span class="sticker rounded-md bg-[#FFD02B] px-2 py-0.5">
							{GESTURE_META[gesture]?.icon} {GESTURE_META[gesture]?.label ?? gesture}
						</span>
					{/if}
				</div>
			{:else}
				<div class="grid place-items-center px-6 py-16 text-center text-white">
					<p class="text-5xl">Kamera</p>
					<p class="font-display mt-3 text-xl uppercase">Kamera mati</p>
					<p class="mx-auto mt-1 max-w-md text-sm opacity-80">
						Nyalakan kamera, lalu peragakan isyarat alfabet SIBI statis (24 huruf, tanpa J dan Z
						yang butuh gerakan). Tahan tiap isyarat sampai bilah stabil penuh.
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
						<p class="mx-auto mt-3 max-w-md rounded-lg border-2 border-yellow-400 bg-yellow-950 px-3 py-2 text-xs font-bold text-yellow-200">
							{visionError}
						</p>
					{/if}
				</div>
			{/if}
		</div>
		{#if camOn && devices.length > 1}
			<div class="flex items-center gap-2 border-t-4 border-black bg-white px-4 py-2">
				<span class="text-[11px] font-black uppercase">Perangkat:</span>
				<select
					value={deviceId}
					onchange={(e) => void switchCamera((e.target as HTMLSelectElement).value)}
					class="min-w-0 flex-1 rounded-lg border-2 border-black bg-[#FFF6D6] px-2 py-1 text-xs font-bold"
				>
					{#each devices as d}
						<option value={d.id}>{d.label}</option>
					{/each}
				</select>
			</div>
		{/if}

		<!-- Readout row -->
		<div class="grid gap-3 border-t-4 border-black bg-[#FFF6D6] p-4 sm:grid-cols-3">
			<div class="brutal-sm rounded-xl bg-white p-3">
				<p class="text-[11px] font-bold tracking-widest uppercase opacity-60">Huruf dari model sendi</p>
				<p class="font-display text-3xl">
					{jointLabel}
					<span class="text-sm font-bold">({Math.round(jointConf * 100)}%)</span>
				</p>
				<p class="mt-1 text-xs font-bold">
					10 huruf terakhir: {recentChars.length ? recentChars.join(' ') : '-'}
				</p>
				<p class="mt-1 text-[11px] font-bold">
					Fusi: {Math.round(fusedConf * 100)}%
					{#if baselineRes}<span class="ml-1 rounded bg-black px-1 text-[10px] text-[#FFD02B]">B:{baselineRes.label}{baselineRes.label === jointLabel ? ' ✓' : ''}</span>{/if}
					{#if imageRes}<span class="ml-1 rounded bg-black px-1 text-[10px] text-[#7DF9FF]">C:{imageRes.label}{imageRes.label === jointLabel ? ' ✓' : ''}</span>{/if}
				</p>
			</div>
			<div class="brutal-sm rounded-xl bg-white p-3">
				<p class="text-[11px] font-bold tracking-widest uppercase opacity-60">Stabilitas isyarat</p>
				<p class="font-display text-3xl">{Math.round(stability * 100)}%</p>
				<div class="mt-1 h-2.5 overflow-hidden rounded-full border-2 border-black bg-white">
					<div class="h-full bg-green-400" style="width:{Math.round(stability * 100)}%"></div>
				</div>
				<p class="mt-1 text-[11px] font-semibold opacity-70">Tahan isyarat sampai 100%, lalu 1 huruf masuk.</p>
			</div>
			<div class="brutal-sm flex items-center gap-3 rounded-xl bg-white p-3">
				<canvas bind:this={previewEl} width="128" height="128" class="h-16 w-16 rounded-lg border-2 border-black bg-black"></canvas>
				<div>
					<p class="text-[11px] font-bold tracking-widest uppercase opacity-60">Potongan citra (cek visual)</p>
					<p class="text-xs font-bold">
						{imageRes ? `${imageRes.label} (${Math.round(imageRes.confidence * 100)}%)` : 'belum ada'}
					</p>
					<p class="mt-1 text-[11px] font-semibold opacity-70">CNN 128x128 · val 73,8%</p>
				</div>
			</div>
		</div>
	</section>

	<!-- OUTPUT + JOINTS side by side -->
	<div class="grid gap-6 lg:grid-cols-2">
		<section class="brutal-lg overflow-hidden rounded-2xl bg-white">
			<div class="border-b-4 border-black bg-black px-4 py-3">
				<h2 class="font-display text-base uppercase text-[#FFD02B] sm:text-lg">Kalimat Indonesia</h2>
			</div>
			<div class="space-y-4 p-4">
				<div class="brutal-sm rounded-xl bg-[#FFD02B] p-3">
					<p class="text-[11px] font-bold tracking-widest uppercase opacity-70">Hasil kalimat</p>
					<p class="font-display min-h-[2.5rem] text-xl leading-snug sm:text-2xl">
						{sentence || 'Belum ada. Isyarat dulu di kamera.'}
					</p>
				</div>

				<div>
					<p class="mb-1.5 text-[11px] font-bold tracking-widest uppercase opacity-60">Kata (klik untuk hapus)</p>
					<div class="flex min-h-[2.5rem] flex-wrap items-center gap-1.5">
						{#each words as w, i}
							<button
								onclick={() => removeWord(i)}
								title="Hapus kata ini"
								class="sticker rounded-lg bg-white px-2.5 py-1 text-sm font-black uppercase hover:bg-[#FF90E8]"
							>
								{w} <span class="opacity-50">×</span>
							</button>
						{:else}
							<span class="text-xs font-bold opacity-50">Belum ada kata.</span>
						{/each}
						{#if prefix}
							<span class="sticker rounded-lg bg-black px-2.5 py-1 text-sm font-black text-[#FFD02B]">
								{prefix}<span class="ml-0.5 inline-block h-4 w-2 animate-pulse bg-[#FFD02B] align-middle"></span>
							</span>
						{/if}
					</div>
				</div>

				<div>
					<p class="mb-2 text-[11px] font-bold tracking-widest uppercase opacity-60">
						Prediksi: {suggestion.mode} (arahkan jari, jepit untuk pilih)
					</p>
					<div class="grid grid-cols-3 gap-2">
						{#each suggestion.suggestions as s, i}
							{@const id = `sug${i}`}
							<button
								data-hover={id}
								onclick={() => applySuggestion(s)}
								class="brutal-btn rounded-xl px-2 py-2.5 text-sm font-black uppercase {hoverId === id || flashId === id
									? 'bg-green-300'
									: 'bg-white'}"
							>
								<span class="mr-1 rounded bg-black px-1.5 py-0.5 text-[11px] text-[#FFD02B]">{i + 1}</span>{s.word}
								<span class="block text-[10px] font-semibold normal-case opacity-60">{s.why}</span>
							</button>
						{/each}
					</div>
				</div>

				<div class="grid grid-cols-4 gap-2">
					<button data-hover="space" onclick={commitSpace} class="brutal-btn rounded-xl px-1 py-2 text-xs font-black uppercase {hoverId === 'space' || flashId === 'space' ? 'bg-green-300' : 'bg-white'}">
						Spasi
					</button>
					<button data-hover="delete" onclick={deleteLast} class="brutal-btn rounded-xl px-1 py-2 text-xs font-black uppercase {hoverId === 'delete' || flashId === 'delete' ? 'bg-green-300' : 'bg-[#FF90E8]'}">
						Hapus
					</button>
					<button data-hover="speak" onclick={speak} class="brutal-btn rounded-xl bg-black px-1 py-2 text-xs font-black uppercase text-[#FFD02B] {hoverId === 'speak' || flashId === 'speak' ? 'outline-4 outline-green-300' : ''}">
						Ucapkan
					</button>
					<button data-hover="save" onclick={() => void saveCurrent()} class="brutal-btn rounded-xl bg-[#7DF9FF] px-1 py-2 text-xs font-black uppercase {hoverId === 'save' || flashId === 'save' ? 'bg-green-300' : ''}">
						Simpan
					</button>
				</div>

				<button onclick={clearAll} class="brutal-btn w-full rounded-xl bg-white px-3 py-2 text-xs font-bold uppercase">
					Bersihkan kalimat
				</button>

				<details class="brutal-sm rounded-xl bg-[#FFF6D6] p-3 text-xs font-semibold">
					<summary class="cursor-pointer text-sm font-black uppercase">Pengaturan pengenalan</summary>
					<label class="mt-2 block">
						Keyakinan minimum: <b>{Math.round(minConf * 100)}%</b>
						<input type="range" min="0.3" max="0.9" step="0.05" bind:value={minConf} onchange={onConfChange} class="w-full accent-black" />
					</label>
					<label class="mt-1 block">
						Jeda antar huruf: <b>{cooldownMs} ms</b>
						<input type="range" min="600" max="2500" step="100" bind:value={cooldownMs} onchange={onConfChange} class="w-full accent-black" />
					</label>
					<label class="mt-1 flex items-center gap-2">
						<input type="checkbox" bind:checked={useBaseline} class="h-4 w-4 accent-black" />
						Suara vote model baseline (tambah yakin bila setuju)
					</label>
					<label class="mt-1 flex items-center gap-2">
						<input type="checkbox" bind:checked={useImage} class="h-4 w-4 accent-black" />
						Cek silang CNN citra (tambah yakin bila setuju)
					</label>
					<label class="mt-1 flex items-center gap-2">
						<input type="checkbox" bind:checked={grayscale} onchange={onConfChange} class="h-4 w-4 accent-black" />
						Uji grayscale (percobaan pengolahan citra)
					</label>
					<label class="mt-1 block">
						Normalisasi citra
						<select bind:value={normMode} onchange={onConfChange} class="ml-2 rounded-md border-2 border-black bg-white px-2 py-1 font-bold">
							<option value="neg-one-one">-1 sampai 1 (MobileNet)</option>
							<option value="zero-one">0 sampai 1</option>
						</select>
					</label>
				</details>
			</div>
		</section>

		<section class="brutal-lg overflow-hidden rounded-2xl bg-white">
			<div class="flex flex-wrap items-center gap-2 border-b-4 border-black bg-[#7DF9FF] px-4 py-3">
				<h2 class="font-display text-base uppercase sm:text-lg">Peta sendi</h2>
				<span class="sticker ml-auto rounded-full bg-white px-3 py-1 text-xs font-bold uppercase">
					21 sendi jadi 63 angka
				</span>
			</div>
			<div class="space-y-3 p-4">
				<div class="brutal-sm overflow-hidden rounded-xl bg-[#111]">
					<canvas bind:this={jointCanvas} width="560" height="300" class="h-auto w-full"></canvas>
					<p class="border-t-2 border-[#FFD02B] px-3 py-1.5 text-[11px] font-bold text-[#FFD02B]">
						Angka 0 sampai 20 adalah sendi tangan. 0 pergelangan, 4/8/12/16/20 ujung jari.
					</p>
				</div>
				<div class="grid grid-cols-5 gap-2">
					{#each fingers as f}
						<div class="brutal-sm rounded-xl p-2 text-center" style="background:{f.extended ? '#7CFC98' : '#fff'}">
							<div class="mx-auto h-3 w-3 rounded-full border-2 border-black" style="background:{f.color}"></div>
							<p class="mt-1 text-xs font-black uppercase">{f.name}</p>
							<p class="text-[11px] font-bold">{f.extended ? 'lurus' : 'tekuk'}</p>
							<div class="mt-1 h-1.5 overflow-hidden rounded-full border border-black bg-white">
								<div class="h-full bg-black" style="width:{Math.round(f.curl * 100)}%"></div>
							</div>
						</div>
					{:else}
						<p class="col-span-5 rounded-xl border-2 border-dashed border-black p-3 text-center text-xs font-bold opacity-60">
							Nyalakan kamera. Status tiap jari tampil di sini.
						</p>
					{/each}
				</div>
				<div class="brutal-sm rounded-xl bg-white p-3">
					<p class="text-[11px] font-bold tracking-widest uppercase opacity-60">Top 3 model sendi</p>
					{#each jointTop3 as t, i}
						<div class="mt-1.5 flex items-center gap-2">
							<span class="font-display w-8 text-xl">{t.label}</span>
							<div class="h-3 flex-1 overflow-hidden rounded-full border-2 border-black bg-white">
								<div class="h-full {i === 0 ? 'bg-[#FFD02B]' : 'bg-black'}" style="width:{Math.round(t.confidence * 100)}%"></div>
							</div>
							<span class="w-12 text-right text-xs font-black">{Math.round(t.confidence * 100)}%</span>
							{#if baselineRes?.label === t.label}<span class="rounded bg-black px-1 text-[10px] font-bold text-[#FFD02B]" title="baseline setuju">B</span>{/if}
							{#if imageRes?.label === t.label}<span class="rounded bg-black px-1 text-[10px] font-bold text-[#7DF9FF]" title="citra setuju">C</span>{/if}
						</div>
					{:else}
						<p class="mt-1 text-xs font-bold opacity-50">Belum ada prediksi. Tunjukkan isyarat ke kamera.</p>
					{/each}
				</div>
				<div class="brutal-sm rounded-xl bg-[#FFF6D6] p-3">
					<p class="text-[11px] font-bold tracking-widest uppercase opacity-60">Vektor fitur 63 angka</p>
					<canvas bind:this={sparkCanvas} width="560" height="64" class="mt-1 h-16 w-full rounded-lg border-2 border-black bg-white"></canvas>
				</div>
			</div>
		</section>
	</div>

	<!-- GESTURES + SIM PAD -->
	<div class="grid gap-6 lg:grid-cols-2">
		<section class="brutal-lg rounded-2xl bg-white p-4">
			<h2 class="font-display text-base uppercase sm:text-lg">Kontrol kamera <span class="text-xs font-bold normal-case opacity-60">(pengatur tombol, bukan isyarat SIBI)</span></h2>
			<div class="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3">
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

		<section class="brutal-lg rounded-2xl bg-white p-4">
			<h2 class="font-display text-base uppercase sm:text-lg">Pad alfabet (tanpa kamera)</h2>
			<p class="mt-1 text-xs font-semibold opacity-70">
				Tombol kuning: 24 huruf yang dikuasai model. Tombol pink <b>J dan Z</b> butuh gerakan
			tangan, jadi belum ada modelnya. Tombol ini hanya untuk uji alur bahasa.
			</p>
			<div class="mt-3 grid grid-cols-6 gap-1.5 sm:grid-cols-8">
				{#each SIBI_LABELS_24 as c}
					<button onclick={() => commitChar(c)} class="brutal-btn rounded-lg bg-[#FFD02B] py-1.5 text-sm font-black hover:bg-green-300">
						{c}
					</button>
				{/each}
				{#each ['J', 'Z'] as c}
					<button onclick={() => commitChar(c)} title="Isyarat gerak, simulasi manual" class="brutal-btn rounded-lg bg-[#FF90E8] py-1.5 text-sm font-black">
						{c}*
					</button>
				{/each}
			</div>
			<div class="mt-2 flex gap-2">
				<button onclick={commitSpace} class="brutal-btn flex-1 rounded-lg bg-white py-1.5 text-xs font-black uppercase">Spasi</button>
				<button onclick={deleteLast} class="brutal-btn flex-1 rounded-lg bg-[#FF90E8] py-1.5 text-xs font-black uppercase">Hapus</button>
			</div>
		</section>
	</div>

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
						<button onclick={() => void deleteHistoryEntry(h.id).then((n: HistoryEntry[]) => (history = n))} class="brutal-btn shrink-0 rounded-md bg-white px-2 py-0.5 text-xs font-black" aria-label="hapus">×</button>
					</li>
				{/each}
			</ul>
		{/if}
	</section>

	<!-- RESEARCH / DATASET -->
	<section class="brutal-lg rounded-2xl bg-black p-4 text-white sm:p-6">
		<h2 class="font-display text-base uppercase text-[#FFD02B] sm:text-lg">Data, model, dan angka evaluasi</h2>
		<div class="mt-3 grid gap-3 text-sm sm:grid-cols-3">
			<div class="rounded-xl border-2 border-[#FFD02B] bg-[#1c1c1c] p-3">
				<p class="font-black uppercase text-[#FFD02B]">Data latih terbuka</p>
				<p class="mt-1 font-semibold opacity-90">
					272 sampel sendi dan 480 foto SIBI (24 huruf statis) dari
					<code>AJustiago/SIBI-Recognition</code> (lisensi MIT, kamus SIBI resmi).
					Setara alfabet dataset Kaggle <code>alvinbintang/sibi-dataset</code> yang butuh token login.
					Kalau punya <code>kaggle.json</code>, latih ulang di data Kaggle dengan
					<code>python/train_sibi.py --data ./SIBI</code>.
				</p>
				<a class="mt-2 inline-block rounded-md bg-[#FFD02B] px-3 py-1 text-xs font-black text-black" href="https://www.kaggle.com/datasets/alvinbintang/sibi-dataset" target="_blank" rel="noreferrer">
					Buka dataset Kaggle
				</a>
			</div>
			<div class="rounded-xl border-2 border-[#FFD02B] bg-[#1c1c1c] p-3">
				<p class="font-black uppercase text-[#FFD02B]">3 model di browser</p>
				<ul class="mt-1 list-disc pl-5 font-semibold opacity-90">
					<li>MLP sendi 63 fitur: <b>val 87,1%</b></li>
					<li>Baseline Conv1D upstream: val 74,2%</li>
					<li>MobileNetV2 ROI: val 73,8%</li>
				</ul>
				<p class="mt-1 font-semibold opacity-90">
					Fusi: keyakinan sendi naik 7% untuk tiap model lain yang setuju. J dan Z
					dikecualikan karena butuh gerakan.
				</p>
			</div>
			<div class="rounded-xl border-2 border-[#FFD02B] bg-[#1c1c1c] p-3">
				<p class="font-black uppercase text-[#FFD02B]">Yang diukur</p>
				<ul class="mt-1 list-disc pl-5 font-semibold opacity-90">
					<li>Akurasi, presisi, recall, F1, confusion matrix (skrip latih)</li>
					<li>Ketahanan: latar, cahaya, jarak, user, webcam</li>
					<li>FPS, latensi, stabilitas prediksi (live di UI)</li>
					<li>Akurasi gestur UI dan salah klik</li>
				</ul>
			</div>
		</div>
		<div class="mt-3 rounded-xl border-2 border-dashed border-[#FFD02B] p-3 text-xs font-semibold opacity-90">
			Reproduksi: <code>retrain_joint.py</code> (sendi, sekitar 1 menit) dan
			<code>train_image.py</code> (citra, sekitar 15 menit CPU), lalu
			<code>tensorflowjs_converter</code> ke <code>static/models/</code>. Rencana lanjut: dataset
			kata dan kalimat dinamis (plus BiLSTM temporal) dan arah sebaliknya Indonesia ke SIBI
			(pemetaan frasa ke animasi atau video).
		</div>
	</section>

	<footer class="pb-8 text-center text-xs font-bold uppercase opacity-60">
		SIBI Translator · SvelteKit + MediaPipe + 3 model TF.js + Web Speech API
	</footer>
</main>
