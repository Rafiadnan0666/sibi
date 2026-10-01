// Browser speech synthesis (Web Speech API), Indonesian voice when available.

export function availableVoices(): SpeechSynthesisVoice[] {
	try {
		return speechSynthesis.getVoices();
	} catch {
		return [];
	}
}

export function pickIndonesianVoice(): SpeechSynthesisVoice | null {
	try {
		const voices = speechSynthesis.getVoices();
		return (
			voices.find((v) => v.lang.toLowerCase().startsWith('id')) ??
			voices.find((v) => v.lang.toLowerCase().includes('ms')) ??
			null
		);
	} catch {
		return null;
	}
}

export function speakIndonesian(text: string, opts: { rate?: number; pitch?: number } = {}) {
	try {
		const clean = text.trim();
		if (!clean) return false;
		speechSynthesis.cancel();
		const u = new SpeechSynthesisUtterance(clean);
		u.lang = 'id-ID';
		u.rate = opts.rate ?? 0.95;
		u.pitch = opts.pitch ?? 1;
		const v = pickIndonesianVoice();
		if (v) u.voice = v;
		speechSynthesis.speak(u);
		// warm voice list on some browsers
		if (speechSynthesis.getVoices().length === 0) {
			speechSynthesis.onvoiceschanged = () => {
				const vv = pickIndonesianVoice();
				if (vv) u.voice = vv;
			};
		}
		return true;
	} catch {
		return false;
	}
}

export function stopSpeaking() {
	try {
		speechSynthesis.cancel();
	} catch {
		// noop
	}
}
