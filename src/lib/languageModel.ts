// Indonesian word prediction: character buffer -> word candidates,
// sentence context -> next-word candidates. Tiny embedded n-gram +
// frequency model; no server, no LLM. Honest MVP layer.

export interface ScoredWord {
	word: string;
	score: number;
	why: string;
}

// Core vocabulary (frequency-ordered, general Indonesian).
const VOCAB: { w: string; f: number }[] = [
	{ w: 'AKU', f: 98 }, { w: 'SAYA', f: 90 }, { w: 'KAMU', f: 85 },
	{ w: 'DIA', f: 80 }, { w: 'KITA', f: 75 }, { w: 'KAMI', f: 60 },
	{ w: 'MAU', f: 92 }, { w: 'SUKA', f: 70 }, { w: 'AKAN', f: 78 },
	{ w: 'MAKAN', f: 88 }, { w: 'MINUM', f: 72 }, { w: 'TIDUR', f: 55 },
	{ w: 'PERGI', f: 74 }, { w: 'PULANG', f: 62 }, { w: 'DATANG', f: 58 },
	{ w: 'BELAJAR', f: 66 }, { w: 'SEKOLAH', f: 68 }, { w: 'RUMAH', f: 86 },
	{ w: 'NASI', f: 84 }, { w: 'AIR', f: 76 }, { w: 'ROTI', f: 48 },
	{ w: 'IBU', f: 82 }, { w: 'AYAH', f: 80 }, { w: 'TEMAN', f: 70 },
	{ w: 'GURU', f: 60 }, { w: 'DOKTER', f: 45 }, { w: 'TOLONG', f: 77 },
	{ w: 'TERIMA', f: 50 }, { w: 'KASIH', f: 64 }, { w: 'MAAF', f: 63 },
	{ w: 'YA', f: 88 }, { w: 'TIDAK', f: 90 }, { w: 'BISA', f: 79 },
	{ w: 'SEDANG', f: 65 }, { w: 'SUDAH', f: 73 }, { w: 'BELUM', f: 71 },
	{ w: 'SEDANG', f: 40 }, { w: 'HARI', f: 67 }, { w: 'INI', f: 75 },
	{ w: 'ITU', f: 74 }, { w: 'DI', f: 70 }, { w: 'KE', f: 68 },
	{ w: 'DARI', f: 66 }, { w: 'DAN', f: 69 }, { w: 'DENGAN', f: 61 },
	{ w: 'APA', f: 72 }, { w: 'KABAR', f: 55 }, { w: 'BAIK', f: 73 },
	{ w: 'SEHAT', f: 57 }, { w: 'SENANG', f: 62 }, { w: 'LAPAR', f: 52 },
	{ w: 'HAUS', f: 50 }, { w: 'SAKIT', f: 54 }, { w: 'LELAH', f: 44 },
	{ w: 'CEPAT', f: 46 }, { w: 'LAMBAT', f: 38 }, { w: 'BESOK', f: 56 },
	{ w: 'SEKARANG', f: 69 }, { w: 'NANTI', f: 60 }, { w: 'DULU', f: 58 },
	{ w: 'LAGI', f: 63 }, { w: 'JUGA', f: 59 }, { w: 'SANGAT', f: 61 },
	{ w: 'INGIN', f: 67 }, { w: 'BUTUH', f: 64 }, { w: 'PUNYA', f: 65 },
	{ w: 'ADA', f: 78 }, { w: 'TIDAK ADA', f: 40 }, { w: 'HALO', f: 66 },
	{ w: 'PAGI', f: 60 }, { w: 'SIANG', f: 52 }, { w: 'SORE', f: 54 },
	{ w: 'MALAM', f: 58 }
];

// Bigram priors: previous word -> likely continuations.
const BIGRAMS: Record<string, string[]> = {
	AKU: ['MAU', 'SUKA', 'AKAN', 'SEDANG', 'SUDAH', 'INGIN'],
	SAYA: ['MAU', 'AKAN', 'SEDANG', 'INGIN', 'SUKA'],
	KAMU: ['MAU', 'SEDANG', 'SUDAH', 'BAIK', 'DI'],
	DIA: ['SEDANG', 'SUDAH', 'AKAN', 'PERGI', 'SAKIT'],
	KITA: ['MAKAN', 'PERGI', 'BELAJAR', 'PULANG'],
	MAU: ['MAKAN', 'MINUM', 'PERGI', 'TIDUR', 'BELAJAR', 'PULANG'],
	SUKA: ['MAKAN', 'MINUM', 'BELAJAR', 'BERMAIN'],
	AKAN: ['PERGI', 'DATANG', 'MAKAN', 'BELAJAR', 'TIDUR'],
	INGIN: ['MAKAN', 'MINUM', 'PERGI', 'TIDUR'],
	MAKAN: ['NASI', 'ROTI', 'DI', 'DULU', 'SEKARANG'],
	MINUM: ['AIR', 'DULU', 'SEKARANG'],
	PERGI: ['KE', 'SEKARANG', 'BESOK', 'DULU'],
	PULANG: ['KE', 'SEKARANG', 'NANTI'],
	KE: ['RUMAH', 'SEKOLAH'],
	DI: ['RUMAH', 'SEKOLAH'],
	HALO: ['APA', 'KAMU'],
	APA: ['KABAR'],
	KABAR: ['BAIK'],
	TERIMA: ['KASIH'],
	TOLONG: ['TOLONG', 'BANTU'],
	SELAMAT: ['PAGI', 'SIANG', 'SORE', 'MALAM']
};

const FREQ = new Map(VOCAB.map((v) => [v.w, v.f]));

function freqOf(w: string): number {
	return FREQ.get(w) ?? 20;
}

/** Complete the in-progress character buffer into word candidates. */
export function completeWord(prefix: string, limit = 6): ScoredWord[] {
	const p = prefix.trim().toUpperCase();
	if (!p) return [];
	return VOCAB.filter((v) => v.w.startsWith(p) && v.w !== p)
		.map((v) => ({
			word: v.w,
			score: v.f + p.length * 4,
			why: `awalan "${p}"`
		}))
		.sort((a, b) => b.score - a.score)
		.slice(0, limit);
}

/** Predict likely next words from committed sentence context. */
export function predictNext(words: string[], limit = 3): ScoredWord[] {
	const ups = words.map((w) => w.toUpperCase()).filter(Boolean);
	const last = ups[ups.length - 1];
	const out: ScoredWord[] = [];
	const seen = new Set<string>();

	if (last && BIGRAMS[last]) {
		for (const w of BIGRAMS[last]) {
			if (seen.has(w)) continue;
			seen.add(w);
			out.push({ word: w, score: 120 + freqOf(w) / 10, why: `setelah "${last}"` });
		}
	}
	// back off to global frequent words to always fill `limit`
	const fallback = [...VOCAB].sort((a, b) => b.f - a.f);
	for (const v of fallback) {
		if (out.length >= Math.max(limit, 3)) break;
		if (seen.has(v.w) || ups.includes(v.w)) continue;
		seen.add(v.w);
		out.push({ word: v.w, score: v.f / 10, why: 'sering dipakai' });
	}
	return out.slice(0, limit);
}

/** Combined: prefix completion wins while typing, else next-word prediction. */
export function suggest(
	sentenceWords: string[],
	currentPrefix: string,
	limit = 3
): { suggestions: ScoredWord[]; mode: 'lengkapi kata' | 'kata berikutnya' } {
	const completions = completeWord(currentPrefix, limit);
	if (completions.length > 0 && currentPrefix.trim().length > 0) {
		return { suggestions: completions.slice(0, limit), mode: 'lengkapi kata' };
	}
	return { suggestions: predictNext(sentenceWords, limit), mode: 'kata berikutnya' };
}

/** Sentence-case an uppercase word list into readable Indonesian. */
export function toSentence(words: string[]): string {
	if (words.length === 0) return '';
	const s = words.join(' ').toLowerCase();
	return s.charAt(0).toUpperCase() + s.slice(1);
}
