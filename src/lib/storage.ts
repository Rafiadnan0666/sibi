// Tiny IndexedDB store (settings, vocabulary, history, cached model flag)
// with localStorage fallback so the MVP never crashes on exotic browsers.

export interface HistoryEntry {
	id: string;
	text: string;
	words: string[];
	createdAt: number;
}

const DB = 'sibi-translator';
const STORE = 'kv';

function idb(): Promise<IDBDatabase> {
	return new Promise((resolve, reject) => {
		const req = indexedDB.open(DB, 1);
		req.onupgradeneeded = () => {
			if (!req.result.objectStoreNames.contains(STORE)) req.result.createObjectStore(STORE);
		};
		req.onsuccess = () => resolve(req.result);
		req.onerror = () => reject(req.error);
	});
}

async function kvGet<T>(key: string): Promise<T | null> {
	try {
		const db = await idb();
		return await new Promise((resolve) => {
			const tx = db.transaction(STORE, 'readonly');
			const rq = tx.objectStore(STORE).get(key);
			rq.onsuccess = () => resolve((rq.result as T) ?? null);
			rq.onerror = () => resolve(null);
		});
	} catch {
		try {
			const raw = localStorage.getItem(`sibi:${key}`);
			return raw ? (JSON.parse(raw) as T) : null;
		} catch {
			return null;
		}
	}
}

async function kvSet(key: string, value: unknown): Promise<void> {
	try {
		const db = await idb();
		await new Promise<void>((resolve) => {
			const tx = db.transaction(STORE, 'readwrite');
			tx.objectStore(STORE).put(value, key);
			tx.oncomplete = () => resolve();
			tx.onerror = () => resolve();
		});
	} catch {
		try {
			localStorage.setItem(`sibi:${key}`, JSON.stringify(value));
		} catch {
			// storage unavailable: ignore, app still works in-memory
		}
	}
}

export async function loadHistory(): Promise<HistoryEntry[]> {
	return (await kvGet<HistoryEntry[]>('history')) ?? [];
}

export async function saveHistoryEntry(text: string, words: string[]): Promise<HistoryEntry[]> {
	const entry: HistoryEntry = {
		id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
		text,
		words,
		createdAt: Date.now()
	};
	const prev = await loadHistory();
	const next = [entry, ...prev].slice(0, 50);
	await kvSet('history', next);
	return next;
}

export async function clearHistory(): Promise<void> {
	await kvSet('history', []);
}

export async function deleteHistoryEntry(id: string): Promise<HistoryEntry[]> {
	const prev = await loadHistory();
	const next = prev.filter((h) => h.id !== id);
	await kvSet('history', next);
	return next;
}

export async function loadSetting<T>(key: string, fallback: T): Promise<T> {
	const v = await kvGet<T>(`setting:${key}`);
	return v ?? fallback;
}

export async function saveSetting(key: string, value: unknown): Promise<void> {
	await kvSet(`setting:${key}`, value);
}
