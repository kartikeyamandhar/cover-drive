// Thin typed client for the Phase 6 serving API. The base URL is configurable so the
// app can point at a local `make serve` or a deployed endpoint (Phase 8).

import type { MatchSummary, PersonaInfo } from "./types";

export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000").replace(
  /\/$/,
  "",
);

async function getJSON<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { signal });
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText} for ${path}`);
  }
  return (await res.json()) as T;
}

export function getMatches(signal?: AbortSignal): Promise<MatchSummary[]> {
  return getJSON<MatchSummary[]>("/matches", signal);
}

export function getMatch(matchId: string, signal?: AbortSignal): Promise<MatchSummary> {
  return getJSON<MatchSummary>(`/matches/${encodeURIComponent(matchId)}`, signal);
}

export function getPersonas(signal?: AbortSignal): Promise<PersonaInfo[]> {
  return getJSON<PersonaInfo[]>("/personas", signal);
}

// The SSE replay endpoint, consumed by an EventSource in useReplay.
export function streamUrl(matchId: string, persona: string, fromBall: number): string {
  const params = new URLSearchParams({ persona, from_ball: String(fromBall) });
  return `${API_BASE}/matches/${encodeURIComponent(matchId)}/stream?${params}`;
}
