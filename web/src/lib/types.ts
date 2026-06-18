// Types mirroring the Phase 6 serving API (app/serve/*). A drift between these and
// the server schema surfaces here as a TypeScript error, which is the intended seam.

export interface MatchSummary {
  match_id: string;
  teams: [string, string]; // (batting-first, bowling-first)
  innings: number;
  balls: number;
}

export interface PersonaInfo {
  key: string;
  display_name: string;
  instruction: string;
}

// Mirrors app/serve/scoreboard.py::Scoreboard. All facts, sourced from the record.
export interface Scoreboard {
  match_id: string;
  ball_id: string;
  innings: number; // 1-based
  over: string; // "11.6"
  batting_team: string;
  bowling_team: string;
  score: string; // "162/4"
  runs: number;
  wickets: number;
  striker: string;
  striker_runs: number;
  striker_balls: number;
  bowler: string;
  bowler_figures: string; // "1/24"
  current_run_rate: number | null;
  required_run_rate: number | null;
  target: number | null;
  runs_required: number | null;
  balls_left: number;
  last_deliveries: string[];
  phase: string;
  event: string; // "FOUR off the bat"
}

// Mirrors app/serve/engine.py::CommentaryResult.
export interface CommentaryResult {
  line: string;
  faithful: boolean;
  source: "model" | "fallback";
  attempts: number;
  reasons: string[];
}

// One delivery as the UI accumulates it across SSE events.
export interface BallEntry {
  ballId: string;
  scoreboard: Scoreboard;
  persona: string;
  text: string; // streamed commentary so far
  result: CommentaryResult | null; // set on the `ball` event
}
