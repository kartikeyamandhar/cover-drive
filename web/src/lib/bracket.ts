// IPL 2017 bracket structure (factual tournament data), keyed on the demo match ids the
// serving API exposes. Teams and ball counts come from the API; this holds each match's
// stage, date, and winner plus the playoff topology, so the picker can render a real
// bracket. To chart another tournament later, add its ids here and expose them server-side.

export const TOURNAMENT = { name: "Indian Premier League", season: 2017 };

export type Stage = "league" | "Qualifier 1" | "Eliminator" | "Qualifier 2" | "Final";

export interface BracketMeta {
  stage: Stage;
  date: string;
  winner: string;
  matchNo?: number;
}

export const BRACKET_META: Record<string, BracketMeta> = {
  "1082591": { stage: "league", matchNo: 1, date: "5 Apr 2017", winner: "Sunrisers Hyderabad" },
  "1082592": { stage: "league", matchNo: 2, date: "6 Apr 2017", winner: "Rising Pune Supergiant" },
  "1082593": { stage: "league", matchNo: 3, date: "7 Apr 2017", winner: "Kolkata Knight Riders" },
  "1082594": { stage: "league", matchNo: 4, date: "8 Apr 2017", winner: "Kings XI Punjab" },
  "1082595": {
    stage: "league",
    matchNo: 5,
    date: "8 Apr 2017",
    winner: "Royal Challengers Bangalore",
  },
  "1082596": { stage: "league", matchNo: 6, date: "9 Apr 2017", winner: "Sunrisers Hyderabad" },
  "1082647": { stage: "Qualifier 1", date: "16 May 2017", winner: "Rising Pune Supergiant" },
  "1082648": { stage: "Eliminator", date: "17 May 2017", winner: "Kolkata Knight Riders" },
  "1082649": { stage: "Qualifier 2", date: "19 May 2017", winner: "Mumbai Indians" },
  "1082650": { stage: "Final", date: "21 May 2017", winner: "Mumbai Indians" },
};

// Where each playoff winner advances (the IPL playoff topology), shown as a node footer.
export const ADVANCE: Record<string, string> = {
  "Qualifier 1": "Winner → Final",
  Eliminator: "Winner → Qualifier 2",
  "Qualifier 2": "Winner → Final",
};

export function metaFor(id: string): BracketMeta | undefined {
  return BRACKET_META[id];
}

// Render the bracket only if the served set actually contains the playoffs we know about.
export function hasBracket(ids: string[]): boolean {
  return ids.some((id) => {
    const stage = BRACKET_META[id]?.stage;
    return stage !== undefined && stage !== "league";
  });
}
