"use client";

// The replay controller. Owns a single EventSource against the Phase 6 SSE endpoint and
// turns its state/token/ball/done events into rendered React state. Key correctness points:
//   - exactly one EventSource at a time; the effect cleanup always closes it (no leaks);
//   - on `done` we close explicitly, or EventSource would auto-reconnect and loop the match;
//   - pause closes the stream (the server paces it); play/persona-switch reopen with
//     `from_ball` so we resume (or re-render the current passage in a new voice) cleanly.

import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { streamUrl } from "./api";
import type { BallEntry, CommentaryResult, Scoreboard } from "./types";

export type ReplayStatus = "idle" | "connecting" | "playing" | "paused" | "done" | "error";

interface State {
  balls: BallEntry[];
  cursor: number; // absolute index of the in-flight delivery, -1 before any
  status: ReplayStatus;
  error: string | null;
}

type Action =
  | { type: "RESET" }
  | { type: "OPENING" }
  | { type: "STATE"; index: number; scoreboard: Scoreboard; persona: string }
  | { type: "TOKEN"; index: number; t: string }
  | { type: "BALL"; index: number; result: CommentaryResult }
  | { type: "DONE" }
  | { type: "PAUSE" }
  | { type: "ERROR"; message: string };

const INITIAL: State = { balls: [], cursor: -1, status: "idle", error: null };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "RESET":
      return { ...INITIAL };
    case "OPENING":
      return { ...state, status: "connecting", error: null };
    case "STATE": {
      const balls = state.balls.slice();
      balls[action.index] = {
        ballId: action.scoreboard.ball_id,
        scoreboard: action.scoreboard,
        persona: action.persona,
        text: "",
        result: null,
      };
      return { ...state, balls, cursor: action.index, status: "playing" };
    }
    case "TOKEN": {
      const entry = state.balls[action.index];
      if (!entry) return state;
      const balls = state.balls.slice();
      balls[action.index] = { ...entry, text: entry.text + action.t };
      return { ...state, balls };
    }
    case "BALL": {
      const entry = state.balls[action.index];
      if (!entry) return state;
      const balls = state.balls.slice();
      balls[action.index] = { ...entry, result: action.result };
      return { ...state, balls };
    }
    case "DONE":
      return { ...state, status: "done" };
    case "PAUSE":
      return { ...state, status: "paused" };
    case "ERROR":
      return { ...state, status: "error", error: action.message };
    default:
      return state;
  }
}

export interface ReplayController extends State {
  persona: string;
  current: BallEntry | undefined;
  play: () => void;
  pause: () => void;
  restart: () => void;
  selectPersona: (persona: string) => void;
}

export function useReplay(matchId: string | null, initialPersona: string): ReplayController {
  const [state, dispatch] = useReducer(reducer, INITIAL);
  const [persona, setPersona] = useState(initialPersona);
  const [paused, setPaused] = useState(true);
  const [epoch, setEpoch] = useState(0); // bump to force a reconnect

  const fromBallRef = useRef(0); // absolute index to (re)open the stream at
  const idxRef = useRef(-1); // absolute index of the in-flight delivery
  const finishedRef = useRef(false);

  // Reset is by key-remount (the page keys this hook's owner on matchId), so the hook
  // starts fresh per match and needs no setState-in-effect reset. It begins paused; the
  // viewer presses play.

  // The single stream lifecycle. Re-runs on match/persona/pause/epoch changes.
  useEffect(() => {
    if (!matchId || paused) return;
    dispatch({ type: "OPENING" });
    finishedRef.current = false;
    idxRef.current = fromBallRef.current - 1;
    const es = new EventSource(streamUrl(matchId, persona, fromBallRef.current));

    const on = (name: string, fn: (data: string) => void) =>
      es.addEventListener(name, (e) => fn((e as MessageEvent).data));

    on("state", (data) => {
      idxRef.current += 1;
      dispatch({
        type: "STATE",
        index: idxRef.current,
        scoreboard: JSON.parse(data) as Scoreboard,
        persona,
      });
    });
    on("token", (data) => {
      const { t } = JSON.parse(data) as { t: string };
      dispatch({ type: "TOKEN", index: idxRef.current, t });
    });
    on("ball", (data) => {
      dispatch({
        type: "BALL",
        index: idxRef.current,
        result: JSON.parse(data) as CommentaryResult,
      });
      fromBallRef.current = idxRef.current + 1; // resume after this delivery
    });
    on("done", () => {
      finishedRef.current = true;
      es.close();
      dispatch({ type: "DONE" });
    });
    es.onerror = () => {
      if (finishedRef.current) return; // normal end-of-stream close
      es.close();
      dispatch({ type: "ERROR", message: "Lost the stream. Is the serving API running?" });
    };

    return () => es.close();
  }, [matchId, persona, paused, epoch]);

  const play = useCallback(() => {
    if (state.status === "done") {
      dispatch({ type: "RESET" });
      fromBallRef.current = 0;
      idxRef.current = -1;
      setEpoch((e) => e + 1);
    }
    setPaused(false);
  }, [state.status]);

  const pause = useCallback(() => {
    setPaused(true);
    dispatch({ type: "PAUSE" });
  }, []);

  const restart = useCallback(() => {
    dispatch({ type: "RESET" });
    fromBallRef.current = 0;
    idxRef.current = -1;
    setEpoch((e) => e + 1);
    setPaused(false);
  }, []);

  const selectPersona = useCallback((next: string) => {
    // re-render the current passage onward in the new voice
    fromBallRef.current = Math.max(idxRef.current, 0);
    setPersona(next);
    setEpoch((e) => e + 1);
    setPaused(false);
  }, []);

  return {
    ...state,
    persona,
    current: state.cursor >= 0 ? state.balls[state.cursor] : undefined,
    play,
    pause,
    restart,
    selectPersona,
  };
}
