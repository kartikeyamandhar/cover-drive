"use client";

import { useEffect, useState } from "react";
import { API_BASE, getMatches, getPersonas } from "@/lib/api";
import { useReplay } from "@/lib/useReplay";
import type { MatchSummary, PersonaInfo } from "@/lib/types";
import { MatchPicker } from "@/components/MatchPicker";
import { Bracket } from "@/components/Bracket";
import { hasBracket } from "@/lib/bracket";
import { Scoreboard, ScoreboardSkeleton } from "@/components/Scoreboard";
import { CommentaryFeed, CommentaryFeedSkeleton } from "@/components/CommentaryFeed";
import { PersonaSwitcher } from "@/components/PersonaSwitcher";
import { ReplayControls } from "@/components/ReplayControls";
import { EventBurst } from "@/components/EventBurst";
import { Skeleton } from "@/components/Skeleton";
import styles from "./page.module.css";

type Load = "loading" | "ready" | "error";

export default function Home() {
  const [matches, setMatches] = useState<MatchSummary[]>([]);
  const [personas, setPersonas] = useState<PersonaInfo[]>([]);
  const [load, setLoad] = useState<Load>("loading");
  const [loadError, setLoadError] = useState("");
  const [matchId, setMatchId] = useState<string | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    void (async () => {
      try {
        const [ms, ps] = await Promise.all([getMatches(ctrl.signal), getPersonas(ctrl.signal)]);
        setMatches(ms);
        setPersonas(ps);
        setMatchId((cur) => cur ?? ms[0]?.match_id ?? null);
        setLoad("ready");
      } catch (e) {
        if (ctrl.signal.aborted) return;
        setLoadError(e instanceof Error ? e.message : "Failed to load");
        setLoad("error");
      }
    })();
    return () => ctrl.abort();
  }, []);

  const selected = matches.find((m) => m.match_id === matchId) ?? null;

  return (
    <div className={styles.page}>
      <header className={styles.topbar}>
        <div className={styles.brand}>
          <BrandMark />
          <div>
            <h1 className={styles.title}>Cover Drive</h1>
            <p className={styles.subtitle}>Ball-by-ball voices · verified facts</p>
          </div>
        </div>
        {load === "ready" && matches.length > 0 && (
          <div className={styles.matchbar}>
            {hasBracket(matches.map((m) => m.match_id)) ? (
              <Bracket matches={matches} active={matchId} onSelect={setMatchId} />
            ) : (
              <>
                <span className={styles.matchbarLabel}>Select match</span>
                <MatchPicker matches={matches} active={matchId} onSelect={setMatchId} />
              </>
            )}
          </div>
        )}
      </header>

      <main className={styles.main}>
        {load === "loading" && <LoadingShell />}
        {load === "error" && <ErrorState message={loadError} />}
        {load === "ready" && matches.length === 0 && <EmptyState />}
        {load === "ready" && matches.length > 0 && matchId && (
          <MatchCenter
            key={matchId}
            matchId={matchId}
            personas={personas}
            totalBalls={selected?.balls ?? 0}
          />
        )}
      </main>

      <footer className={styles.footer}>
        Facts from Cricsheet ball data · voice from a fine-tuned Qwen2.5-1.5B · a contradicted line
        never ships
      </footer>
    </div>
  );
}

function BrandMark() {
  return (
    <svg
      className={styles.mark}
      width="36"
      height="36"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.5" opacity="0.9" />
      <path
        d="M9 3.6 C 6 9, 6 15, 9 20.4"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinecap="round"
      />
      <g stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" opacity="0.8">
        <path d="M7.1 7.2 h1.9 M6.7 10 h1.9 M6.7 14 h1.9 M7.1 16.8 h1.9" />
      </g>
    </svg>
  );
}

function MatchCenter({
  matchId,
  personas,
  totalBalls,
}: {
  matchId: string;
  personas: PersonaInfo[];
  totalBalls: number;
}) {
  const replay = useReplay(matchId, "broadcast");
  const live = replay.status === "playing" || replay.status === "connecting";
  return (
    <div className={styles.grid}>
      <aside className={styles.left}>
        <div className={styles.boardWrap}>
          {replay.current ? (
            <Scoreboard data={replay.current.scoreboard} />
          ) : (
            <ScoreboardSkeleton />
          )}
          {replay.current && (
            <EventBurst key={replay.current.ballId} event={replay.current.scoreboard.event} />
          )}
        </div>
        <PersonaSwitcher
          personas={personas}
          active={replay.persona}
          onSelect={replay.selectPersona}
        />
      </aside>

      <section className={styles.right}>
        <div className={styles.feedHead}>
          <h2 className={styles.feedTitle}>Commentary</h2>
          {replay.status === "error" && replay.error && (
            <span className={styles.feedError}>{replay.error}</span>
          )}
        </div>
        <div className={styles.feedScroll}>
          <CommentaryFeed balls={replay.balls} cursor={replay.cursor} live={live} />
        </div>
        <ReplayControls
          status={replay.status}
          ballNumber={replay.cursor + 1}
          totalBalls={totalBalls}
          pace={replay.pace}
          onPlay={replay.play}
          onPause={replay.pause}
          onRestart={replay.restart}
          onSpeed={replay.setSpeed}
        />
      </section>
    </div>
  );
}

function LoadingShell() {
  return (
    <div className={styles.grid}>
      <aside className={styles.left}>
        <ScoreboardSkeleton />
        <Skeleton height="9rem" radius="var(--r3)" />
      </aside>
      <section className={styles.right}>
        <div className={styles.feedHead}>
          <h2 className={styles.feedTitle}>Commentary</h2>
        </div>
        <div className={styles.feedScroll}>
          <CommentaryFeedSkeleton />
        </div>
      </section>
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <div className={styles.notice}>
      <h2 className={styles.noticeTitle}>Can&apos;t reach the serving API</h2>
      <p className={styles.noticeBody}>
        Expected it at <code className={styles.code}>{API_BASE}</code>. Start it and reload:
      </p>
      <pre className={styles.cmd}>make serve ARGS=&quot;--stub&quot;</pre>
      <p className={styles.noticeDim}>{message}</p>
    </div>
  );
}

function EmptyState() {
  return (
    <div className={styles.notice}>
      <h2 className={styles.noticeTitle}>No matches bundled</h2>
      <p className={styles.noticeBody}>
        The serving API returned no matches. Point it at a directory of processed deliveries.
      </p>
    </div>
  );
}
