"use client";

import { useEffect, useState } from "react";
import { track } from "@vercel/analytics";
import { getPersonas } from "@/lib/api";
import { type Catalog, findMatch, loadCatalog } from "@/lib/catalog";
import { useReplay } from "@/lib/useReplay";
import type { PersonaInfo } from "@/lib/types";
import { SeasonSelector } from "@/components/SeasonSelector";
import { Bracket } from "@/components/Bracket";
import { Scoreboard, ScoreboardSkeleton } from "@/components/Scoreboard";
import { CommentaryFeed, CommentaryFeedSkeleton } from "@/components/CommentaryFeed";
import { PersonaSwitcher } from "@/components/PersonaSwitcher";
import { ReplayControls } from "@/components/ReplayControls";
import { EventBurst } from "@/components/EventBurst";
import { Skeleton } from "@/components/Skeleton";
import styles from "./page.module.css";

type Load = "loading" | "ready" | "error";

export default function Home() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [personas, setPersonas] = useState<PersonaInfo[]>([]);
  const [load, setLoad] = useState<Load>("loading");
  const [loadError, setLoadError] = useState("");
  const [seasonId, setSeasonId] = useState<string | null>(null);
  const [matchId, setMatchId] = useState<string | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    void (async () => {
      try {
        const cat = await loadCatalog(ctrl.signal);
        if (ctrl.signal.aborted) return;
        setCatalog(cat);
        const def = cat.seasons.find((s) => s.season === "2017") ?? cat.seasons.at(-1);
        setSeasonId(def?.season ?? null);
        setLoad("ready");
      } catch (e) {
        if (ctrl.signal.aborted) return;
        setLoadError(e instanceof Error ? e.message : "Failed to load the catalog");
        setLoad("error");
        return;
      }
      // Personas need the serving API; the bracket still works without it.
      try {
        const ps = await getPersonas(ctrl.signal);
        if (!ctrl.signal.aborted) setPersonas(ps);
      } catch {
        /* serving API not up yet; replay will surface its own error */
      }
    })();
    return () => ctrl.abort();
  }, []);

  const season = catalog?.seasons.find((s) => s.season === seasonId) ?? null;
  const matchBalls = catalog && matchId ? (findMatch(catalog, matchId)?.balls ?? 0) : 0;

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
        {load === "ready" && catalog && (
          <SeasonSelector seasons={catalog.seasons} active={seasonId} onSelect={setSeasonId} />
        )}
      </header>

      <main className={styles.main}>
        {load === "loading" && <LoadingShell />}
        {load === "error" && <ErrorState message={loadError} />}
        {load === "ready" && season && (
          <div className={styles.bracketPanel}>
            <Bracket season={season} active={matchId} onSelect={setMatchId} />
          </div>
        )}
        {load === "ready" &&
          (matchId ? (
            <MatchCenter
              key={matchId}
              matchId={matchId}
              personas={personas}
              totalBalls={matchBalls}
            />
          ) : (
            <SelectHint />
          ))}
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
          onPlay={() => {
            if (replay.status !== "paused") track("replay_play");
            replay.play();
          }}
          onPause={replay.pause}
          onRestart={replay.restart}
          onSpeed={replay.setSpeed}
        />
      </section>
    </div>
  );
}

function SelectHint() {
  return (
    <div className={styles.hint}>
      <p className={styles.hintTitle}>Pick a match to begin</p>
      <p className={styles.hintBody}>
        Choose a season, then any match in the bracket above. Make sure the serving API is running (
        <code className={styles.code}>make serve</code>).
      </p>
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
      <h2 className={styles.noticeTitle}>Couldn&apos;t load the match catalog</h2>
      <p className={styles.noticeBody}>
        The app reads <code className={styles.code}>/catalog.json</code>. Rebuild it with{" "}
        <code className={styles.code}>uv run python -m scripts.build_catalog</code> and reload.
      </p>
      <p className={styles.noticeDim}>{message}</p>
    </div>
  );
}
