import type { Scoreboard as ScoreboardData } from "@/lib/types";
import { Skeleton } from "./Skeleton";
import styles from "./Scoreboard.module.css";

// The deterministic facts panel. Every value here comes straight from the structured
// record (the model never touches it), which is the whole point of the project.

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className={styles.stat}>
      <span className={styles.statLabel}>{label}</span>
      <span className={`${styles.statValue} tnum ${accent ? styles.accent : ""}`}>{value}</span>
    </div>
  );
}

export function ScoreboardSkeleton() {
  return (
    <section className={styles.board} aria-busy="true">
      <div className={styles.head}>
        <Skeleton width="180px" height="0.8em" />
      </div>
      <div className={styles.scoreRow}>
        <Skeleton width="160px" height="3rem" radius="var(--r2)" />
      </div>
      <div className={styles.grid}>
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} height="2.4em" radius="var(--r2)" />
        ))}
      </div>
    </section>
  );
}

const PHASE_LABEL: Record<string, string> = {
  powerplay: "Powerplay",
  middle: "Middle overs",
  death: "Death overs",
};

export function Scoreboard({ data }: { data: ScoreboardData }) {
  const chasing = data.runs_required != null && data.target != null;
  return (
    <section className={styles.board} aria-label="Match scoreboard">
      <header className={styles.head}>
        <div className={styles.teams}>
          <span className={styles.batting}>{data.batting_team}</span>
          <span className={styles.vs}>v {data.bowling_team}</span>
        </div>
        <span className={`${styles.phase} tnum`}>
          {PHASE_LABEL[data.phase] ?? data.phase} · Inns {data.innings}
        </span>
      </header>

      <div className={styles.scoreRow}>
        <span className={`${styles.score} tnum`}>{data.score}</span>
        <span className={`${styles.over} tnum`}>
          <span className={styles.overNum}>{data.over}</span>
          <span className={styles.overLabel}>OVERS</span>
        </span>
      </div>

      {chasing && (
        <div className={styles.equation}>
          <span className="tnum">{data.runs_required}</span> needed off{" "}
          <span className="tnum">{data.balls_left}</span>
        </div>
      )}

      <div className={styles.grid}>
        <Stat label="Striker" value={`${data.striker_runs} (${data.striker_balls})`} accent />
        <Stat label="On strike" value={data.striker} />
        <Stat label="Bowler" value={data.bowler} />
        <Stat label="Figures" value={data.bowler_figures} />
        <Stat
          label="Run rate"
          value={data.current_run_rate != null ? data.current_run_rate.toFixed(2) : "—"}
        />
        <Stat
          label="Req. rate"
          value={data.required_run_rate != null ? data.required_run_rate.toFixed(2) : "—"}
        />
      </div>

      <div className={styles.lastRow}>
        <span className={styles.lastLabel}>This over</span>
        <div className={styles.balls}>
          {data.last_deliveries.length === 0 && <span className={styles.emptyBalls}>—</span>}
          {data.last_deliveries.map((d, i) => (
            <span
              key={i}
              className={`${styles.ball} tnum ${ballClass(d, styles)}`}
              title={describeBall(d)}
            >
              {d}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}

function ballClass(d: string, s: typeof styles): string {
  if (d === "W") return s.wicket;
  if (d === "6" || d === "4") return s.boundary;
  if (d === "0") return s.dot;
  return "";
}

function describeBall(d: string): string {
  if (d === "W") return "Wicket";
  if (d === "6") return "Six";
  if (d === "4") return "Four";
  if (d === "0") return "Dot ball";
  return `${d} run${d === "1" ? "" : "s"}`;
}
