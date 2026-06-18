import { Fragment } from "react";
import type { CatalogMatch, Season } from "@/lib/catalog";
import { roundLabel } from "@/lib/catalog";
import { teamCode } from "@/lib/teams";
import { TeamBadge } from "./TeamBadge";
import styles from "./Bracket.module.css";

interface Props {
  season: Season;
  active: string | null;
  onSelect: (id: string) => void;
}

function Side({ team, won }: { team: string; won: boolean }) {
  return (
    <span className={`${styles.side} ${won ? styles.won : styles.lost}`}>
      <TeamBadge team={team} size={24} />
      <span className={styles.code}>{teamCode(team)}</span>
      {won && (
        <span className={styles.tick} aria-label="winner">
          ✓
        </span>
      )}
    </span>
  );
}

function LeagueRow({
  m,
  active,
  onSelect,
}: {
  m: CatalogMatch;
  active: boolean;
  onSelect: (id: string) => void;
}) {
  const [a, b] = m.teams;
  return (
    <button
      type="button"
      className={`${styles.row} ${active ? styles.activeRow : ""}`}
      aria-pressed={active}
      onClick={() => onSelect(m.id)}
    >
      <span className={`${styles.no} tnum`}>{m.matchNo ?? "·"}</span>
      <span className={styles.rowTeams}>
        <span className={m.winner === a ? styles.w : styles.l}>{teamCode(a)}</span>
        <span className={styles.rowVs}>v</span>
        <span className={m.winner === b ? styles.w : styles.l}>{teamCode(b)}</span>
      </span>
      <span className={`${styles.rowBalls} tnum`}>{m.balls}</span>
    </button>
  );
}

function PlayoffNode({
  m,
  active,
  onSelect,
}: {
  m: CatalogMatch;
  active: boolean;
  onSelect: (id: string) => void;
}) {
  const [a, b] = m.teams;
  const isFinal = m.round === 3;
  return (
    <button
      type="button"
      className={`${styles.node} ${active ? styles.active : ""}`}
      aria-pressed={active}
      onClick={() => onSelect(m.id)}
    >
      <div className={styles.nodeTop}>
        <span className={styles.stage}>{m.stage}</span>
        <span className={styles.date}>{m.date}</span>
      </div>
      <div className={styles.teams}>
        <Side team={a} won={m.winner === a} />
        <span className={styles.vs}>v</span>
        <Side team={b} won={m.winner === b} />
      </div>
      <div className={styles.nodeFoot}>
        <span className={`${styles.balls} tnum`}>{m.balls} balls</span>
        <span className={isFinal ? styles.champ : styles.advance}>
          {isFinal && m.winner ? `Champions · ${teamCode(m.winner)}` : "Winner advances"}
        </span>
      </div>
    </button>
  );
}

function Chevron() {
  return (
    <div className={styles.chev} aria-hidden="true">
      <svg
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
      >
        <path d="M9 6l6 6-6 6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
}

export function Bracket({ season, active, onSelect }: Props) {
  const league = season.matches
    .filter((m) => m.round === 0)
    .sort((x, y) => (x.matchNo ?? 0) - (y.matchNo ?? 0));
  const rounds = [1, 2, 3]
    .map((r) => ({
      r,
      label: roundLabel(r, season.matches),
      matches: season.matches.filter((m) => m.round === r),
    }))
    .filter((c) => c.matches.length > 0);

  return (
    <div className={styles.bracket}>
      <header className={styles.head}>
        <h2 className={styles.title}>{season.label}</h2>
        {season.champion && <span className={styles.champBadge}>🏆 {season.champion}</span>}
        <span className={styles.sub}>{league.length} league matches · pick any to replay</span>
      </header>

      <div className={styles.flow}>
        <section className={styles.col}>
          <span className={styles.colLabel}>Group Stage</span>
          <div className={styles.group}>
            {league.map((m) => (
              <LeagueRow key={m.id} m={m} active={m.id === active} onSelect={onSelect} />
            ))}
          </div>
        </section>

        {rounds.map((c) => (
          <Fragment key={c.r}>
            <Chevron />
            <section className={styles.col}>
              <span className={styles.colLabel}>{c.label}</span>
              {c.matches.map((m) => (
                <PlayoffNode key={m.id} m={m} active={m.id === active} onSelect={onSelect} />
              ))}
            </section>
          </Fragment>
        ))}
      </div>
    </div>
  );
}
