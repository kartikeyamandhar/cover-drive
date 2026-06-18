import type { MatchSummary } from "@/lib/types";
import { ADVANCE, BRACKET_META, type BracketMeta, type Stage, TOURNAMENT } from "@/lib/bracket";
import { teamCode } from "@/lib/teams";
import { TeamBadge } from "./TeamBadge";
import styles from "./Bracket.module.css";

interface Props {
  matches: MatchSummary[];
  active: string | null;
  onSelect: (matchId: string) => void;
}

function Side({ team, won }: { team: string; won: boolean }) {
  return (
    <span className={`${styles.side} ${won ? styles.won : styles.lost}`}>
      <TeamBadge team={team} size={26} />
      <span className={styles.code}>{teamCode(team)}</span>
      {won && (
        <span className={styles.tick} title="Winner" aria-label="winner">
          ✓
        </span>
      )}
    </span>
  );
}

function MatchNode({
  match,
  meta,
  label,
  active,
  onSelect,
}: {
  match: MatchSummary;
  meta: BracketMeta;
  label: string;
  active: boolean;
  onSelect: (id: string) => void;
}) {
  const [a, b] = match.teams;
  const advance =
    meta.stage === "Final" ? `Champions · ${teamCode(meta.winner)}` : ADVANCE[meta.stage];
  return (
    <button
      type="button"
      className={`${styles.node} ${active ? styles.active : ""}`}
      aria-pressed={active}
      onClick={() => onSelect(match.match_id)}
    >
      <div className={styles.nodeTop}>
        <span className={styles.stage}>{label}</span>
        <span className={styles.date}>{meta.date}</span>
      </div>
      <div className={styles.teams}>
        <Side team={a} won={meta.winner === a} />
        <span className={styles.vs}>v</span>
        <Side team={b} won={meta.winner === b} />
      </div>
      <div className={styles.nodeFoot}>
        <span className={`${styles.balls} tnum`}>{match.balls} balls</span>
        {advance && (
          <span className={meta.stage === "Final" ? styles.champ : styles.advance}>{advance}</span>
        )}
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

export function Bracket({ matches, active, onSelect }: Props) {
  const byId = new Map(matches.map((m) => [m.match_id, m]));
  const ofStage = (stage: Stage): MatchSummary | undefined => {
    const id = Object.keys(BRACKET_META).find(
      (k) => BRACKET_META[k]?.stage === stage && byId.has(k),
    );
    return id ? byId.get(id) : undefined;
  };

  const league = Object.entries(BRACKET_META)
    .filter(([id, m]) => m.stage === "league" && byId.has(id))
    .sort((x, y) => (x[1].matchNo ?? 0) - (y[1].matchNo ?? 0))
    .map(([id]) => byId.get(id)!);

  const q1 = ofStage("Qualifier 1");
  const elim = ofStage("Eliminator");
  const q2 = ofStage("Qualifier 2");
  const final = ofStage("Final");

  const node = (match: MatchSummary | undefined, label: string) =>
    match ? (
      <MatchNode
        match={match}
        meta={BRACKET_META[match.match_id]!}
        label={label}
        active={match.match_id === active}
        onSelect={onSelect}
      />
    ) : null;

  return (
    <div className={styles.bracket}>
      <header className={styles.head}>
        <h2 className={styles.title}>
          {TOURNAMENT.name} <span className={styles.season}>{TOURNAMENT.season}</span>
        </h2>
        <span className={styles.sub}>Group stage to the final · pick any match to replay</span>
      </header>

      <div className={styles.flow}>
        <section className={styles.col}>
          <span className={styles.colLabel}>Group Stage</span>
          <div className={styles.group}>
            {league.map((m) => (
              <MatchNode
                key={m.match_id}
                match={m}
                meta={BRACKET_META[m.match_id]!}
                label={`Match ${BRACKET_META[m.match_id]?.matchNo ?? ""}`}
                active={m.match_id === active}
                onSelect={onSelect}
              />
            ))}
          </div>
        </section>

        <Chevron />

        <section className={styles.col}>
          <span className={styles.colLabel}>Playoffs</span>
          {node(q1, "Qualifier 1")}
          {node(elim, "Eliminator")}
        </section>

        <Chevron />

        <section className={styles.col}>
          <span className={styles.colLabel}>Qualifier 2</span>
          {node(q2, "Qualifier 2")}
        </section>

        <Chevron />

        <section className={styles.col}>
          <span className={styles.colLabel}>Final</span>
          {node(final, "Final")}
        </section>
      </div>
    </div>
  );
}
