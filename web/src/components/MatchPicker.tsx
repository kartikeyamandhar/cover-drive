import type { MatchSummary } from "@/lib/types";
import { teamCode } from "@/lib/teams";
import { TeamBadge } from "./TeamBadge";
import styles from "./MatchPicker.module.css";

interface Props {
  matches: MatchSummary[];
  active: string | null;
  onSelect: (matchId: string) => void;
}

export function MatchPicker({ matches, active, onSelect }: Props) {
  return (
    <div className={styles.row} role="tablist" aria-label="Bundled matches">
      {matches.map((m) => {
        const isActive = m.match_id === active;
        const [a, b] = m.teams;
        return (
          <button
            key={m.match_id}
            type="button"
            role="tab"
            aria-selected={isActive}
            className={`${styles.card} ${isActive ? styles.active : ""}`}
            onClick={() => onSelect(m.match_id)}
          >
            {isActive && <span className={styles.liveTag}>Selected</span>}
            <div className={styles.crests}>
              <TeamBadge team={a} size={34} />
              <span className={styles.vs}>v</span>
              <TeamBadge team={b} size={34} />
            </div>
            <div className={styles.matchup}>
              {teamCode(a)} <span className={styles.vsText}>vs</span> {teamCode(b)}
            </div>
            <div className={styles.names} title={`${a} v ${b}`}>
              {a} · {b}
            </div>
            <div className={`${styles.meta} tnum`}>
              T20 · {m.balls} balls · {m.innings} {m.innings === 1 ? "innings" : "innings"}
            </div>
          </button>
        );
      })}
    </div>
  );
}
