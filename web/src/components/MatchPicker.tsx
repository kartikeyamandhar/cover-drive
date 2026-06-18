import type { MatchSummary } from "@/lib/types";
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
        return (
          <button
            key={m.match_id}
            type="button"
            role="tab"
            aria-selected={isActive}
            className={`${styles.card} ${isActive ? styles.active : ""}`}
            onClick={() => onSelect(m.match_id)}
          >
            <span className={styles.teams}>
              {m.teams[0]} <span className={styles.v}>v</span> {m.teams[1]}
            </span>
            <span className={`${styles.meta} tnum`}>{m.balls} balls</span>
          </button>
        );
      })}
    </div>
  );
}
