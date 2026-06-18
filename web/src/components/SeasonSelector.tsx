import type { Season } from "@/lib/catalog";
import styles from "./SeasonSelector.module.css";

interface Props {
  seasons: Season[];
  active: string | null;
  onSelect: (season: string) => void;
}

export function SeasonSelector({ seasons, active, onSelect }: Props) {
  return (
    <div className={styles.wrap}>
      <span className={styles.label}>Season</span>
      <div className={styles.row} role="tablist" aria-label="IPL season">
        {seasons.map((s) => {
          const isActive = s.season === active;
          return (
            <button
              key={s.season}
              type="button"
              role="tab"
              aria-selected={isActive}
              className={`${styles.chip} ${isActive ? styles.active : ""}`}
              onClick={() => onSelect(s.season)}
              title={s.champion ? `Champions: ${s.champion}` : undefined}
            >
              {s.label.replace("IPL ", "")}
            </button>
          );
        })}
      </div>
    </div>
  );
}
