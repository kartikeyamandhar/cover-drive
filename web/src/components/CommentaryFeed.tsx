import type { CSSProperties } from "react";
import type { BallEntry } from "@/lib/types";
import { VOICE_META, voiceHue, voiceTag } from "@/lib/personas";
import { Skeleton } from "./Skeleton";
import styles from "./CommentaryFeed.module.css";

interface Props {
  balls: BallEntry[];
  cursor: number;
  live: boolean; // a delivery is currently streaming
}

export function CommentaryFeedSkeleton() {
  return (
    <div className={styles.feed} aria-busy="true">
      {Array.from({ length: 4 }).map((_, i) => (
        <div key={i} className={styles.row}>
          <Skeleton width="74px" height="1.1em" radius="var(--r-full)" />
          <Skeleton width={`${80 - i * 8}%`} height="1.1em" />
        </div>
      ))}
    </div>
  );
}

export function CommentaryFeed({ balls, cursor, live }: Props) {
  if (cursor < 0) {
    return (
      <div className={styles.empty}>
        {live ? (
          <>
            <span className={styles.warming} aria-hidden="true" />
            <p className={styles.emptyTitle}>Waking the commentator…</p>
            <p className={styles.emptyHint}>
              The fine-tuned model is spinning up, so the first ball can take a few seconds on the
              free CPU.
            </p>
          </>
        ) : (
          <>
            <p className={styles.emptyTitle}>No commentary yet</p>
            <p className={styles.emptyHint}>Press play to start the replay.</p>
          </>
        )}
      </div>
    );
  }

  // Newest first: the in-flight delivery leads, history flows beneath it.
  const ordered: { entry: BallEntry; index: number }[] = [];
  for (let i = cursor; i >= 0; i--) {
    const entry = balls[i];
    if (entry) ordered.push({ entry, index: i });
  }

  return (
    <div className={styles.feed}>
      {ordered.map(({ entry, index }) => {
        const isLive = index === cursor && live && entry.result === null;
        const hue = voiceHue(entry.persona);
        const meta = VOICE_META[entry.persona];
        return (
          <article
            key={`${entry.ballId}-${index}-${entry.persona}`}
            className={`${styles.line} ${isLive ? styles.liveLine : ""}`}
            style={{ "--voice": hue } as CSSProperties}
            aria-live={isLive ? "polite" : "off"}
          >
            <div className={styles.lineHead}>
              <span className={styles.voice}>
                <span className={styles.voiceTag}>{voiceTag(entry.persona)}</span>
                <span className={styles.voiceName}>{meta?.label ?? entry.persona}</span>
              </span>
              <span className={`${styles.event} tnum`}>{entry.scoreboard.event}</span>
            </div>
            <p className={styles.text}>
              {entry.text}
              {isLive && <span className={styles.caret} aria-hidden="true" />}
            </p>
            {entry.result?.source === "fallback" && (
              <span
                className={styles.fallback}
                title="Model output failed the fact check; a verified line was substituted."
              >
                studio fill
              </span>
            )}
          </article>
        );
      })}
    </div>
  );
}
