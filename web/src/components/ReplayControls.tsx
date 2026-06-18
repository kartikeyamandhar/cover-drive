import type { ReplayStatus } from "@/lib/useReplay";
import styles from "./ReplayControls.module.css";

interface Props {
  status: ReplayStatus;
  ballNumber: number; // 1-based, 0 if none yet
  totalBalls: number;
  onPlay: () => void;
  onPause: () => void;
  onRestart: () => void;
}

const STATUS_LABEL: Record<ReplayStatus, string> = {
  idle: "Ready",
  connecting: "Connecting",
  playing: "Live",
  paused: "Paused",
  done: "Innings done",
  error: "Disconnected",
};

export function ReplayControls({
  status,
  ballNumber,
  totalBalls,
  onPlay,
  onPause,
  onRestart,
}: Props) {
  const isPlaying = status === "playing" || status === "connecting";
  const progress = totalBalls > 0 ? Math.min(ballNumber / totalBalls, 1) : 0;

  return (
    <div className={styles.bar}>
      <button
        type="button"
        className={styles.play}
        onClick={isPlaying ? onPause : onPlay}
        aria-label={isPlaying ? "Pause replay" : "Play replay"}
      >
        {isPlaying ? <PauseIcon /> : <PlayIcon />}
      </button>

      <button type="button" className={styles.restart} onClick={onRestart} aria-label="Restart">
        <RestartIcon />
      </button>

      <div className={styles.progressWrap}>
        <div className={styles.meta}>
          <span className={styles.status} data-status={status}>
            <span className={styles.dot} />
            {STATUS_LABEL[status]}
          </span>
          <span className={`${styles.count} tnum`}>
            {ballNumber}
            {totalBalls > 0 ? ` / ${totalBalls}` : ""} balls
          </span>
        </div>
        <div className={styles.track} role="progressbar" aria-valuenow={Math.round(progress * 100)}>
          <div className={styles.fill} style={{ width: `${progress * 100}%` }} />
        </div>
      </div>
    </div>
  );
}

function PlayIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M8 5v14l11-7z" />
    </svg>
  );
}

function PauseIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M6 5h4v14H6zM14 5h4v14h-4z" />
    </svg>
  );
}

function RestartIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      aria-hidden="true"
    >
      <path d="M3 12a9 9 0 1 0 3-6.7" />
      <path d="M3 4v5h5" />
    </svg>
  );
}
