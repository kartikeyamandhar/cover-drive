// IPL team identities: a short code and brand accent for each known team, used to render
// crest badges. We deliberately do NOT bundle official (trademarked) team logos; these
// colored monogram crests read as logos and are safe to ship. To use real artwork, drop
// files in web/public/teams/<CODE>.png and enable NEXT_PUBLIC_TEAM_LOGOS=1.

interface TeamId {
  code: string;
  color: string;
}

const TEAMS: Record<string, TeamId> = {
  "Chennai Super Kings": { code: "CSK", color: "#f4c430" },
  "Mumbai Indians": { code: "MI", color: "#1f7ae0" },
  "Royal Challengers Bangalore": { code: "RCB", color: "#e23744" },
  "Royal Challengers Bengaluru": { code: "RCB", color: "#e23744" },
  "Sunrisers Hyderabad": { code: "SRH", color: "#f6822d" },
  "Kolkata Knight Riders": { code: "KKR", color: "#8e5cc4" },
  "Delhi Capitals": { code: "DC", color: "#2f6fe0" },
  "Delhi Daredevils": { code: "DD", color: "#2f6fe0" },
  "Punjab Kings": { code: "PBKS", color: "#e2484d" },
  "Kings XI Punjab": { code: "KXIP", color: "#e2484d" },
  "Rajasthan Royals": { code: "RR", color: "#ea4faa" },
  "Gujarat Titans": { code: "GT", color: "#4aa3df" },
  "Lucknow Super Giants": { code: "LSG", color: "#2c9be0" },
  "Deccan Chargers": { code: "DEC", color: "#9aa7b8" },
  "Pune Warriors": { code: "PW", color: "#5ab0ff" },
  "Pune Warriors India": { code: "PW", color: "#5ab0ff" },
  "Rising Pune Supergiant": { code: "RPS", color: "#c46be0" },
  "Rising Pune Supergiants": { code: "RPS", color: "#c46be0" },
  "Gujarat Lions": { code: "GL", color: "#f4a83d" },
  "Kochi Tuskers Kerala": { code: "KTK", color: "#f6822d" },
};

const STOP = new Set(["of", "the", "and"]);

function initials(name: string): string {
  const words = name.split(/\s+/).filter((w) => w && !STOP.has(w.toLowerCase()));
  const code = words
    .map((w) => w[0]?.toUpperCase() ?? "")
    .join("")
    .slice(0, 3);
  return code || name.slice(0, 3).toUpperCase();
}

function hashHue(name: string): number {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) % 360;
  return h;
}

export function teamCode(name: string): string {
  return TEAMS[name]?.code ?? initials(name);
}

export function teamColor(name: string): string {
  return TEAMS[name]?.color ?? `hsl(${hashHue(name)} 58% 56%)`;
}

export const TEAM_LOGOS_ENABLED = process.env.NEXT_PUBLIC_TEAM_LOGOS === "1";
