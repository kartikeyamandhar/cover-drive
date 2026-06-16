"""Hand-authored, license-clean few-shot style seed.

PROVENANCE: every line below is original, written in-house for this project. None
is copied or paraphrased from any broadcast, website, or dataset. This file is
therefore committable with zero redistribution risk, and it is the single biggest
lever on the teacher's (and thus the student's) voice quality.

Faithfulness rule the exemplars follow (and that the system prompt enforces):
a voice may add natural descriptive COLOR (the shot, the mood, the pressure) as
style, but must never state or contradict a HARD fact - a number, a name, or an
outcome - that is not in the supplied facts. "Over deep midwicket" is allowed
color; "for four" when it was a six, or naming a player not on the ball, is a
defect.

Each exemplar pairs a factual ball event and a state string (in the exact
``app/features/serialize.py`` format) with one line per persona.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Exemplar:
    """A ball event + state, with a commentary line per persona key."""

    event: str
    state: str
    lines: dict[str, str]


EXEMPLARS: tuple[Exemplar, ...] = (
    Exemplar(
        event="SIX off the bat",
        state=(
            "T20 Indian Premier League | Inns 2 | 18.3 | Chennai Super Kings 162/4 | "
            "need 13 off 9 | CRR 8.76 RRR 8.67 | Striker R Patel 41(24) | "
            "Bowler M Khan 1/34 | P'ship 47(26) | Last 1 4 0 1 W 6 | death"
        ),
        lines={
            "broadcast": (
                "And that is HUGE! Patel gets under the length and sends it sailing into "
                "the night - the equation crashes to 13 needed off 9."
            ),
            "radio": (
                "Khan hangs it on a length, Patel swings hard through the line, and this is "
                "up and carrying all the way back - six more, and Chennai suddenly need just "
                "13 from 9 balls."
            ),
            "analyst": (
                "Patel backed himself against the slower length there, and with the chase "
                "pinching he picked the right ball to clear the ropes - that one six has "
                "taken the required rate right off the boil."
            ),
            "text": (
                "18.3 Khan to Patel, SIX. Length ball heaved over the ropes. CSK 162/4, "
                "need 13 off 9."
            ),
        },
    ),
    Exemplar(
        event="WICKET, bowled",
        state=(
            "T20 Indian Premier League | Inns 1 | 7.2 | Mumbai Indians 58/3 | "
            "CRR 7.91 | Striker S Verma 12(9) | Bowler A Joseph 2/19 | P'ship 5(4) | "
            "Last 0 1 0 W | middle"
        ),
        lines={
            "broadcast": (
                "Bowled him! Joseph fires it through and the stumps are rattled - Verma's "
                "gone for 12, and Mumbai are wobbling at 58 for 3."
            ),
            "radio": (
                "Joseph in off his shorter run, full and quick, through the gate - and the "
                "off stump is flattened; Verma walks back for 12, Mumbai three down for 58."
            ),
            "analyst": (
                "Joseph had set that up beautifully, and the full one did Verma through the "
                "gate; at 58 for 3 in the seventh, Mumbai's middle order is properly exposed "
                "now."
            ),
            "text": (
                "7.2 Joseph to Verma, OUT, bowled. Beaten through the gate, off stump gone. "
                "Verma 12(9). MI 58/3."
            ),
        },
    ),
    Exemplar(
        event="FOUR off the bat",
        state=(
            "T20 Indian Premier League | Inns 1 | 2.4 | Royal Challengers Bengaluru 21/0 | "
            "CRR 8.40 | Striker V Iyer 16(9) | Bowler T Boult 0/14 | P'ship 21(15) | "
            "Last 1 0 4 1 4 | powerplay"
        ),
        lines={
            "broadcast": (
                "Crashed away! Iyer leans into it and finds the gap - four more, and the "
                "powerplay is humming along at better than eight an over."
            ),
            "radio": (
                "Boult pitches it up, Iyer drives on the up through the off side, and it "
                "races away to the rope - a lovely shot, 16 from 9 now for the opener."
            ),
            "analyst": (
                "Good intent from Iyer in the powerplay - he met the fuller ball with a "
                "clean swing of the hands and put it where the fielders weren't; 21 without "
                "loss is a tidy start."
            ),
            "text": (
                "2.4 Boult to Iyer, FOUR. Driven through the off side for a boundary. "
                "RCB 21/0, Iyer 16(9)."
            ),
        },
    ),
    Exemplar(
        event="dot ball",
        state=(
            "T20 Indian Premier League | Inns 2 | 19.2 | Kolkata Knight Riders 178/6 | "
            "need 9 off 4 | CRR 9.20 RRR 13.50 | Striker R Singh 33(21) | "
            "Bowler J Bumrah 2/29 | P'ship 18(11) | Last 1 1 0 W 1 0 | death"
        ),
        lines={
            "broadcast": (
                "Squeezed out - no run! Bumrah nails the yorker and the pressure ratchets "
                "right back up: still 9 needed, and now only 4 balls to get them."
            ),
            "radio": (
                "Bumrah at the death, right in the blockhole, and Singh can only jam down on "
                "it - no run, a huge dot, Kolkata still needing 9 from the last 4 balls."
            ),
            "analyst": (
                "That is the over-changing ball - Bumrah's yorker gave Singh nothing to work "
                "with, and a dot at this stage swings the maths firmly the bowler's way."
            ),
            "text": ("19.2 Bumrah to Singh, no run. Yorker dug out. KKR 178/6, need 9 off 4."),
        },
    ),
    Exemplar(
        event="FOUR off the bat, batter nearing fifty",
        state=(
            "T20 Indian Premier League | Inns 1 | 12.5 | Rajasthan Royals 104/2 | "
            "CRR 8.32 | Striker Y Jaiswal 48(31) | Bowler R Ashwin 1/28 | P'ship 62(40) | "
            "Last 1 2 1 0 4 | middle | nearing: batter_nearing_50"
        ),
        lines={
            "broadcast": (
                "Worked away for four, and that takes Jaiswal to 48 - he is two away from a "
                "well-made fifty, Rajasthan motoring at 104 for 2."
            ),
            "radio": (
                "Ashwin tosses it up, Jaiswal uses his feet and places it through the gap "
                "for four - 48 now, the fifty all but a formality, and the partnership ticks "
                "past sixty."
            ),
            "analyst": (
                "Jaiswal has paced this innings nicely - that boundary lifts him to 48 off "
                "31, and with a sixty-run stand behind him he has earned the platform to "
                "accelerate from here."
            ),
            "text": (
                "12.5 Ashwin to Jaiswal, FOUR. Placed through the gap. Jaiswal 48(31), "
                "nears fifty. RR 104/2."
            ),
        },
    ),
    Exemplar(
        event="two runs, tight finish",
        state=(
            "T20 Indian Premier League | Inns 2 | 19.4 | Gujarat Titans 197/5 | "
            "need 4 off 2 | CRR 9.95 RRR 12.00 | Striker D Miller 52(29) | "
            "Bowler M Shami 1/41 | P'ship 33(18) | Last 4 1 1 1 2 | death"
        ),
        lines={
            "broadcast": (
                "Two! Miller drops and runs hard, and it comes down to the simplest of "
                "equations - 4 needed off the last 2 balls, this is going to the wire."
            ),
            "radio": (
                "Shami full and wide, Miller opens the face and steals a brisk second - "
                "scores level in the head of every fan; Gujarat need 4 from 2, Miller on 52."
            ),
            "analyst": (
                "Clever from Miller - rather than risk it all he took the certain two and "
                "kept the strike alive; 4 from 2 is gettable, but the next ball decides this "
                "one."
            ),
            "text": (
                "19.4 Shami to Miller, 2 runs. Worked into the off side, hard running. "
                "GT 197/5, need 4 off 2."
            ),
        },
    ),
    Exemplar(
        event="dot ball, bowler on a hat-trick",
        state=(
            "T20 Indian Premier League | Inns 1 | 14.3 | Sunrisers Hyderabad 121/6 | "
            "CRR 8.44 | Striker B Kumar 0(1) | Bowler K Rabada 3/24 | P'ship 0(1) | "
            "Last 4 0 W W 0 | middle | nearing: bowler_on_hat_trick"
        ),
        lines={
            "broadcast": (
                "On a hat-trick, and... defended! Rabada has the new man Kumar groping "
                "forward, no run, but the field is up and the crowd is on its feet."
            ),
            "radio": (
                "Two in two for Rabada, everyone around the bat, he charges in - full, "
                "straight - and Kumar smothers it; no hat-trick this ball, but what a spell."
            ),
            "analyst": (
                "Rabada has ripped the heart out of this innings with two in two, and even "
                "the dot here keeps the squeeze on - Sunrisers six down and the pressure is "
                "all one way."
            ),
            "text": (
                "14.3 Rabada to Kumar, no run. On a hat-trick; defended solidly. "
                "SRH 121/6, Rabada 3/24."
            ),
        },
    ),
    Exemplar(
        event="SIX off the bat, powerplay",
        state=(
            "T20 Indian Premier League | Inns 1 | 4.1 | Delhi Capitals 44/1 | "
            "CRR 10.75 | Striker P Salt 29(13) | Bowler M Theekshana 0/19 | "
            "P'ship 22(11) | Last 1 0 1 4 6 | powerplay"
        ),
        lines={
            "broadcast": (
                "Gone, all the way! Salt skips down and launches it into the stands - the "
                "powerplay is flying, Delhi 44 for 1 and this man is 29 off 13."
            ),
            "radio": (
                "Theekshana floats it up, Salt is down the track in a flash and swings "
                "cleanly over the long-on rope for six - a brilliant, busy start, 44 for 1."
            ),
            "analyst": (
                "Salt is taking the spinner on inside the powerplay while the fielders are "
                "still up, and that is the smart percentage play - 29 from 13 sets the tone "
                "for the chase total."
            ),
            "text": (
                "4.1 Theekshana to Salt, SIX. Down the track, lofted over long-on. "
                "DC 44/1, Salt 29(13)."
            ),
        },
    ),
    Exemplar(
        event="WICKET, caught",
        state=(
            "T20 Indian Premier League | Inns 2 | 17.5 | Punjab Kings 154/7 | "
            "need 28 off 13 | CRR 8.66 RRR 12.92 | Striker S Curran 19(12) | "
            "Bowler H Pandya 2/30 | P'ship 11(8) | Last 1 4 1 0 W | death"
        ),
        lines={
            "broadcast": (
                "Caught! Curran goes for the big one and finds the fielder - Pandya strikes "
                "at the death, and surely that is the chase as good as over: 28 still needed "
                "off 13."
            ),
            "radio": (
                "Pandya slower ball, Curran swings across the line and skies it - taken in "
                "the deep; a huge wicket, Punjab seven down and still 28 away with barely two "
                "overs left."
            ),
            "analyst": (
                "That is the percentages catching up with Punjab - chasing twelve an over "
                "you have to take the aerial risk, and Pandya's change of pace did the rest; "
                "seven down, the required rate is now beyond them."
            ),
            "text": (
                "17.5 Pandya to Curran, OUT, caught in the deep. Slower ball, skied. "
                "Curran 19(12). PBKS 154/7, need 28 off 13."
            ),
        },
    ),
    Exemplar(
        event="one run, middle overs",
        state=(
            "T20 Indian Premier League | Inns 1 | 10.6 | Lucknow Super Giants 88/3 | "
            "CRR 8.06 | Striker N Pooran 24(17) | Bowler R Jadeja 1/22 | P'ship 19(16) | "
            "Last 1 0 1 2 1 | middle"
        ),
        lines={
            "broadcast": (
                "Pushed into the off side, comfortable single - Pooran keeps it ticking, "
                "Lucknow 88 for 3 at the halfway mark."
            ),
            "radio": (
                "Jadeja darts it in flat, Pooran works it off the back foot into the gap and "
                "trots through for one - rotating the strike nicely here, 24 to his name."
            ),
            "analyst": (
                "Sensible cricket through the middle from Pooran - taking the single, "
                "refusing the risk against Jadeja, and keeping the scoreboard moving without "
                "losing a fourth; a base to launch from later."
            ),
            "text": (
                "10.6 Jadeja to Pooran, 1 run. Worked into the off side. LSG 88/3, "
                "Pooran 24(17)."
            ),
        },
    ),
    Exemplar(
        event="FOUR off the bat, team nearing 200",
        state=(
            "T20 Indian Premier League | Inns 1 | 19.3 | Kolkata Knight Riders 196/4 | "
            "CRR 10.14 | Striker A Russell 38(16) | Bowler J Hazlewood 1/52 | "
            "P'ship 51(22) | Last 6 1 2 1 4 | death | nearing: team_nearing_200"
        ),
        lines={
            "broadcast": (
                "Four more, and that brings up... almost 200! Russell muscles it away, "
                "Kolkata 196 for 4 and threatening a monster total."
            ),
            "radio": (
                "Hazlewood goes wide of the crease, Russell reaches out and slaps it square "
                "for four - the two hundred is in sight now, 196 for 4 with an over to come."
            ),
            "analyst": (
                "Russell has turned this into a different total - 38 from 16 in this cameo, "
                "and a fifty stand at better than two a ball has dragged Kolkata to the brink "
                "of 200."
            ),
            "text": (
                "19.3 Hazlewood to Russell, FOUR. Slapped square. KKR 196/4, nearing 200; "
                "Russell 38(16)."
            ),
        },
    ),
    Exemplar(
        event="dot ball, powerplay",
        state=(
            "T20 Indian Premier League | Inns 1 | 1.3 | Mumbai Indians 6/0 | "
            "CRR 4.50 | Striker R Sharma 4(5) | Bowler Mohammed Siraj 0/2 | "
            "P'ship 6(9) | Last 0 1 0 0 0 | powerplay"
        ),
        lines={
            "broadcast": (
                "Beaten! Siraj nips it past the edge - no run, a probing start with the new "
                "ball, Mumbai watchful at 6 without loss."
            ),
            "radio": (
                "Siraj on a good length around off, Sharma leaves the line late and the ball "
                "shapes past the outside edge - no run, just the two early dots in a row, a "
                "fine opening burst."
            ),
            "analyst": (
                "This is exactly the channel Siraj wants early - on a length outside off, "
                "asking the question; Sharma is right to respect it and take no risk at 6 "
                "for none."
            ),
            "text": ("1.3 Siraj to Sharma, no run. Beaten outside off. MI 6/0, Sharma 4(5)."),
        },
    ),
)


def exemplars_for(persona_key: str) -> tuple[Exemplar, ...]:
    """Exemplars that have a line for this persona (all of them, currently)."""
    return tuple(e for e in EXEMPLARS if persona_key in e.lines)
