# ID-SEN dossier (as of 2026-11-01)

**Kalshi:** Jim Risch (R) 0.944 normalised (mid 0.948) vs Todd Achilles (I) 0.055, David Roth (D, withdrawn) 0.001. Volume about $1.17M.

## Polls (polls_2026.csv / Wikipedia rev 1382544118)
- VoteNexus Research (Knock Every Door PAC + Achilles campaign), Oct 21-22, 700 LV: Risch 49, Achilles 43
- Peak Insights (R; Risch campaign), Sep 28-30, 500 LV: Risch 49, Achilles 24 (released Oct 2)
- Advanced Targeting Research (sponsor not listed), Sep 13-16, 700 RV: Achilles 50, Risch 39 / Achilles 46, Risch 39 with Libertarian Loesby at 5
- Older: Peak Insights (Risch), Aug 11-13: 52-18. Bullfinch (Achilles), Aug 4-7: Achilles 38-33 / Risch 34-30. Change Research (Stegner), Jul 28-30: Risch 45-21 (43-19-13 with Roth).
- There is no independent media or academic poll. Silver Bulletin average (updated Oct 26): Risch 50.6, Achilles 34.6 (R+16).

## Ratings
Safe/Solid R from all raters (Cook Solid R, Oct 20). No changes.

## Money (FEC through Oct 14, Wikipedia)
Risch raised $5.01M, $2.12M cash on hand. Achilles raised $1.92M (up from $0.69M at June 30), $0.20M cash on hand. fec_2026.csv shows Risch receipts of $3.94M, which looks like a different window. Wikipedia gives no outside-spending data. Knock Every Door PAC is active for Achilles.

## News
- Jul 28: Democratic nominee David Roth withdrew after a dispute with the state party (Wikipedia). His ballot status is unconfirmed.
- Endorsements: Risch has Trump (Mar 2025), Butch Otter (Sep 13) and the US Chamber (Sep 17). Achilles has the Forward Party/Whitman/Yang (op-ed Oct 15), AFGE (Sep 2) and the Idaho Education Association (Oct 16).
- I found no debates, scandals or health news. A web search budget limit cut off further news searching.

## Structure
Incumbent R in an R+18 state against a de facto unified independent challenger (a former Democratic legislator). Natalie Fleming (I) and Matt Loesby (L) are also on the ballot. A withdrawn Democrat may still appear on the ballot. No RCV, so plurality wins.

## Judgment
**market_about_right. q_underdog (Achilles) = 0.04. Confidence: medium.**

The polls supporting Achilles come from his own camp or allies (VoteNexus, Bullfinch) or from an unlisted sponsor (ATR, RV). Even his own late poll has him trailing by 6. Risch's internals show a lead of more than 20 points. Idaho is very red, and Achilles has only a small cash reserve. A 5.5% price is about fair, maybe slightly generous.

## Sources
- https://en.wikipedia.org/wiki/2026_United_States_Senate_election_in_Idaho (accessed 2026-11-01)
- data/clean/polls_2026.csv, ratings_2026.csv, fec_2026.csv

## Skeptic review (2026-11-01)
**Verdict: weakened. The conclusion (Risch heavily favoured) stands, but the dossier's polling evidence is stale. Skeptic q_underdog (Achilles) = 0.06.**

Note: the shared WebSearch budget was used up. This review uses direct fetches only (Idaho Capital Sun and Ballotpedia returned 403 or empty), and late Idaho news was not checked.

**Claims checked**
- These polls match Wikipedia (https://en.wikipedia.org/wiki/2026_United_States_Senate_election_in_Idaho, accessed 2026-11-01):
  - VoteNexus Oct 21-22 (49-43; commissioned by Knock Every Door PAC and the Achilles campaign)
  - Peak Insights/Risch Sep 28-30 (49-24; other: Fleming 3, Loesby 2, none 6)
  - ATR Sep 13-16, 700 RV (Achilles 50-39; 46-39 with Loesby 5; no sponsor listed)
- FEC numbers match: Risch $5.01M raised, $2.12M cash on hand; Achilles $1.92M raised, $195,819 cash on hand.
- Every rater has Safe or Solid R.

**Errors**
- **Main error:** the dossier cites the Silver Bulletin average (Risch 50.6, Achilles 34.6, +16) as current. Wikipedia says that average was "updated Oct 26" but **uses polls only through Aug 13**. It leaves out all three September and October polls, so it is not evidence about the race's late state.
- The newer poll aggregate is much closer. PollCheck (updated **1 Nov 2026**) shows **Risch 46.1, Achilles 42.0, a 73% Risch win probability, "Lean R"** (https://www.pollcheck.co.uk/us-midterms-2026/senate/idaho). It averages sponsored polls naively, but its 27% for Achilles is far above Kalshi's 5.5%. That is the largest model-versus-market gap of these five races.
- Data issue: one polls_2026.csv row for Peak Sep 28-30 shows Achilles at 35. The Wikipedia topline is 24.

**Refuting the case for an Achilles upset**
Taken at face value, the gap in the previous point looks like an underdog signal. It rests on thin evidence:
1. Achilles's own late poll, VoteNexus, has him **down 6** with 8% undecided. Campaign-released polls usually flatter the sponsor.
2. The one poll with Achilles ahead, ATR, is a September RV poll with no listed sponsor and an unknown track record. I could not verify who paid for it.
3. Peak (Risch's campaign) has him up 25. The truth is probably somewhere in between, which still means a Risch lead.
4. Idaho is R+18 (lean_2026.csv).
5. Achilles had less than $200K cash on hand on Oct 14. No outside-spending total for him could be verified.
6. Recent independent challenges have lost: Osborn in NE in 2024 polled close but lost by about 7 points. That margin is from memory, not a source checked in this review.

**Assessment**
Risch is still the strong favourite. Because no high-quality public poll exists and the race is much closer in the polls that do exist, I would put Achilles slightly above the market, at about 6% against 5.5%. The difference is too small and too poorly evidenced to support an Achilles pick.
