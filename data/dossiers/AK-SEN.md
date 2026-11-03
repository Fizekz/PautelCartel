# AK-SEN: Alaska Senate (Sen. Dan S. Sullivan vs. Mary Peltola, ranked-choice)

**Kalshi (2026-11-01 07:48Z):** Kalshi lists parties only. Democratic 0.715 / Republican 0.295 (normalised D 0.708). Volume about $5.5M.
**Verdict:** market_underrates_underdog (slightly). q_underdog (Republican/Sullivan) = 0.37. Confidence: low-medium.

## Polls since 2026-09-15 (source: Wikipedia rev 1383141665 via data/clean/polls_2026.csv). Format: first-choice, then final RCV round where reported.
| Pollster (sponsor) | Dates | n | Peltola | Sullivan | Final RCV round |
|---|---|---|---|---|---|
| Alaska Survey Research | Oct 26-29 | 1326 LV | 48 | 46 | 51-49 Peltola |
| Data for Progress (D) | Oct 8-16 | 660-694 LV | 47-49 | 44-46 | 50-50 / 51-49 Peltola |
| Rasmussen (R) | Oct 12-15 | 775 LV | 44 | 42 | n/a |
| PPP (D) | Oct 14-15 | 578 RV | 48 | 47 | head-to-head |
| InsiderAdvantage (R) | Oct 9-10 | 1200 LV | 48 | 43 | 50-48 Peltola |
| Alaska Survey Research | Oct 1-3 | 1512 LV | 47 | 45 | 51-49 Peltola |
| NYT/Siena | Sep 24-Oct 1 | 504 LV | 50 | 43 | head-to-head |
| Quantus (R) | Sep 28-30 | 758 LV | 46-47 | 46-47 | 50-50 |
| Trafalgar (R) | Sep 22-24 | 1079 LV | 46 | 47 | n/a |
| co/efficient (R) | Sep 14-17 | 799 LV | 46 | 49 | n/a |

Read-across: the first-choice average is about Peltola +2 to +3. Every final-round RCV simulation is 50-50 to 51-49 Peltola, except InsiderAdvantage at 50-48. NYT/Siena's +7 head-to-head is the outlier. Namesake Dan J. Sullivan takes 3-5% of first choices and spoiler Gerald Heikes (R) about 1-4%, and both redistribute mostly to Sullivan. PollCheck's average (Oct 27) is Peltola 47.2, Sullivan 44.1.

## Ratings
- As of 2026-10-30 (ratings_2026.csv): Cook Toss-up, Sabato Lean D, Inside Toss-up. Split Ticket and FiftyPlusOne have it at Lean D.
- Sabato moved it from Toss-up to Lean D on 2026-10-06 (search summary of Sabato's rating-changes page). Inside Elections did not change the AK Senate rating in its final Oct 31 batch (Roll Call).

## Money
- FEC through 2026-10-14: Peltola raised $36.9M ($4.10M cash on hand); Sullivan raised $11.5M ($2.84M cash on hand). I found no figure for total outside spending.

## News
- 2026-10-28: the New York Times reported, based on interviews with 12 people including fired former campaign manager Elisa Rios, that DSCC aides introduced Democratic operative Jerid Kurtz to Peltola's then-campaign manager, and that Kurtz located retired teacher Daniel J. Sullivan to run. The Peltola campaign and DSCC deny involvement. (AP via WAFB, https://www.wafb.com/2026/10/28/report-peltolas-campaign-involved-effort-recruit-second-dan-sullivan-alaska-senate-race/)
- 2026-10-29: Gov. Dunleavy ordered the Department of Public Safety to investigate. DOJ amended its suit (filed Oct 23) seeking to block certification if Sullivan loses by a margin attributable to "voter confusion", and issued at least three dozen subpoenas. Asked whether Senate Republicans should block Peltola from being seated, Trump said "you're going to have to see what happens." (CNBC, https://www.cnbc.com/2026/10/29/alaska-senate-governor-investigation-sullivan-peltola-doj.html)
- 2026-10-26: Judge Sharon Gleason fast-tracked the case. The state asked for a dismissal ruling by Nov 2. (CNBC, https://www.cnbc.com/2026/10/26/alaska-senate-peltola-doj-lawsuit-election-sullivan.html)
- 2026-09-23/24: earlier reports of hostile workplace behaviour by Peltola toward ex-staffers (ADN 2026-09-26). The NYT/Siena +7 poll was taken just after.
- 2026-10-15: final debate, described as fiery and full of mudslinging (Alaska Beacon, 2026-10-16). Trump endorsed Sullivan and campaigned in the state.
- The August top-four primary had Peltola 49.5% and Sullivan 41.4% (search summary of 2026-10/11 coverage).
- RCV counting can take up to about 15 days after Nov 3 (Newsweek).

## Structure
Top-four primary, ranked-choice general. Four candidates are on the ballot: Dan S. Sullivan (R, inc.), Peltola (D), Daniel J. Sullivan (no party label), and Gerald Heikes (R). The namesake confusion probably costs the incumbent some first-choice votes, but RCV transfers recover most of them, so the exhausted-ballot share matters. Federal litigation could delay or contest certification. If the Kalshi contract settles on certification, that adds tail risk; I have not checked the rules.

## Judgment
Peltola is a strong candidate, has a 3:1 money edge and a D+8 to D+9 national environment, and leads first-choice polling. But the decisive final-round RCV numbers are effectively 50-50 to 51-49 across ASR, DFP, Quantus and InsiderAdvantage. Alaska is Trump +13 (2024), and polling there is sparse and noisy. A fresh scandal (NYT, Oct 28) broke after most polls were fielded; only ASR (Oct 26-29) straddles it, and it shows no change. A 1-2 point final-round lead suggests about 60-65%, not 71%. The certification and litigation tail cuts slightly against Peltola as well. I set q(R) = 0.37. This is a modest, low-medium-confidence disagreement, not a strong underdog call.

## Sources
- data/clean/polls_2026.csv (https://en.wikipedia.org/wiki/2026_United_States_Senate_election_in_Alaska?oldid=1383141665)
- https://www.cnbc.com/2026/10/29/alaska-senate-governor-investigation-sullivan-peltola-doj.html
- https://www.cnbc.com/2026/10/26/alaska-senate-peltola-doj-lawsuit-election-sullivan.html
- https://www.wafb.com/2026/10/28/report-peltolas-campaign-involved-effort-recruit-second-dan-sullivan-alaska-senate-race/
- https://www.newsweek.com/mary-peltola-is-pulling-ahead-in-the-final-stretch-against-dan-sullivan-12592675
- https://alaskabeacon.com/2026/10/16/alaskas-final-u-s-senate-debate-devolves-into-mudslinging-between-peltola-sullivan/
- https://www.adn.com/politics/2026/09/26/how-ex-staffers-allegations-of-hostile-behavior-by-peltola-might-affect-alaskas-key-us-senate-race/
- https://www.pollcheck.co.uk/us-midterms-2026/senate/alaska
- https://rollcall.com/2026/10/31/midterm-elections-race-rating-changes/

## Skeptic review
_Independent skeptic, 2026-11-01. The web-search budget was exhausted, so this review relies on direct fetches of cited and primary pages._

**Verdict: WEAKENED (market_underrates_underdog).** Skeptic q(R / Sullivan) = 0.33, against Kalshi's 0.295.

**Claims verified**
- ASR Oct 26-29, n=1326 LV: first choice Peltola 48, Sullivan 46; final RCV round 51-49 Peltola. DFP Oct 8-16: final round 51-49, with a 50-50 hypothetical. InsiderAdvantage Oct 9-10: 48-43 first choice. Rasmussen Oct 12-15: 44-42. PPP Oct 14-15: 48-47. Source: Wikipedia AK-SEN page, fetched 2026-11-01.
- Dunleavy ordered a DPS investigation, the DOJ amended its Oct 23 suit, and NBC reported at least three dozen subpoenas. Source: CNBC, 2026-10-29.
- Elisa Rios, fired in August, is the ex-campaign manager behind the NYT report. Source: WAFB/AP, 2026-10-28.

**Errors and unverified claims**
- The AP/WAFB article does not name the operative. Wikipedia's account spells the name "Jarid Kurtz", not "Jerid". I could not verify the "12 sources" figure.
- WAFB says Rios "tried to retract her comments last week", which weakens the core sourcing of the scandal. The dossier omits this.
- "Only ASR partly postdates the scandal": correct. Its fieldwork (Oct 26-29) overlaps the Oct 28 story by only about 1.5 days.

**Missed evidence (favours Peltola)**
- Sabato (Oct 27), Split Ticket (Oct 27) and FiftyPlusOne (Oct 31) all rate the race Lean D (flip). Every other rater has it at Toss-up and none leans R. Source: Wikipedia predictions table; ratings_2026.csv.
- Polling aggregates updated Oct 31 all show Peltola ahead: DDHQ +1.3, FiftyPlusOne +1.9, RCP +1.2, Silver Bulletin +3.0. Source: Wikipedia, fetched 2026-11-01.
- Peltola leads the final RCV round in every published final-round simulation, ASR twice and DFP once. DFP's 50-50 is a hypothetical variant.

**Missed evidence (favours Sullivan)**
- On Oct 30, AP reported that Alaska's top federal prosecutor said a criminal probe into the "would-be spoiler" is underway.
- On Oct 31, AP/ADN reported that the FBI visited a Dan J. Sullivan donor in Petersburg as the inquiry widened.
- On Oct 30, Peltola denied knowing of the other candidates (ABC).
- These headlines and dates come from Wikipedia's reference list; I did not read the articles themselves.

**Assessment.** The scandal is escalating, but it broke on Oct 28, before the Nov 1 Kalshi snapshot, so the price should already reflect it. The dossier's 0.37 leans on reading 51-49 final rounds as near-coin-flips, yet every aggregate and every leaning rater favours Peltola. The disagreement with the market is about 3-4 points, which is noise. I have not checked how Kalshi settles the contract if certification is delayed.
