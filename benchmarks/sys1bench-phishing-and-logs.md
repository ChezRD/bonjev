# Phishing urgency and log paging

Reader: someone asking which phishing and log misses are still counted after the crooked urgency templates were removed.

Source: [phishing_email.py](https://github.com/rssr25/system-one-bench/blob/main/src/sys1bench/generators/phishing_email.py) and [log_triage.py](https://github.com/rssr25/system-one-bench/blob/main/src/sys1bench/generators/log_triage.py) in [system-one-bench](https://github.com/rssr25/system-one-bench), `n=500`, `seed=42`, framing `f0`.

Urgency is graded by the level with the highest probability. Paging is a yes/no question: correct when the probability of `true` is on the same side of 0.5 as the author label.

Raw urgency was **308/500**. Raw paging was **419/500**. Log severity was **500/500** and nothing was removed from it.

## What was left out

**98** urgency questions, all misses.

| Template | Items | Author level | What the model did |
| --- | ---: | ---: | --- |
| "Team lunch Thursday" / "Reply if you have dietary needs." | 51 | 1 | Level 2 |
| "Scanned document" / "Open to review." | 47 | 1 | Level 0 or 2 |

Level 1's caption is "Mild: 'when you get a chance'." Level 2's caption is "Moderate: action requested within days." Thursday is within days, and "open to review" does not use the level-1 phrase. Those 98 are removed from both sides of the fraction, so urgency becomes **308/402**.

The canary uses the same two templates and drops **28** urgency questions the same way.

## Timesheet: 43 real misses

50 items. The model got 7 right and missed 43. All 43 chose level 2.

The body contains the level-1 phrase: "Hi Alex, timesheets are due Friday. Submit through the usual portal when you get a chance." The author level is 1 because that phrase is the caption of level 1.

`phish_42_00001`. Author level 1. Model level 2 at 0.50, with level 1 at 0.44. The Friday deadline pulled the answer to "within days."

## Invoice mail: 51 real misses

51 items. The model missed all of them, every time at level 0.

"Hi Noor, attached is the invoice we discussed on the call. Let me know if anything looks off." There is no deadline and no threat. Level 0 is "No time pressure." Level 1 is the mild follow-up. The author level is 1.

`phish_42_00000`. Author level 1. Model level 0 at 0.93.

No urgency miss sits outside these four templates (lunch, scanned document, timesheet, invoice). Lunch and scanned document are the 98 already removed. Timesheet and invoice are the 94 that stay.

## Paging: 81 real misses

Nothing was removed. The item says: "Page the on-call engineer only when severity is 2 or higher AND env is production." The generator sets page-on-call to that rule, and ERROR is severity 2.

All 81 misses are ERROR, production, author label true, model answer false. Severity on the same items was right. The model did not apply the "and production" line once the severity was exactly 2.

`logs_42_00001`. Log level ERROR, env production, "token validation failing for 23% of requests since v2.24." Author label true. Probability of true: 0.21.

There is no other paging miss in the 500.
