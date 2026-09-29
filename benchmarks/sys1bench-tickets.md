# Support tickets

Reader: someone asking why English ticket priority is 246/425 and multilingual priority is 280/476.

Source: [support_tickets.py](https://github.com/rssr25/system-one-bench/blob/main/src/sys1bench/generators/support_tickets.py) and [multilingual_tickets.py](https://github.com/rssr25/system-one-bench/blob/main/src/sys1bench/generators/multilingual_tickets.py) in [system-one-bench](https://github.com/rssr25/system-one-bench), `n=500`, `seed=42`, framing `f0`.

## How a priority is scored

The question prints this rule: start from the template urgency (0 informational, 1 routine, 2 blocked or money at stake), add 1 if the customer is angry, add 1 if the tier is gold or enterprise, and cap at 4.

The grade is the level with the highest probability. Before any cut the model scored **246/500** English and **280/500** multilingual.

Queue choice was 500/500 on the English set. The anger yes/no was also 500/500 there. The misses below are the priority number only.

## What was left out

**75** English questions and **24** multilingual questions.

The author level is 0. Level 0's caption is "No action needed or informational." The customer asks for something: an invoice, an email change, a delivery date, whether access survives cancellation, or an annual discount. The model chose level 1 on all 99, so the cut removes misses only.

Example, `tickets_42_00001`: "If I cancel, do I keep access until the end of the billing period?" Author level 0. Model level 1 at 0.79.

The 24 multilingual items are the calm annual-discount question. Angry or enterprise variants of that question have author level 1 or 2, so they stay in the score.

After the cut: **246/425** English and **280/476** multilingual.

## What remains

**179** English priority misses and **196** multilingual ones. On every one of them the author level equals the printed formula. The model picks the neighbouring caption instead.

Level 2 says "Customer blocked or money at stake." Level 3 says "Urgent: blocked plus angry or high-value customer." A double charge is template urgency 1, so a calm standard-tier ticket is level 1. The model reads "money" and answers 2. An enterprise ticket is base plus one, so it is often level 2. The model reads "high-value" and answers 3.

### English, 179

| Pattern | Count | Author level | Model level |
| --- | ---: | ---: | ---: |
| Duplicate charge or autopay late fee, standard or free tier | 31 | 1 | 2 |
| Missing delivery, standard or free tier | 7 | 1 | 2 |
| High-tier routine billing or shipping | 33 | 2 | 3 |
| Angry billing, template urgency 1 | 6 | 2 | 3 |
| Angry missing delivery, standard or free | 7 | 2 | 3 |
| Angry cancellation | 5 | 2 | 3 |
| App crash, template urgency 2, not angry | 7 | 2 | 3 |
| Angry technical, template urgency 1 | 13 | 2 | 3 |
| Formula says 2, model stays at 1 | 33 | 2 | 1 |
| Formula says 1, model jumps to 3 | 6 | 1 | 3 |
| Formula says 4, model stops at 3 | 6 | 4 | 3 |
| Other formula-2 tickets scored as 3 | 24 | 2 | 3 |
| Formula says 3, model says 4 | 1 | 3 | 4 |

31+7+33+6+7+5+7+13+33+6+6+24+1 = 179.

`tickets_42_00011`. "My autopay failed on Sept 14 and now there's a late fee of $120. I had funds available." Template urgency 1, not angry, standard tier, so the rule says 1, caption "Routine request; respond within 2 business days." Model level 2 at 0.80.

`tickets_42_00014`. "Order #49092 says delivered on Sept 14 but nothing arrived." Enterprise tier, so base 1 plus 1 is 2. Model level 3 at 0.54, the caption that mentions a high-value customer.

`tickets_42_00010`. "The app crashes on launch since the update on Monday. I can't access anything." Template urgency 2, not angry, free tier, so the rule says 2. Model level 3 at 0.67.

### Multilingual, 196

The question and the level captions are English. The customer text is German, Spanish, French, or Hindi. The formula is the same. Only the sales template has urgency 0, and the calm cases are the 24 already removed.

| Pattern | Count | Author level | Model level |
| --- | ---: | ---: | ---: |
| Duplicate charge, standard or free | 22 | 1 | 2 |
| Duplicate charge, gold or enterprise | 32 | 2 | 3 |
| Angry duplicate charge | 10 | 2 | 3 |
| Missing delivery, standard or free | 17 | 1 | 2 |
| Missing delivery, gold or enterprise | 30 | 2 | 3 |
| Angry missing delivery | 13 | 2 | 3 |
| Cancellation, gold or enterprise | 23 | 2 | 3 |
| Angry cancellation | 10 | 2 | 3 |
| Angry annual-discount question | 12 | 2 | 3 |
| Formula says 3, model says 4 | 15 | 3 | 4 |
| Formula says 2, model stays at 1 | 12 | 2 | 1 |

22+32+10+17+30+13+23+10+12+15+12 = 196.

`mltickets_42_de_00004`. "Mir wurde am Montag zweimal 19 € für Bestellung #17692 abgebucht." Author level 1. Model level 2 at 0.62.

`mltickets_42_es_00005`. A double charge on an enterprise account. The rule says 2. Model level 3 at 0.64.
