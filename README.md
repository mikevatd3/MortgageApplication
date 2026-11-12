# LLM Bias in Mortgage Lending use cases

## Data pipeline

Creating highly-realistic prompts relies on real-world source data. Prompt
generation either draws from these directly, or draws from an intermediate
model. Names are drawn from various voter files (first and last together).

Sources included:

- HMDA - 2025
- ACS - 2024
- Names (review the information on this source)
- Voter Files
- others ...

## City market normalization

ACS place, HMDA price

## Comparison queue system

Once all the data is compiled and the modeling complete, prompts can be
generated. This code deals with book-keeping for these large generated datasets.

- Markdown vs csv tables
- Manifest documents
- Model output

## Scope points

This tool allows you to place points or spans in a prompt at which to collect
activations to make analysis more tractable. Can this go further? It would be
interesting to see how this can turn into graphs. Can you model collection
points on outputs?

## Preliminary token models

- [ ] Name tokenization against Rosenman tables (used for BISFG)
- [ ] Name tokenization against NC voter file (contains race data)
- [ ] City tokenization against race data
- [ ] One vs two token spans for name and city

## TODOs

- [x] Set up probe points system
- [ ] Set up prompt template with probe points
    - Probe points on name and city
- [ ] Figure out probes on ordering task
- [ ] Run example
- [ ] Analyze (how)

- [ ] Circuit tracing for logit difference
    - More interesting if the selection was different -- we can pick up those
      cases. How and why those selections are different.
    - Logit difference is a really useful tool. Control variable of a regression
      model. Why a certain feature moves over another. Pre-made SAEs cross-layer
      transcoder models.
- [ ] Generate quite a few of the 'different options'
- [ ] Case of white-sounding names -- if the selection becomes different.
- [ ] Model has some biases for white-sounding names, we would thinking that we
      could see some differences on white sounding names. White-sounding name
      intervention.
- [ ] Baseline data doesn't already -- use ambiguous names for this


### Challenge against baseline & basic tracing

- [ ] The prioritize-one task
    - [ ] Change to last-name only for choice
    - [ ] Try to limit to one-token names
    - Baselines to try:
        - [ ] White
        - [ ] Ambiguous
Working 
Workflow: write experiment specifics into quarto doc. Write full experiments as
functions in package, drawing those into quarto docs. Do analysis directly in
the doc.

Try Yes / No task
Look at what is attended to:

- Yes - No
- Accept - Reject
- Pass - Fail
- Approve - Review
- A - E Grades

Do out-of-fold


7,190
+ 1,000

Look at features -- make sure it all lines up.
How they created or generated value -- similar to blue book.
Same exact vehicle hybrid - stay within the area - close to area as possible.
120 mi -- stay within the market. Listed as comps -- blue rectangle. Mileage,
find similar vehicles in the market. 101,000 miles decrease or subtract off of
the price. Higher mileage than that, condition, and options. Better options.

Need VIN numbers. Same year, make model and trim. Mileage and condition and
features should be the same.

Alicia

Getting billable
