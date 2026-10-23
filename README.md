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
