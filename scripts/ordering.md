# How do LLMs order things?

I want to figure out how an LLM handles an ordering task.

## Basic layout of the project

Here is the prompt:

> Please put the following dates in order:
>
> August 2nd, March 15, December 21, October 31, June 21, July 20

The output from gemma-2-9b-it:

> Here are the dates in order from earliest to latest:
> 
> 1. **March 15**
> 2. **June 21**
> 3. **July 20**
> 4. **August 2nd**
> 5. **October 31**
> 6. **December 21** 
> 
> 
> Let me know if you have any other dates you'd like me to sort!

What we're trying to figure out is using transformer_lens tools to see how the
model figures out which date to put in which place.
