# to run this thing

run first `python learn.py`, this will train the literal listener on S0 and the pragmatic listener on S_CE
basically run this until L0 is able to solve most of the utterances generated by S_CE (look at the diagnostic pictures)

then run `python interact.py`, this will allow you to play with both listeners, to specify a coordinate to give, simply type `(x,y)` into the terminal, and the current guesses (from l0 and l1) will be visualized as well
