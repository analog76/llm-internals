# Gradient Descent: How a Machine Learns Without Being Told the Answer

Every neural network starts out wrong. Not a little wrong — *randomly* wrong, because its numbers were literally filled in at random.

Nobody ever tells it what those numbers should have been. And yet it fixes itself.

The thing doing the fixing is gradient descent. It's one line of arithmetic, repeated. This post builds it up from nothing: why it has to exist, what the formula means, what happens when you run it, and how the same line scales from one number to the hundreds of billions inside a language model.

---

## Why do we need gradient descent at all?

Picture a machine with one knob on the front.

You feed it a `2`. It's supposed to output a `10`. Right now the knob is at `3`, and the machine multiplies — so it outputs `6`.

Wrong. You need the output to climb by 4.

Here's the part that trips everyone up the first time: **you know the right answer, but you don't know the right knob setting.**

Those are completely different things. "The output should be 10" is what you were given. "The knob should be at 5" is what you have to *find*. The answer key tells you about the output. It says nothing about the knob.

So how do you find the setting? You can't look inside — a real network has millions of knobs wired to each other and you have no map. All you can do is turn a knob a little and see whether the output got better or worse.

That's the whole situation, and there's a good metaphor for it:

> You're standing on a hillside in thick fog. You want to reach the valley floor. You can't see the valley, you can't see the hill, you can't see three feet in front of you. But you *can* feel which way the ground tilts under your feet.
>
> So you step downhill. A small step. Then you feel the ground again — it tilts differently now, you've moved. Step downhill again. Repeat.

You'll reach the bottom. Not because you knew where it was, but because "downhill" was available at every single point along the way.

Gradient descent is that walk, done with arithmetic. The tilt of the ground is called the **gradient**. Everything else in this post is detail.

---

## What if you didn't have it? What would you do instead?

Worth asking, because the alternatives are what make gradient descent look less like a clever trick and more like the only door out of the room.

**Option 1: guess randomly.** Set the knob to a random number, check the error, keep it if it improved. With one knob you'd stumble into something decent eventually. With two knobs it's slower. With a thousand, you will never randomly hit a good combination — not in the lifetime of the universe. Random guessing dies the moment you have more than a couple of knobs.

**Option 2: try every possible value.** Be systematic instead of lucky. Sweep the knob from -1000 to +1000 in steps of 0.001 and keep the best. Two million tries, but a computer does that instantly. Problem solved?

It's solved for *one knob*. Watch what happens when you add more, because every new knob multiplies the work rather than adding to it:

| Knobs | Combinations to check (1,000 values each) |
| --- | --- |
| 1 | 1,000 |
| 2 | 1,000,000 |
| 3 | 1,000,000,000 |
| 10 | 1 followed by 30 zeros |
| 100 | 1 followed by 300 zeros |

For scale: there are about 10⁸⁰ atoms in the observable universe. We blew past that at **27 knobs**.

A small neural network has thousands. A language model has hundreds of billions. Trying combinations isn't slow — it's not a thing that can happen.

**Option 3: just solve it with algebra.** For our toy machine you genuinely can. "Find the knob where the error is zero" is a one-line equation, and it gives `w = 5` exactly, no searching.

This works, and then it stops working. Stack a few layers with nonlinear functions between them and "solve for the knobs" becomes a system of billions of tangled equations with no closed-form solution. Not hard — *nonexistent*. There is no formula to write down.

So: can't guess, can't enumerate, can't solve.

**What's left is the fog.** You can't see the landscape, but you can always measure the slope *right where you're standing*. That measurement is cheap, it's exact, and — this is the crucial part — **its cost doesn't explode as you add knobs**. One pass through the network hands you the slope for every knob at once. A hundred billion of them, in one go.

That's the trade gradient descent makes. Give up the global view. Keep the local slope. Take a lot of steps.

---

## The formula

In words, before symbols:

```
new setting  =  old setting  −  (step size × slope)
```

That's it. That's gradient descent. In symbols:

$$
w \;\leftarrow\; w \;-\; \eta \cdot \frac{\partial L}{\partial w}
$$

Four pieces:

| Symbol | Name | In plain terms |
| --- | --- | --- |
| $w$ | weight | the knob — the only thing you're allowed to change |
| $L$ | loss | one number for how wrong you currently are |
| $\dfrac{\partial L}{\partial w}$ | gradient | the slope: if I nudge the knob, does the error rise or fall, and how sharply |
| $\eta$ | learning rate | how big a step to take, chosen by you |

Three things to notice, because they're where the cleverness hides.

**The minus sign does the thinking.** You never decide whether to add or subtract. The slope comes out signed, and subtracting it always moves downhill:

- Slope is **positive** — turning the knob up makes things worse — subtracting a positive turns it **down**. ✓
- Slope is **negative** — turning the knob up makes things better — subtracting a negative turns it **up**. ✓

Same line of code, both directions, no `if` statement. That's the trick.

**The step size isn't $\eta$.** It's $\eta$ **times the slope**. So on a steep slope you take a big stride, and as the ground flattens out near the bottom your steps automatically shrink. Gradient descent brakes on its own as it arrives. Nobody programs that in; it falls out of multiplying by the slope.

So if $\eta$ isn't the step, what is it? Back to the foggy hillside.

The ground tells you how steep it is. That part isn't up to you — you just feel it. $\eta$ is your **temperament**: how much stride you give per unit of steepness.

Send three hikers down the identical hill, feeling the identical ground:

> 🐢 **The timid one** ($\eta$ too small) shuffles. Even where the ground plunges away, they inch forward. They'll reach the bottom — sometime next week.
>
> 🚶 **The sensible one** ($\eta$ about right) strides out where it's steep and shortens up as it levels off. Down in a dozen steps.
>
> 🏃 **The reckless one** ($\eta$ too big) feels a steep patch and *leaps*. Sails clean over the valley and lands higher up the far side than where they started. Feels an even steeper patch there. Leaps again, harder. They are now going up the mountain, at speed, forever.

Nobody handed them different information. The hill was the same for all three, and all three read the slope correctly. The only difference is how aggressively each one converts "steep" into "stride."

That's $\eta$ — and it's also why $\eta$ has to exist at all. A slope isn't measured in knob-units, so you can't simply subtract it from the knob; the units don't match. $\eta$ is the exchange rate between *how wrong we are* and *how far to turn*.

**The slope is measured, not guessed.** This isn't a heuristic or an estimate. Calculus gives the exact tilt at the exact point you're standing. The only guesswork in the entire system is where you started and how big $\eta$ is.

---

## Watching it actually run

Let's make the fog concrete. Here's our error, as a function of the knob:

$$
L(w) = (w-4)^2
$$

Its slope at any point is $2(w-4)$. We'll use a step size of $\eta = 0.3$.

You can see the valley is at `w = 4`. **The algorithm cannot.** It only ever gets to ask one question: *what's the slope right here?*

### Starting too low, at w = 1

| Step | Knob | Error | Slope | What it does |
| --- | --- | --- | --- | --- |
| 0 | 1.000 | 9.000 | −6.00 | steep, and negative → big step **up** |
| 1 | 2.800 | 1.440 | −2.40 | less steep → smaller step up |
| 2 | 3.520 | 0.230 | −0.96 | gentler still |
| 3 | 3.808 | 0.037 | −0.38 | creeping in |
| 4 | 3.923 | 0.006 | −0.15 | nearly there |
| 5 | 3.969 | 0.001 | −0.06 | → 4 |

### Starting too high, at w = 10

Same formula. Nothing changed. We just dropped it somewhere else on the hill.

| Step | Knob | Error | Slope | What it does |
| --- | --- | --- | --- | --- |
| 0 | 10.000 | 36.000 | +12.00 | steep, and positive → big step **down** |
| 1 | 6.400 | 5.760 | +4.80 | overshooting less |
| 2 | 4.960 | 0.922 | +1.92 | closing in |
| 3 | 4.384 | 0.147 | +0.77 | slowing down |
| 4 | 4.154 | 0.024 | +0.31 | nearly there |
| 5 | 4.061 | 0.004 | +0.12 | → 4 |

**Both runs land on 4, from opposite directions, with identical code.** One walked up, one walked down. Neither was told which way to go, and neither was told where 4 was. The slope handled it.

Look at the error columns: 9 → 1.44 → 0.23 → 0.037, and 36 → 5.76 → 0.92 → 0.147. Both collapsing fast, and both decelerating as they arrive, exactly as promised.

Here's the whole thing, runnable:

```python
w, eta = 1.0, 0.3            # try 10.0 too

for step in range(6):
    error = (w - 4) ** 2
    slope = 2 * (w - 4)
    print(f"step {step}:  w={w:7.4f}   error={error:8.4f}   slope={slope:+7.3f}")
    w = w - eta * slope      # ← the entire algorithm
```

Every neural network ever trained is that last line, wearing a lot of scaffolding.

### The one thing you actually choose

The slope is handed to you. The step size $\eta$ is yours, and it's the difference between training working and training exploding. Same starting point, same problem, six different $\eta$:

| $\eta$ | Step 0 | Step 1 | Step 2 | Step 3 | Verdict |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 1.00 | 1.60 | 2.08 | 2.46 | works, but crawling |
| 0.3 | 1.00 | 2.80 | 3.52 | 3.81 | ✓ good |
| 0.5 | 1.00 | **4.00** | 4.00 | 4.00 | perfect — by luck, see below |
| 0.7 | 1.00 | 5.20 | 3.52 | 4.19 | overshoots, zigzags, still gets there |
| 1.0 | 1.00 | 7.00 | 1.00 | 7.00 | bounces forever, never arrives |
| 1.2 | 1.00 | 8.20 | −1.88 | 12.23 | 💥 exploding |

That last row is the one to burn into memory. **The slope was correct on every single step.** The math was perfect. The steps were just too big, so each one flew past the valley and landed further up the opposite side than where it started — and then did it again, worse. This is what people mean when a training run "goes to NaN." Nothing was wrong except one number that a human picked.

That's our reckless hiker, in a table.

If you've ever overcorrected a car, you've felt the same failure from the inside. You drift slightly right, so you pull the wheel left — too hard. Now you're aimed at the oncoming lane, so you pull right, harder still. Each correction is bigger than the mistake that prompted it, and thirty seconds later you're in a ditch.

Notice what wasn't wrong there: your *reading* of the situation. You correctly identified which way you were off, every single time. What was wrong was how violently you translated that reading into steering. Engineers call that the **gain**, and too much of it makes any correcting system oscillate and then explode.

$\eta$ is the gain. That's the job.

And $\eta = 0.5$ hitting the exact answer in one step is not a discovery — it's an accident of this particular parabola. Change the problem and that same value might explode. There is no universally good learning rate, which is why tuning it is a real part of the job, and why most training runs *schedule* it: bold strides early, careful steps near the bottom.

---

## Two knobs, and where the chain rule comes in

One knob was a warm-up. Real models have more, and the moment you have two, something new is needed.

Give our neuron a second knob — a bias that just shifts the output up or down:

$$
y = w \cdot x + b
$$

With input `x = 2`, target `t = 10`, starting at `w = 3`, `b = 0`.

### Why one slope isn't enough any more

We need to know how the error responds to `w`, *and* separately how it responds to `b`. Two knobs, two slopes.

If that sounds odd — surely a hill has just *one* steepness? — here is the thing. **A slope is never the slope of a place.** It is always the slope of something *with respect to something you moved*.

Picture standing on a mountain path. Walk *along* the path and it's flat. Step *off* the edge and it drops away beneath you. You haven't moved an inch — same spot, same boots — yet there are two completely different steepnesses depending on which way you go. "How steep is it here?" isn't a finished question until you say **in which direction**.

Turning `w` is one direction. Turning `b` is another. Two directions, two answers — and genuinely different numbers, as we'll see in a moment: `w` comes out at −16 while `b` is −8.

You can't squash them into one, either. Suppose I handed you a single number — −24 — and called it "the slope." How far do you turn `w`? How far do you turn `b`? There's no way to split it back apart. **Every knob needs its own instruction, because every knob is turned separately.** One knob, one slope. Two knobs, two slopes. A hundred billion knobs, a hundred billion slopes.

(Your instinct that there ought to be *one* thing isn't wrong, by the way. It's just one **arrow** rather than one number — we'll get to that at the end of this section.)

And here's the wrinkle:

**Neither knob touches the error directly.** The knobs make a prediction. The *prediction* makes the error. Nothing connects `w` to `L` except by going through `y` first:

```
    w ──┐
        ├──▶  y  ──▶  L
    b ──┘   (prediction)  (error)
```

So "how does the error respond to `w`?" can't be answered in one move. It has to be answered in two hops, and that's precisely what the chain rule is for.

### The chain rule, in one sentence

> To get the slope across a chain, multiply the slopes of each link.

That's it. If the error is twice as sensitive to the prediction, and the prediction is three times as sensitive to the knob, then the error is six times as sensitive to the knob. Sensitivities **multiply** as they pass down a chain.

### How to picture it

Think of gears. Turn a small gear by one tooth; the gear it drives turns by three; the gear *that* one drives turns by two. Turn the first gear one tooth and the last one moves 3 × 2 = 6. You never needed a direct connection between first and last — you just multiplied along the linkage.

Or think of it as a message passed backwards. The error announces how upset it is, and that message travels back up the wires, getting scaled by each link it crosses:

```
    w  ◀────────  y  ◀────────  L
        "×2"          "−8"
     (a 1 change     (raising y by 1
      in w moves      drops the error
      y by 2)         by 8)
```

Start at the error. Walk backwards. Multiply as you go. By the time you reach a knob, you have its slope: −8 × 2 = **−16**.

That backwards walk has a name. It's **backpropagation**. It isn't a separate idea from the chain rule — it's the chain rule, applied to a whole network, organised so that no work gets repeated.

### Both knobs, in one pass

The beautiful part: the backwards message `−8` gets computed **once**, then reused by every knob feeding into that point. Each knob just scales it by its own local link:

- `w` is connected through the input, so: −8 × **2** = −16
- `b` is connected directly, so: −8 × **1** = −8

Now update both, same formula, same step size ($\eta = 0.01$):

| Step | w | b | Prediction | Error | slope for w | slope for b |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 3.000 | 0.000 | 6.000 | 16.000 | −16.00 | −8.00 |
| 1 | 3.160 | 0.080 | 6.400 | 12.960 | −14.40 | −7.20 |
| 2 | 3.304 | 0.152 | 6.760 | 10.498 | −12.96 | −6.48 |
| 3 | 3.434 | 0.217 | 7.084 | 8.503 | −11.66 | −5.83 |

### Where those numbers actually come from

That table shows results. Here is the arithmetic behind the very first row, because "the knob went from 3 to 3.16" deserves to be shown rather than asserted.

**Work out the slopes.** Starting at `w = 3`, `b = 0`:

```
prediction   y = w × 2 + b  =  3 × 2 + 0  =  6
we wanted    t = 10         →  4 short

slope on the prediction  =  2 × (6 − 10)  =  −8      ← computed once
      slope for w  =  −8 × 2  =  −16      (w is multiplied by the input 2)
      slope for b  =  −8 × 1  =   −8      (b is added on directly)
```

**Turn each knob.** Same line, twice, with a step size of 0.01:

```
w:   3 − 0.01 × (−16)  =  3 − (−0.16)  =  3 + 0.16  =  3.16
b:   0 − 0.01 × (−8)   =  0 − (−0.08)  =  0 + 0.08  =  0.08
```

Look closely at what just happened. The rule says **subtract** — always, no exceptions. But the slope is **negative**, and subtracting a negative adds. Minus times minus makes plus, and the knob goes *up*, without a single line of code ever deciding to increase it.

Had the prediction come out too *high* instead — say 14 — the slope would have been `+16`, and the identical line gives `3 − 0.16 = 2.84`. Down. Same arithmetic, opposite direction, nobody choosing.

Do that four times and you get the table above. The `w` knob:

| Step | w before | turn | w after |
| --- | --- | --- | --- |
| 0 | 3.0000 | + 0.1600 | **3.1600** |
| 1 | 3.1600 | + 0.1440 | **3.3040** |
| 2 | 3.3040 | + 0.1296 | **3.4336** |
| 3 | 3.4336 | + 0.1166 | **3.5502** |

And the `b` knob, same rule, its own slope:

| Step | b before | turn | b after |
| --- | --- | --- | --- |
| 0 | 0.0000 | + 0.0800 | **0.0800** |
| 1 | 0.0800 | + 0.0720 | **0.1520** |
| 2 | 0.1520 | + 0.0648 | **0.2168** |
| 3 | 0.2168 | + 0.0583 | **0.2751** |

The turns shrink every step — 0.160, 0.144, 0.130, 0.117 — with nothing arranging that. As the prediction closes in on 10 the slope shrinks, and the step is the slope times 0.01. The self-braking from earlier, visible as a column of numbers.

Here it is in code:

```python
x, t, eta = 2.0, 10.0, 0.01
w, b = 3.0, 0.0

for step in range(4):
    y     = w * x + b              # forward:  make a prediction
    error = (y - t) ** 2           # how wrong

    dL_dy = 2 * (y - t)            # backward: one message...
    dL_dw = dL_dy * x              #   ...scaled for w
    dL_db = dL_dy * 1              #   ...scaled for b

    w = w - eta * dL_dw            # update, same line as before
    b = b - eta * dL_db            # ...once per knob
```

Two details worth pointing out:

**`w` moves faster than `b`.** Look at the columns — `w` travels 3.000 → 3.434 while `b` only goes 0.000 → 0.217, exactly twice as slow. That's the input `x = 2` doing it: `w` gets multiplied by the input on its way to the prediction, so it has twice the leverage, so it gets twice the slope, so it takes twice the step. Knobs with more influence get moved more. Automatically.

**Two knobs beat one.** With only `w`, the error would have gone 16 → 13.54. With both, it goes 16 → 12.96. More knobs, more ways down, faster descent.

### Picturing two knobs at once

With one knob, the error was a curve and you walked left or right along it.

With two, it's a **surface** — a bowl, with `w` running east-west and `b` running north-south, and height being the error. You're a marble on the inside of that bowl.

The two slopes are the tilt in each direction. Put them together and you get an arrow pointing straight downhill on the surface — and that arrow is what "the gradient" really means. Not a number: a **direction in knob-space**.

Add a third knob and the bowl becomes something in 4D that nobody can picture, which is fine, because the arithmetic never needed you to picture it. The gradient is just a list of slopes, one per knob, and the update is applied to each one independently:

```
for every knob i:
    knob[i] = knob[i] - eta * slope_for_knob_i
```

One knob or a hundred billion — that loop is unchanged.

---

## So what does this mean inside an actual LLM?

Everything above used a target I simply handed you: "the answer should be 10." Where does that come from when you're training a language model? Nobody is sitting there writing down correct answers for a trillion words of text.

**The text is its own answer key.** Take any sentence from the training data:

> The cat sat on the ___

Hide the last word, ask the model to predict it, and you already have the target sitting right there — it's `mat`. You didn't need a human to label anything. Every single word in every document is a free training example, with the answer built in. Do this a trillion times and that's the bulk of what training a language model *is*.

From there, everything maps onto what you've already seen:

| Our toy | An LLM |
| --- | --- |
| knob `w` | ~10¹¹ weights |
| input `x = 2` | the tokens so far |
| prediction `y = 6` | a probability for every word in the vocabulary |
| target `t = 10` | the word that actually came next |
| error `(y − t)²` | cross-entropy — bigger when it was confident *and* wrong |
| one backward pass | backprop through every layer |
| `w = w − η × slope` | **the identical line, applied to all 10¹¹ weights** |

Two things genuinely change at that scale, and they're worth naming:

**You don't do one example at a time.** You run a batch of them — say a few hundred — average their slopes, and take one step on the average. It's steadier than reacting to each example individually, and GPUs are built to do the whole batch at once. That's the "SGD" in every training script you'll ever read.

**Plain gradient descent gets upgraded.** Optimisers like Adam keep a little memory of recent slopes and give each weight its own effective step size. Useful, and standard. But they're all wearing the same engine underneath: measure the slope, step against it, repeat.

And the reason this works at all — the reason a hundred billion knobs is merely expensive rather than impossible — is that one backward pass produces the slope for *every* weight at once. Not one at a time. All of them, together, in a single sweep back through the network.

That's the whole reason the fog strategy wins.

---

## Summary

**The problem.** A model's knobs start random. You know what the output should be; you have no idea what the knobs should be. You can't guess them (too many combinations), can't enumerate them (the count explodes past the number of atoms in the universe by 27 knobs), and can't solve for them (no formula exists for a deep network).

**The move.** Stop trying to see the landscape. Just measure the slope where you stand, step against it, and do it again.

**The formula.**

$$
w \;\leftarrow\; w - \eta \frac{\partial L}{\partial w}
$$

- the slope's **sign** picks the direction — the minus sign makes it downhill, always, from either side
- the slope's **size** scales the step — big when steep, small when flat, so it brakes into the valley
- **$\eta$** is the one number you choose, and the one that blows training up if you get it wrong

**The loop.**

1. **Forward** — run the input through, get a prediction
2. **Loss** — collapse the error into one number
3. **Backward** — chain rule back through the network for every knob's slope
4. **Update** — `knob = knob − η × slope`, all knobs at once
5. **Repeat**

**The one thing to hold onto.** A slope is a single snapshot at a single point — it knows nothing about where you came from or where you're going. The journey isn't in the math. **It's in the loop.** You take a step, the ground tilts differently, you measure again. Do that enough times and you're at the bottom of a valley you were never able to see.
