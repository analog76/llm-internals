# Softmax

A foundations page: the math of turning raw scores into a probability distribution. It is not an architecture deep dive. For how softmax is used inside attention, see [self-attention](../self-attention.md).

The intuition-only version of this material is the [Medium post](https://medium.com/@analog76/softmax-function-3e144f82c76e). This page keeps every worked example from that post and then adds the derivations.

## Short version

Softmax turns a vector of raw scores (logits) $z$ into a probability vector. Three facts do most of the work:

1. $e^{z_i}$ is always positive, so every output is a positive share of a pie that sums to 1.
2. Adding the same constant to every score does not change the output (the harsh-judge property). Among well-behaved positive functions, the exponential is the unique choice that does this.
3. Stretching the scores, or dividing them by a temperature $T$, does change the output. Large gaps, or $T \to 0$, push the result toward a one-hot vector. $T \to \infty$ pushes it toward uniform.

The rest of the page works every example with the scores 5, 3, 2, then derives shift invariance, uniqueness, the Jacobian, the softmax + cross-entropy gradient $s - y$, the stable implementation, and the two temperature limits.

## The formula

$$\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

In three steps:

1. Take each score and raise $e$ to that power.
2. Add up all those results.
3. Divide each one by the total.

That’s it. The output is a set of numbers that are all positive and add up to exactly 1 — in other words, percentages.

## Example: three contestants

Three contestants get judged. Their raw scores — in machine learning these raw scores are called **logits** — are:

| Contestant | Score |
|---|---|
| A | 5 |
| B | 3 |
| C | 2 |

We want to turn those into percentages.

## The obvious way: plain share

Add them up: $5 + 3 + 2 = 10$. Divide each by 10. Simple. Intuitive. And as we’ll see in a minute, quietly broken.

| Contestant | Score | Percentage |
|---|---|---|
| A | 5 | 50% |
| B | 3 | 30% |
| C | 2 | 20% |

## The softmax way — step by step

**Step 1: Exponentiate each score.**

| Contestant | Score | $e^{\text{score}}$ |
|---|---|---|
| A | 5 | $e^{5} = 148.41315910$ |
| B | 3 | $e^{3} = 20.08553692$ |
| C | 2 | $e^{2} = 7.38905610$ |

(If you want to follow along: $e$ is about 2.71828. So $e^{5}$ means 2.71828 multiplied by itself 5 times.)

**Step 2: Add them up.**

```
148.41315910 + 20.08553692 + 7.38905610 = 175.88775212
```

**Step 3: Divide each one by the total.**

```
A:  148.41315910 / 175.88775212 = 0.843795  →  84.38%
B:   20.08553692 / 175.88775212 = 0.114195  →  11.42%
C:    7.38905610 / 175.88775212 = 0.042010  →   4.20%
```

Sanity check — they should add to 1:

```
0.843795 + 0.114195 + 0.042010 = 1.000000
```

## Side by side

| Contestant | Score | Plain share | Softmax |
|---|---|---|---|
| A | 5 | 50% | 84.38% |
| B | 3 | 30% | 11.42% |
| C | 2 | 20% | 4.20% |

(The more precise softmax percentages, which appear in the original side-by-side table, are 84.3795%, 11.4195%, and 4.2010%.)

The first thing you notice: **softmax is opinionated.**

Plain share is a democratic accountant — it splits the pie strictly by share. Softmax is a rich-get-richer amplifier. A modest lead of 2 points became a landslide. That’s often exactly what you want from a model that has to commit to an answer.

But that amplification is the *shallow* difference. Here’s the one that actually matters.

## The harsh judge problem

A grumpy judge shows up and docks **1 point from everybody**. New scores: **4, 2, 1**.

Stop and think about what changed in reality. Nothing. Same contestants. Same ranking. Same gaps between them. One judge was in a bad mood, uniformly.

**The percentages should not move.**

### What plain share does

New total: $4 + 2 + 1 = 7$.

```
A: 4/7 = 57.14%
B: 2/7 = 28.57%
C: 1/7 = 14.29%
```

A jumped from 50% to **57.14%**. Because a judge was grumpy. Nothing real changed and the answer moved by 7 percentage points. **That’s broken.**

### What softmax does

**Step 1:** $e^{4} = 54.59815003$, $e^{2} = 7.38905610$, $e^{1} = 2.71828183$

**Step 2:** total = $64.70548796$

**Step 3:**

```
A: 54.59815003 / 64.70548796 = 0.843795  →  84.38%
B:  7.38905610 / 64.70548796 = 0.114195  →  11.42%
C:  2.71828183 / 64.70548796 = 0.042010  →   4.20%
```

**Identical to before. To six decimal places.** Not approximately — exactly.

### Why this happens

Because of one schoolbook rule about exponents:

$$e^{z+c} = e^{z}\,e^{c}$$

Adding a constant $c$ to every score **multiplies every exponentiated value by the same factor** $e^{c}$. And a common factor on the top and bottom of a fraction cancels out. The shift vanishes.

$$\frac{e^{z_i+c}}{\sum_j e^{z_j+c}} = \frac{e^{c}\,e^{z_i}}{e^{c}\sum_j e^{z_j}} = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

A constant added to every logit does not change the probabilities.

In our case, subtracting 1 from everything multiplied every $e^{\text{score}}$ by $e^{-1} \approx 0.3679$. The top and bottom were both scaled by 0.3679 ($e^{-1}$), so each value kept 36.79% of its size (a 63.21% drop) and the ratio didn’t budge.

## And here’s the kicker

**No other simple function does this as well.** Squaring, absolute values and a fixed “shift up until positive” fail the grumpy judge, or they behave badly in other ways. Subtracting the minimum is shift-proof, but it assigns 0% to the last place and is not smooth. The exponential is the natural choice that is smooth, strictly positive *and* shift-proof.

If you write down the requirement (*“adding the same number to every score must not change the output”*) for a rule that transforms each score on its own and then divides by the total, and solve it mathematically, the exponential function is the **unique** answer. It falls out as the only possibility. The proof is in [Why the exponential is the unique choice](#why-the-exponential-is-the-unique-choice) below.

## Why you should care

The scores coming out of a neural network have **no fixed baseline.** Nothing in the training process pins down where “zero” is. The whole layer can drift up or down by some constant and it means absolutely nothing about the prediction.

Softmax is immune to that drift, as the algebraic identity above shows. Plain share would have your model’s confidence wobbling around based on a meaningless number. So the $e^{x}$ isn’t cosmetic. It’s the thing that makes the output trustworthy.

## The flip side: softmax cares how loudly you shout

Plain share only sees **ratios**. Softmax sees **gaps after an exponential**, so stretching the scores changes the output.

**Plain share** (unchanged if you double):

$$\frac{z_i}{\sum_j z_j} \qquad \frac{2z_i}{\sum_j 2z_j} = \frac{2z_i}{2\sum_j z_j} = \frac{z_i}{\sum_j z_j}$$

Now let’s go the other direction. Keep the original scores **5, 3, 2** and **double** them: **10, 6, 4**.

**Softmax** (changes if you double):

$$\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}} \qquad \mathrm{softmax}(2z)_i = \frac{e^{2z_i}}{\sum_j e^{2z_j}} = \frac{\bigl(e^{z_i}\bigr)^{2}}{\sum_j \bigl(e^{z_j}\bigr)^{2}}$$

The 2 does **not** factor out of the sum. You are no longer cancelling a common multiplier; you are squaring each term.

On 5, 3, 2:

$$e^{5} : e^{3} : e^{2} \approx 148 : 20 : 7.4 \;\rightarrow\; 84\%,\; 11\%,\; 4\%$$

On 10, 6, 4:

$$e^{10} : e^{6} : e^{4} \approx 22026 : 403 : 55 \;\rightarrow\; 98\%,\; 1.8\%,\; 0.2\%$$

Same ranking, same ratios of the raw scores, sharper probabilities.

**Calculation for 10, 6, 4**

```
e^10 = 22026.46579
e^6  =   403.42879
e^4  =    54.59815
total = 22484.49273

A: 22026.46579 / 22484.49273  →  97.96%
B:   403.42879 / 22484.49273  →   1.79%
C:    54.59815 / 22484.49273  →   0.24%
```

A went from 84% to **98%**. Same scores, doubled. So the two functions are **exactly inverted**:

| Method | Add a constant | Multiply the scores |
|---|---|---|
| Plain share | changes the answer (bad) ✗ | ignores it (loses information) ✗ |
| Softmax | ignores it (good) ✓ | responds to it (keeps the spread) ✓ |

Softmax throws away the one thing that’s meaningless (the baseline) and pays attention to the one thing that carries information (the spread).

## Temperature in an LLM: that sensitivity is a feature

Softmax’s reaction to scaling isn’t a bug. It’s a **volume knob**, and the knob is called the **temperature**, $T > 0$. You divide every score by $T$ before exponentiating:

$$\mathrm{softmax}(z/T)_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

$T = 1$ is plain softmax. A $T$ below 1 stretches the scores apart and sharpens the output. A $T$ above 1 squeezes them together and flattens it. The doubling example above was exactly $T = \tfrac{1}{2}$, because dividing by $\tfrac{1}{2}$ is the same as multiplying by 2.

While querying an LLM you pass this same knob as a parameter. The model still emits one logit vector $z$ for the next token, and the API divides by $T$ and softmaxes before it samples. In the table below, $T = 0.1$ is the ice-cold row and $T = 2$ is the hedge. You never changed the scores, only how sharp they become.

Let’s turn the knob on our original **5, 3, 2**. The $T = 2$ row is the least obvious, so here it is worked out:

```
Scores divided by 2:  2.5, 1.5, 1.0
e^2.5 = 12.18249396
e^1.5 =  4.48168907
e^1.0 =  2.71828183
total = 19.38246486

A: 12.18249396 / 19.38246486 = 0.628532  →  62.85%
B:  4.48168907 / 19.38246486 = 0.231224  →  23.12%
C:  2.71828183 / 19.38246486 = 0.140244  →  14.02%
```

Sum: $0.628532 + 0.231224 + 0.140244 = 1.000000$ ✓

**This is the same temperature setting you’ve adjusted in an LLM API.** Crank it up and the model divides its scores down, flattening the distribution, so unlikely words get a real chance — output feels creative or unhinged. Turn it down and the distribution sharpens toward the single most likely token — output feels safe and repetitive. Same model, same scores, different personality, entirely because softmax is scale-sensitive.

Same scores, different personalities:

| Temperature | Label | Scores ÷ $T$ | Probabilities | Personality |
|---|---|---|---|---|
| 0.1 | ice cold | [50, 30, 20] | A: 99.9999997939%, B: $2.061153618 \times 10^{-7}$%, C: $9.35762295 \times 10^{-12}$% | Dictator |
| 0.5 | cold | [10, 6, 4] | A: 97.96%, B: 1.79%, C: 0.24% | Very confident |
| 1 | default | [5, 3, 2] | A: 84.38%, B: 11.42%, C: 4.20% | Confident |
| 2 | warm | [2.5, 1.5, 1] | A: 62.85%, B: 23.12%, C: 14.02% | Hedging |
| $\infty$ | boiling | [0, 0, 0] | A: 33.33%, B: 33.33%, C: 33.33% | Total shrug |

At $T \to \infty$, every score divided by $T$ goes to 0, so the three options become equal. The B and C values at $T = 0.1$ are percentages too — $2.061 \times 10^{-7}$ percent is a probability of $2.061 \times 10^{-9}$. A has essentially all of the mass.

## Self-attention: the same volume knob

If you’ve read the attention formula, you’ve seen a mysterious division:

$$\mathrm{Attention} = \mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$

That $\sqrt{d_k}$ is **exactly this volume knob.** Raw attention scores come out too loud, and a too-loud softmax becomes a dictator: one option sits at 99.99% and everything else is effectively zero. When that happens there’s almost no useful feedback signal left for the network to learn from — training stalls. The division is a volume limiter, keeping softmax in the range where it still responds to changes.

If softmax really were “just normalization,” that $\sqrt{d_k}$ would do nothing at all. It would cancel out. The fact that it’s load-bearing is proof the $e^{x}$ matters.

How the scores $QK^{\top}$ are built, and what $V$ does after the softmax, belongs in [self-attention](../self-attention.md). The only fact needed here is that $\sqrt{d_k}$ is a temperature.

## Softmax can never say “none of the above”

Every softmax output is **strictly greater than zero.** Always. Because $e^{x}$ is never zero, no matter how negative $x$ gets.

A probability can be one in a trillion. It can never be *actually* zero.

That holds for finite logits, in exact arithmetic. In float32, `exp` underflows: a logit gap of about 100 already gives roughly $4 \times 10^{-44}$, and a gap of about 104 or more gives exactly `0.0`. Masked attention positions are set to $-\infty$ on purpose, so they get an exact 0 too.

Usually harmless. Occasionally it causes something genuinely weird. Inside a Transformer, an attention head that wants to say *“nothing in this sentence is relevant to me right now”* **has no way to express that.** It is forced to hand out 100% of its attention somewhere.

*The published Medium post, and the author’s RSS feed, both end at the sentence above. The next two paragraphs close the section; they were not in the original post.*

That “must look at something” property is why later designs add an extra sink token — a padded slot the head can dump unused attention into — or replace softmax with a sigmoid, which does not have to sum to 1. Softmax itself has no off switch: it always distributes the whole mass.

The rest of this page is the calculus the blog post left out: why the exponential is the unique shift-invariant choice, how the probabilities change when a logit moves, why softmax and cross-entropy simplify to a subtraction, how to compute softmax without overflowing, and what happens at the two ends of the temperature knob.

---

## Calculus and identities

Every derivation below uses the same three scores $z = [5, 3, 2]$, so you can check the algebra against a number you already know: $s = [0.843795,\; 0.114195,\; 0.042010]$.

### Shift invariance, formally

Write $s_i(z) = e^{z_i} / \sum_j e^{z_j}$. Replace every score by $z_j + c$:

$$
s_i(z + c\cdot\mathbf{1})
= \frac{e^{z_i+c}}{\sum_j e^{z_j+c}}
= \frac{e^{c}\,e^{z_i}}{e^{c}\sum_j e^{z_j}}
= \frac{e^{z_i}}{\sum_j e^{z_j}}
= s_i(z).
$$

The factor $e^{c}$ is the same in every term, so it cancels. This is the harsh-judge calculation with the constant $c$ left unspecified. In the example, $c = -1$ and $e^{c} = e^{-1} \approx 0.367879$, and the probabilities did not move at all.

The same identity is what makes the numerically stable implementation in a later section legal: subtracting $\max(z)$ is just a particular choice of $c$, so it cannot change the answer.

### Why the exponential is the unique choice

Suppose we want a converter of the form

$$s_i(z) = \frac{f(z_i)}{\sum_j f(z_j)}$$

where $f$ is always positive, and we require shift invariance:

$$s_i(z + c\cdot\mathbf{1}) = s_i(z) \quad \text{for every score vector } z \text{ and every constant } c.$$

Then $f(z_i+c)/\sum_j f(z_j+c)$ equals $f(z_i)/\sum_j f(z_j)$. The vector $\bigl(f(z_1+c),\ldots,f(z_n+c)\bigr)$ must be a common multiple of $\bigl(f(z_1),\ldots,f(z_n)\bigr)$. Taking ratios, $f(z_i+c)/f(z_i)$ is the same for every $i$. The ratio $f(x+c)/f(x)$ cannot depend on $x$. Call it $g(c)$:

$$f(x+c) = g(c)\,f(x).$$

This is a Cauchy-type functional equation. Set $x = 0$: $f(c) = g(c)\,f(0)$. Since $f$ is always positive, $f(0) \neq 0$, and $g(c) = f(c)/f(0)$. Then

$$f(x+c) = \frac{f(x)\,f(c)}{f(0)}.$$

Let $h(x) = f(x)/f(0)$. Then $h$ is positive and satisfies $h(x+c) = h(x)\,h(c)$.

If we also assume $f$ is continuous — or even just continuous at one point, or monotonic; any of these is enough — the solutions are $h(x) = e^{kx}$ for some constant $k$. (Without some regularity there are pathological solutions that nobody uses.) Hence

$$f(x) = A\,e^{kx}, \qquad A = f(0) > 0.$$

Plug this $f$ back into the converter. The constant $A$ cancels, and $k$ just rescales the scores: you get ordinary softmax of $kz$, which is $\mathrm{softmax}(z/T)$ with $T = 1/k$ when $k > 0$.

So the only shift-invariant normalized maps of this form are softmax with a temperature. If $k = 0$, $f$ is constant and you get the uniform distribution, which is the $T \to \infty$ limit. If $k < 0$, the ranking reverses.

That is the precise meaning of “no other simple function does this as well.” Squaring, absolute values, and a fixed “shift up until positive, then divide” all fail the equation $f(x+c) = g(c)\,f(x)$ for a $g$ that does not depend on $x$. Subtracting the minimum escapes this argument only because it is not a per-score rule: it looks at all the scores first. It is shift-proof (on 5, 3, 2 and on 4, 2, 1 it gives 75%, 25%, 0% both times), but it hands last place exactly 0% and has a kink wherever the minimum changes. So the exponential is the natural smooth, strictly positive, shift-proof choice.

### The Jacobian

We want every partial derivative $\partial s_i / \partial z_j$: how probability $i$ moves when logit $j$ moves.

Write $S = \sum_k e^{z_k}$ and $s_i = e^{z_i}/S$. Differentiate with the quotient rule.

When $i = j$, the numerator and the sum both depend on $z_i$:

$$
\frac{\partial s_i}{\partial z_i}
= \frac{e^{z_i}\,S - e^{z_i}\,e^{z_i}}{S^{2}}
= \frac{e^{z_i}}{S}\left(1 - \frac{e^{z_i}}{S}\right)
= s_i(1 - s_i).
$$

When $i \neq j$, only the sum in the denominator depends on $z_j$:

$$
\frac{\partial s_i}{\partial z_j}
= \frac{0\cdot S - e^{z_i}\,e^{z_j}}{S^{2}}
= -s_i s_j.
$$

The two cases together are one formula:

$$\frac{\partial s_i}{\partial z_j} = s_i\bigl(\delta_{ij} - s_j\bigr)$$

where $\delta_{ij} = 1$ if $i = j$ and $0$ otherwise. The $n \times n$ matrix of these entries is the Jacobian $J$, with $J_{ij} = \partial s_i / \partial z_j$.

Now the numbers, using $s_A = 0.843795$, $s_B = 0.114195$, $s_C = 0.042010$. Each diagonal entry is $s_i(1-s_i)$. Each off-diagonal entry is $-s_i s_j$.

|  | $\partial/\partial z_A$ | $\partial/\partial z_B$ | $\partial/\partial z_C$ |
|---|---|---|---|
| $\partial s_A$ | $0.843795 \times (1-0.843795) = 0.131805$ | $-0.843795 \times 0.114195 = -0.096357$ | $-0.843795 \times 0.042010 = -0.035448$ |
| $\partial s_B$ | $-0.114195 \times 0.843795 = -0.096357$ | $0.114195 \times (1-0.114195) = 0.101155$ | $-0.114195 \times 0.042010 = -0.004797$ |
| $\partial s_C$ | $-0.042010 \times 0.843795 = -0.035448$ | $-0.042010 \times 0.114195 = -0.004797$ | $0.042010 \times (1-0.042010) = 0.040245$ |

As a matrix, rounded to six decimal places:

$$
J = \begin{bmatrix}
 0.131805 & -0.096357 & -0.035448 \\
-0.096357 &  0.101155 & -0.004797 \\
-0.035448 & -0.004797 &  0.040245
\end{bmatrix}
$$

Two checks you can do by eye:

- $J$ is symmetric, because $J_{ij} = -s_i s_j = J_{ji}$ off the diagonal.
- Every row sums to 0, and every column sums to 0. That has to be true: the probabilities always add to 1, so if you nudge any one logit, the changes in $s_A$, $s_B$, and $s_C$ must cancel. For the first row: $0.131805 - 0.096357 - 0.035448 = 0$.

The diagonal is positive and the off-diagonal is negative: raising $z_A$ increases $s_A$ and decreases $s_B$ and $s_C$. And $J_{AA} = 0.131805$ is modest, not huge, because A is already at 84%. There is only 16% of the pie left to take. If one probability is already near 1, its diagonal entry $s_i(1-s_i)$ is near 0 — that is the “dictator” regime, where softmax has almost stopped responding.

### Softmax and cross-entropy: the gradient is $s - y$

The usual training loss for a classification (or next-token) problem is cross-entropy between the softmax output $s$ and a target distribution $y$ (often a one-hot vector for the correct class):

$$L = -\sum_k y_k \log s_k, \qquad \sum_k y_k = 1, \quad y_k \ge 0.$$

We want $\partial L / \partial z_j$, the slope of the loss with respect to each logit. First, $\partial L / \partial s_i = -y_i / s_i$. Then the chain rule, using the Jacobian from the previous section:

$$
\begin{align*}
\frac{\partial L}{\partial z_j}
&= \sum_i \frac{\partial L}{\partial s_i}\,\frac{\partial s_i}{\partial z_j} \\
&= \sum_i \left(-\frac{y_i}{s_i}\right) s_i(\delta_{ij} - s_j) \\
&= \sum_i -y_i(\delta_{ij} - s_j) \\
&= -y_j + s_j\sum_i y_i \\
&= s_j - y_j.
\end{align*}
$$

The $s_i$ in the numerator and denominator cancel, and $\sum_i y_i = 1$. The whole backpropagation through softmax and cross-entropy collapses to a subtraction:

$$\frac{\partial L}{\partial z} = s - y.$$

**Numeric example.** Take $z = [5, 3, 2]$ so $s = [0.843795,\; 0.114195,\; 0.042010]$, and suppose A is the correct answer: $y = [1,\; 0,\; 0]$. The loss is $L = -\log s_A = -\log 0.843795 \approx 0.169846$. The gradient is

$$
s - y = [0.843795-1,\; 0.114195-0,\; 0.042010-0] = [-0.156205,\; 0.114195,\; 0.042010].
$$

| Entry | Value | What it means for a gradient step |
|---|---|---|
| $\partial L/\partial z_A$ | $-0.156205$ | To reduce the loss, *increase* $z_A$. |
| $\partial L/\partial z_B$ | $0.114195$ | To reduce the loss, *decrease* $z_B$. |
| $\partial L/\partial z_C$ | $0.042010$ | To reduce the loss, *decrease* $z_C$. |

The three entries still sum to 0, because $s$ and $y$ both sum to 1. The model is being asked to move mass from B and C onto A. Notice that $\partial L/\partial z_B$ equals $s_B$ itself when $y_B = 0$: a wrong class is penalized in proportion to the probability the model currently assigns it.

### Computing softmax without overflow

A direct $e^{z_i}$ overflows in ordinary 64-bit floats once $z_i$ is around 710 or larger. The same thing happens in 32-bit floats much earlier, around 88. Neural-network logits can cross that line.

Shift invariance is the fix. Subtract the maximum score first. Every exponent is then $\le 0$, the largest term is $e^{0} = 1$, and nothing overflows:

$$
\mathrm{softmax}(z)_i
= \frac{e^{z_i - m}}{\sum_j e^{z_j - m}}, \qquad m = \max_k z_k.
$$

This is the same identity as the harsh-judge proof, with $c = -m$. Equivalently, people work with the log-sum-exp form

$$\log s_i = (z_i - m) - \log\sum_j e^{z_j - m}$$

when they need log-probabilities (for example, inside a cross-entropy loss).

```python
import numpy as np

def softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - z.max()          # every exponent is now <= 0
    e = np.exp(z)
    return e / e.sum()

def log_softmax(z):
    z = np.asarray(z, dtype=float)
    m = z.max()
    return z - m - np.log(np.exp(z - m).sum())

# The two inputs differ by 1000; the probabilities are identical.
print(softmax([5, 3, 2]))
print(softmax([1005, 1003, 1002]))
# [0.84379473 0.1141952  0.04201007]
# [0.84379473 0.1141952  0.04201007]

# The naive formula overflows on the second input:
# np.exp([1005, 1003, 1002])  →  [inf, inf, inf]
```

### The two ends of the temperature knob

$$\mathrm{softmax}(z/T)_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

**As $T \to 0$.** The largest score dominates. If $z_{i^*}$ is the unique maximum, then $z_{i^*}/T$ grows faster than every other $z_j/T$, and

$$\mathrm{softmax}(z/T) \;\longrightarrow\; \text{one-hot at } \arg\max(z).$$

That is the ice-cold “dictator” row of the table, taken all the way. In an LLM API, $T = 0$ is usually implemented as greedy decoding: just pick the argmax, skip softmax.

If two scores are tied for the maximum, the mass splits equally among the ties, and everyone else still goes to 0. For a quick check: $\mathrm{softmax}([5, 5, 2]/T)$ goes to $[0.5,\; 0.5,\; 0]$ as $T \to 0$.

**As $T \to \infty$.** Every $z_i/T$ goes to 0, and $e^{0} = 1$, so

$$\mathrm{softmax}(z/T) \;\longrightarrow\; \left(\tfrac{1}{n},\;\tfrac{1}{n},\;\ldots,\;\tfrac{1}{n}\right).$$

For our three contestants that is $[1/3,\; 1/3,\; 1/3]$, the “total shrug” row.

A couple of extra values on the same $[5, 3, 2]$, so the two limits are visible as numbers, not just statements:

| $T$ | Probabilities | What you are looking at |
|---|---|---|
| 0.01 | A: $\approx 100$%, B and C: $\approx 0$ | essentially $\arg\max$ |
| 0.1 | A: 99.9999997939%, B: $2.061 \times 10^{-7}$%, C: $9.358 \times 10^{-12}$% | dictator |
| 1 | A: 84.38%, B: 11.42%, C: 4.20% | default softmax |
| 100 | A: 33.89%, B: 33.22%, C: 32.89% | already close to uniform |
| $\infty$ | A: 33.33%, B: 33.33%, C: 33.33% | uniform |

---

*Source: [Softmax function](https://medium.com/@analog76/softmax-function-3e144f82c76e) by Anandha L Ranganathan, 23 September 2026. Tables and formulas that were images in the Medium post have been rebuilt as Markdown and LaTeX. Numbers were recomputed in Python. The calculus section, and the two closing paragraphs of “none of the above,” were added for this foundations page.*
