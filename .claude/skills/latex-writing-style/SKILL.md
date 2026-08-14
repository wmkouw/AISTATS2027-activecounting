---
name: latex-writing-style
description: Expert LaTeX writing assistant following academic conventions for mathematical papers. Use when writing or editing LaTeX files, managing citations, formatting equations, or structuring academic documents.
---

# LaTeX Paper Writing Style

Expert LaTeX writing assistant following academic conventions for mathematical papers.

## References and Citations

- Never start a sentence with a citation in parentheses
- Use `\label{}` immediately after `\section`, `\subsection`, `\begin{equation}`, `\begin{figure}`, `\begin{table}`, etc.

## Label Naming Conventions

- Sections: `sec:name` (e.g., `sec:introduction`)
- Equations: `eq:name` (e.g., `eq:vfe`)
- Figures: `fig:name` (e.g., `fig:factor_graph`)
- Tables: `tab:name` (e.g., `tab:results`)
- Theorems: `thm:name` (e.g., `thm:main`)
- Lemmas: `lem:name` (e.g., `lem:helper`)
- Appendix sections: `appx:name` (e.g., `appx:proofs`)

## Math Notation

- Use `\bm{x}` for bold sequences of random variables, e.g., `\bm{x} = x_{k+1:T}`
- Individual time-indexed variables are not bolded: `x_t`, `u_t`, `y_t`
- Order variables by rate of change (fastest first): observations, states, controls, parameters
  - Example: `p(\bm{y}, \bm{x}, \bm{u}, \theta)` or `q(\bm{y}, \bm{x}, \theta | \bm{u})`
- Use `\mathbb{E}_{q}[ ]` for expectations.
- Use `\mathbb{V}_{q}[ ]` for variances and `\mathbb{C}_{q}[ ]` for covariances.
- Use `D_{\text{KL}}[q( ) \| p( )]` for KL divergences.
- Use `\mathcal{F}[q]` for a standard free energy functional.
- Use `\mathcal{B}[q]` for a Bethe free energy functional.
- Use `\mathcal{L}[q]` for a Lagrangian.
- Use `\mathcal{G}[q]` for an expected free energy functional.
- Use `\mathcal{D}` for collections of observed data, with a subscript if time is involved, e.g., `\mathcal{D}_t`.
- Use `G(u)` for an expected free energy function.
- Use `\dif` for differentials: `\int f(x) \dif x`.
- Use `\text{}` for text within math mode, not `\mathrm{}` for words.
- Subscripts that are words should use `\text{}`: `D_{\text{KL}}` not `D_{KL}`.
- Use `[]` for functionals and `()` for functions.
- Use `\given` for conditioning, where it is defined as `\newcommand\given[1][]{\,#1\vert\,}`.
- Always write `p(x) = ..` followed by the parametric distribution, e.g., `\mathcal{N}(x \given \mu, \Sigma)`.
- The tilde notation, e.g., `x ~ \mathcal{N}(\mu, \Sigma)`, is reserved for sampling. 
- Only use `\tfrac` in inline equations.

### Moment operators name their distribution in the subscript

The distribution an expectation, variance or covariance is taken over belongs in the
subscript, never inside the brackets. `\mathrm{Var}[y \given u, \mathcal{D}]` is ambiguous:
the bar reads as conditioning on the bracketed expression rather than as naming the law.
Write the law out as a subscript and put only the quantity in the brackets:

```latex
\providecommand{\Ex}[2]{\mathbb{E}_{#1}\!\left[#2\right]}
\providecommand{\Var}[2]{\mathbb{V}_{#1}\!\left[#2\right]}
\providecommand{\Cov}[2]{\mathbb{C}_{#1}\!\left[#2\right]}
```

- ✓ `\Var{p(y \given u, \mathcal{D})}{y}` renders as `V_{p(y|u,D)}[y]`
- ✗ `\mathrm{Var}[y \given u, \mathcal{D}]`
- ✗ `\Ex{}{y \given \lambda, u}` — an empty subscript with the conditioning smuggled inside

This nests correctly, so the law of total variance reads
`\Ex{p(\theta \given \mathcal{D})}{\Var{p(y \given \theta, u)}{y}} + \Var{p(\theta \given
\mathcal{D})}{\Ex{p(y \given \theta, u)}{y}}`.

### Parametric distributions

Name parametric distributions with a single `\mathcal` letter, never with a spelled-out
name. Write `\mathcal{P}(y \given \lambda)`, not `\mathrm{Poisson}(y; \lambda)` or
`\text{Poisson}(y \given \lambda)`. The conditioning bar is `\given`; a semicolon is not
a substitute for it.

- `\mathcal{N}` for the Gaussian (normal)
- `\mathcal{G}` for the gamma
- `\mathcal{P}` for the Poisson
- `\mathcal{B}` for the Bernoulli, and for the binomial that generalises it
- `\mathcal{T}` for the Student-`t`

Where one letter will not do, add a second, upright one rather than inventing a letter:
`\Bet` for the beta, `\Bpr` for the beta-prime, `\NBin` for the negative binomial, `\Expo`
for the exponential.

Introduce the letters in text at first use, so the reader is never asked to infer them
from context.

**`\mathcal` is only defined for uppercase letters.** `\mathcal{Be}` does not render as
"Be"; it produces a stray glyph (`B]`), and `\mathcal{Ex}` produces `E§`. Two-letter names
must therefore be built as a calligraphic capital followed by an upright lowercase, and are
worth a macro:

```latex
\providecommand{\Bet}{\mathcal{B}\mathrm{e}}   % beta
\providecommand{\Bpr}{\mathcal{B}\mathrm{p}}   % beta-prime
\providecommand{\NBin}{\mathcal{N}\mathrm{B}}  % negative binomial
\providecommand{\Expo}{\mathcal{E}\mathrm{x}}  % exponential
```

Two notes on collisions. These symbols are *functions* and take `()`, whereas
`\mathcal{B}[q]` and `\mathcal{G}[q]` above are *functionals* and take `[]`, so the
bracket type carries the distinction. And `\mathcal{D}` is reserved for observed data, so
it is not available for a distribution.

## Equations

- Use `equation` environment for single important equations
- Use `align` for multi-line derivations (not `eqnarray`)
- Use `split` inside `equation` to break long single equations
- Add `\label{}` to equations that will be referenced

### Equation Punctuation

Displayed equations are part of the sentence and must have appropriate punctuation:

- End with a **period** if the equation concludes a sentence
- End with a **comma** if the sentence continues after the equation (e.g., "where...")
- End with **no punctuation only** if followed by conditions on the same line (e.g., `\quad \text{if ...}`)
- In multi-line `align` environments, add punctuation to **each line** that ends a clause:
  - Intermediate lines typically end with `,` (comma)
  - The final line ends with `.` (period) or `,` if text follows
- Use `\,,` or `\,.` for proper spacing before punctuation in display math
- Example:
  ```latex
  \begin{align}
      F[q] &= \text{term}_1\,, \label{eq:first} \\
      &= \text{term}_2\,. \label{eq:second}
  \end{align}
  ```

## Figures and Tables

- Figures: caption goes below (`\caption{}` after `\includegraphics`)
- Tables: caption goes above (`\caption{}` before `\begin{tabular}`)
- Use `\centering` not `\begin{center}...\end{center}` inside floats
- Use `booktabs` package: `\toprule`, `\midrule`, `\bottomrule` (no vertical lines)
- Use `table*` or `figure*` for two-column spanning floats

## Writing Style

- Vary sentence structure.
- Define acronyms on first use: "Variational Free Energy (VFE)"
- Use "we" for actions taken in the paper, "the reader" or passive voice for general statements
- Prefer active voice when clarity permits
- Keep paragraphs focused on one idea
- Use `\emph{}` for emphasis and introducing terms, not bold
- Avoid the use of the em-dash as it reveals LLM-generated text

## Capitalization Conventions

### Named Quantities and Principles (Always Capitalize)
These are specific named technical terms in the literature:
- **Free Energy Principle** (FEP) - the theoretical framework
- **Expected Free Energy** (EFE) - the scoring of policies
- **Kullback-Leibler** (KL) divergence - proper name

### General Techniques and Methods (Always Lowercase)
These are general methodological terms, not specific named quantities:
- **variational inference** - the general technique/method
- **planning as inference** - the general approach/framework (unless starting a sentence)
- **active inference** - the general field/framework (same treatment as reinforcement learning)
- **reinforcement learning** - general field
- **optimal control** - general field
- **message passing** - general technique

### Examples of context-dependent usage
- ✓ "We minimize the variational free energy functional $F[q]$"
- ✓ "The Expected Free Energy combines instrumental and epistemic value"
- ✓ "Planning is performed using variational inference"
- ✓ "We use planning as inference to solve this problem"
- ✗ "We use Variational Inference to minimize the cost" (should be lowercase)
- ✗ "The expected free energy is defined..." (should be capitalized when referring to the EFE)

## Structure

- Each `\section` and `\subsection` should have a `\label`
- Use `%` comments to mark section boundaries and TODOs
- Keep one sentence per line for easier git diffs
- Use consistent indentation (2 or 4 spaces)

## Common Mistakes to Avoid

- Don't use `\def`; use `\newcommand`
- Don't load conflicting packages
- Don't hardcode spacing; let LaTeX handle it
- Don't use `\\ ` for paragraph breaks; use blank lines
- Don't use `\left[` and `\right'`. Use `\Big[` and `\Big]` or other scale variants (`\big`, `\small`, etc.).
- Use British spelling over American.
- Don't use em-dashes (—) as they reveal LLM-generated text; use commas or parentheses instead
- Never change the `references.bib` file directly; prompt the user to add them to their Zotero library. 
- Do not use `\!` before or after parentheses or braces.
