---
name: use-claude-todonote-for-comments
description: In this repo's LaTeX, leave comments as \claude{...} todonotes rather than as % comments, so they render in the draft.
metadata:
  type: feedback
---

Wouter added a `todonotes` setup to `main.tex` on 2026-08-14 and asked that my comments go
through it:

```latex
\usepackage[disable]{todonotes}     % swap for \usepackage{todonotes} to see them
\newcommand{\claude}[2][]{\todo[inline,backgroundcolor=blue!20!white, #1]{(Claude) #2}}
```

Use `\claude{...}` in the body for anything I would otherwise have written as a `% TODO`:
open questions, missing citations, caveats cut from the prose, places where a claim outruns
the evidence.

**Why:** a `%` comment is invisible in the compiled draft, so it only reaches a co-author who
happens to open the source at the right line. A todonote is visible when the package is
enabled and disappears when it is not, which is what a draft annotation should do.

**How to apply:** prefer `\claude{}` over `%` for anything addressed to a reader. Keep `%`
only for notes about the source itself, such as why a table is laid out a particular way.
Do not enable the package; it is disabled deliberately so the draft compiles clean, and
turning it on is the author's choice.
