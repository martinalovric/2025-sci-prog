# Class Exercise: Prompt Optimization
## Using Gemini 2.5 Pro to Optimize Prompts for 2.5 Flash-Lite

---

## The Setup: Judge vs. Student

**Goal:** Use expensive model to optimize prompts for cheap model.

| Role | Model | Cost | Strength |
|------|-------|------|----------|
| **Judge** | Gemini 2.5 Pro | $4.50 per 1M | Complex reasoning |
| **Student** | Gemini 2.5 Flash-Lite | $0.10/$0.40 per 1M | Fastest, cheapest |

> **Economics:** Pay once to optimize $\to$ Save on every inference forever.

---

## The Workflow

```
1. Define task + evaluation criteria
2. Write initial system prompt for Flash-Lite
3. Run Flash-Lite on test cases
4. Gemini 3 Pro analyzes failures
5. Gemini 3 Pro rewrites the prompt
6. Repeat until convergence
```

> **Key Insight:** Flash-Lite is 30-120x cheaper. 10% quality boost = massive ROI at scale.

---

## Example Task for Optimization

**Croatian-English Translation**
Translate technical docs. Eval: BLEU + terminology ("funkcija" $\to$ "function").

## Running the app
1. Create a .env file in `/students/06/rkrstic/README.md` and add the following variable:
   ```
   GEMINI_API_KEY=your-gemini-api-key-here
   ```
2. Run the following command:
   ```
   marimo run ./students/06/rkrstic/prompt_optimization.py
   ```
3. What happens next:
    - The script will iteratively translate the Croatian technical text to English.
    - Each iteration improves terminology, structure, and technical accuracy.
    - Iterations continue until the translation stabilizes.
