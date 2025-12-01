import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import re
    from pathlib import Path
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    script_folder = Path(__file__).parent
    dotenv_path = script_folder / ".env"
    load_dotenv(dotenv_path=dotenv_path)

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    client = genai.Client(api_key=GEMINI_API_KEY)
    return SmoothingFunction, client, re, sentence_bleu, types


@app.cell
def _():
    text_to_translate = """
    Cilj rada "Empirijska studija postupaka strojnog učenja za prepoznavanje malicioznih napada"
    Antonia Carevića i Maria Dudjaka s FERIT-a je definirati tok učenja algoritama klasifikacije
    iz skupova podataka koji opisuju različite vrste malicioznih napada i odrediti pojedinačne
    procedure unutar tog toka. Primjenom definiranog toka dobiveni su rezultati koji pokazuju
    visoku kvalitetu klasifikacijskih modela u prepoznavanju malicioznih napada, što potvrđuje
    njegovu primjenjivost u području kibernetičke sigurnosti, posebno u sustavima za detekciju upada.

    Korišteni algoritmi strojnog učenja su: naivni Bayesov algoritam, k-najbližih susjeda,
    stablo odluke, nasumična šuma i logistička regresija. Tijekom odabira značajki korišteni
    su filtri s Pearsonovim koeficijentom korelacije, zajedničkom informacijom i ANOVA
    F-vrijednosti te omotač slijedna pretraga unaprijed.

    Za obradu neuravnoteženih skupova podataka primijenjeni su postupci nasumičnog
    preuzorkovanja i poduzorkovanja. Najbolje rezultate postigao je algoritam stablo odluke
    s F1 mjerom od 1.0 na većini skupova podataka, dok je naivni Bayesov algoritam imao
    znatno slabije performanse, s F1 vrijednostima u rasponu od 0.12 do 0.98.

    Tehnike odabira značajki uglavnom su poboljšale performanse, pri čemu se posebno
    istaknuo omotač. Među postupcima za smanjenje neuravnoteženosti podataka,
    nasumično preuzorkovanje dosljedno je poboljšalo performanse svih algoritama,
    dok je poduzorkovanje dovelo do značajnog smanjenja performansi kod pojedinih
    algoritama, uz pad F1 mjere i do 0.22.

    Predloženi tok učenja omogućuje sustavno vrednovanje utjecaja različitih metoda
    predobrade podataka i algoritama klasifikacije, čime doprinosi boljem razumijevanju
    procesa detekcije malicioznih napada u neuravnoteženim i heterogenim podatkovnim
    skupovima te može poslužiti kao temelj za razvoj učinkovitijih sustava kibernetičke
    obrane u stvarnim okruženjima.
    """

    initial_prompt = """
    You are an expert Croatian-to-English translator.
    You translate highly technical AI/ML/cybersecurity documents.
    Maintain correct terminology (e.g., "funkcija"="function", "model"="model",
    "skup podataka"="dataset", "preuzorkovanje"="resampling").
    Aim for BLEU-like precision and terminological consistency.
    Preserve sentence structure unless simplification improves clarity.
    """
    return initial_prompt, text_to_translate


@app.cell
def _(SmoothingFunction, client, re, sentence_bleu, types):
    def run_student(system_prompt, text):
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            ),
            contents=text
        )
        return response.text

    def run_judge(system_prompt, input_text, output_text):
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=f"""
    You are a prompt optimization judge.

    Your task:
    - Evaluate Flash-Lite translation quality
    - Identify errors
    - Produce a BETTER system prompt

    Evaluation criteria:
    - BLEU precision
    - Terminology fidelity (AI/ML/cybersecurity)
    - Sentence faithfulness
    - Logical consistency
    - Handling long technical sentences

    Respond in format:

    ERRORS:
    - bullet points

    NEW_PROMPT:
    <<<
    (new system prompt here)
    >>>

    REASONING:
    (one paragraph)
    """
        )
        return response.text
    
    def extract_new_prompt(judge_response):
        pattern = r"NEW_PROMPT:\s*<<<(.*?)>>>"
        match = re.search(pattern, judge_response, re.DOTALL)
        if not match:
            raise ValueError("Judge output did not contain NEW_PROMPT block.")
        return match.group(1).strip()

    def compute_bleu(reference, hypothesis):
        smoothie = SmoothingFunction().method4
        ref_tokens = [reference.split()]
        hyp_tokens = hypothesis.split()
        return sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothie)
    return compute_bleu, extract_new_prompt, run_judge, run_student


@app.cell
def _(
    compute_bleu,
    extract_new_prompt,
    initial_prompt,
    run_judge,
    run_student,
    text_to_translate,
):
    MAX_ITERATIONS = 10
    MIN_BLEU_IMPROVEMENT = 0.005
    current_prompt = initial_prompt
    previous_translation = ""
    previous_bleu = 0.0

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n====================== ITERATION {iteration} ======================")

        translation = run_student(current_prompt, text_to_translate)
        print("\nFLASH-LITE TRANSLATION (preview):\n", translation[:500], "...")

        if previous_translation:
            bleu = compute_bleu(previous_translation, translation)
            improvement = bleu - previous_bleu
            print(f"\nBLEU improvement over previous: {improvement:.4f}")
            if improvement < MIN_BLEU_IMPROVEMENT:
                print("Improvement below threshold. Stopping iterations.")
                break
        else:
            bleu = compute_bleu(text_to_translate, translation)
            improvement = bleu
            print(f"\nInitial BLEU score: {bleu:.4f}")

        previous_translation = translation
        previous_bleu = bleu

        judge_output = run_judge(current_prompt, text_to_translate, translation)
        print("\nJUDGE OUTPUT (preview):\n", judge_output[:500], "...")

        try:
            new_prompt = extract_new_prompt(judge_output)
        except ValueError:
            print("Could not extract NEW_PROMPT. Stopping iterations.")
            break

        print("\nNEW PROMPT (preview):\n", new_prompt[:500], "...")
        current_prompt = new_prompt

    print("\n====================== FINAL OPTIMIZED PROMPT ======================")
    print(current_prompt)
    return


if __name__ == "__main__":
    app.run()
