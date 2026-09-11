"""Build the Phase 2 pilot plan and print 5 complete prompts exactly as they will be sent.
NO API CALLS ARE MADE HERE.

Prompt caching analysis was removed with the move off the Anthropic API: neither the Gemini
free tier nor Groq exposes an equivalent explicit-breakpoint cache to analyse, and nothing is
being purchased, so there is no cost model to report either. Run-sizing and wall-clock
projection now live in src/07_run.py --status, which reads the live provider config.

Outputs: results/06_pilot_prompts.txt   (5 full prompts, verbatim)
         results/06_pilot_plan.csv       (the 60 pilot calls)
         results/06_token_profile.txt
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
import dices_io as D, personas as P, prompts as PR, phase2_config as CFG

CHARS_PER_TOKEN = 3.6        # heuristic; labelled an estimate wherever it is used
PILOT_SEED = 20260826


def est_tokens(s):
    return len(s) / CHARS_PER_TOKEN


def pilot_plan():
    """The 60 pilot calls: 15 items x 4 conditions.

    Composition adapted at Checkpoint 2 (item 4). The brief's 20 items x 3 personas cannot
    answer the question that "N distinct personas per cell" creates: do two DISTINCT personas
    inside the SAME cell differ? If they do not, a cell collapses to one persona. So the 60
    calls are re-spent as:
       Asian-Woman#1, Asian-Woman#2   two distinct personas, SAME cell  -> within-cell
       White-Man#1                    the most distant cell from it     -> between-cell
       baseline#1                     no demographics at all            -> persona effect
    Asian-Woman and White-Man are the most distant human pair in the chosen scheme (mean
    pairwise TVD 0.368, results/05_cell_selection.txt), so it is the contrast most likely to
    show an effect. If the model cannot separate those two it will separate nothing.

    Items are stratified: 5 per unfiltered-entropy tertile, so the pilot spans uncontested
    and contested conversations.
    """
    ent = pd.read_csv("results/03_item_jsd.csv")[["item_id", "entropy_all", "ent_bin"]]
    rng = np.random.default_rng(PILOT_SEED)
    picks = []
    for b in ["low", "medium", "high"]:
        picks += sorted(rng.choice(ent[ent.ent_bin == b].item_id.to_numpy(), 5,
                                   replace=False).tolist())
    conds = [("Asian-Woman#1", "persona", "Asian-Woman", 0),
             ("Asian-Woman#2", "persona", "Asian-Woman", 1),
             ("White-Man#1", "persona", "White-Man", 0),
             ("baseline#1", "baseline", None, 0)]
    rows = [dict(request_id=f"pilot-{iid}-{cid}", item_id=int(iid), condition=cid,
                 kind=kind, cell=cell or "", persona_k=k)
            for iid in picks for cid, kind, cell, k in conds]
    plan = pd.DataFrame(rows).merge(ent, on="item_id", validate="many_to_one")
    assert len(plan) == 60 and plan.request_id.is_unique
    return plan


def main():
    df = D.load_350()
    items = df.drop_duplicates("item_id").set_index("item_id")[["context", "response"]]
    plan = pilot_plan()
    plan.to_csv("results/06_pilot_plan.csv", index=False)
    prof = CFG.profile()

    show = [("Asian-Woman#1", "persona", "Asian-Woman", 0),
            ("Asian-Woman#2", "persona", "Asian-Woman", 1),
            ("White-Man#1", "persona", "White-Man", 0),
            ("Black-Woman#3", "persona", "Black-Woman", 2),
            ("baseline#1", "baseline", None, 0)]
    first_item = int(plan.item_id.iloc[0])
    ctx, rsp = items.loc[first_item, "context"], items.loc[first_item, "response"]

    out = ["=" * 100,
           "FIVE COMPLETE PROMPTS, EXACTLY AS THEY WILL BE SENT",
           "=" * 100,
           f"provider     {prof['provider']}",
           f"model        {prof['model']}",
           f"temperature  {CFG.TEMPERATURE}",
           f"max output   {CFG.MAX_OUTPUT_TOKENS} tokens",
           f"item_id      {first_item}  (the same conversation in all five, so the only",
           f"             difference between them is the persona text)",
           "",
           "Two strings are sent per call: the SYSTEM string and the USER string. The banner",
           "lines below are display only. The persona is the last paragraph of SYSTEM - read",
           "it in each of the five to confirm the demographic text is present and differs.",
           ""]
    for cid, kind, cell, k in show:
        block = P.condition_prompt_block(kind, cell, k)
        out += ["\n" + "#" * 100, f"#####  CONDITION: {cid}   ({kind})", "#" * 100, "",
                PR.render_full_prompt(ctx, rsp, block)]
    txt = "\n".join(out)
    with open("results/06_pilot_prompts.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)

    # ------------------------------------------------------------------- token profile
    c = ["\n" + "=" * 100, "TOKEN PROFILE (ESTIMATES)",
         f"From a {CHARS_PER_TOKEN} chars/token heuristic, not measured. Each provider "
         "tokenises differently;",
         "the pilot reports real counts from the API and supersedes these.", "=" * 100]
    rub = est_tokens(PR.RUBRIC)
    per = float(np.mean([est_tokens(P.persona_text(cl, k))
                         for cl in P.CELLS for k in range(3)] + [est_tokens(P.baseline_text())]))
    convs = [est_tokens(PR.USER_TEMPLATE.format(context=r.context, response=r.response))
             for r in items.itertuples()]
    c += [f"\n   rubric (identical in every call)          {rub:8.0f}",
          f"   persona block (mean over the 19 texts)   {per:8.0f}",
          f"   conversation + cue (mean over 350)       {np.mean(convs):8.0f}"
          f"   [min {min(convs):.0f}, max {max(convs):.0f}]",
          f"   ------------------------------------------------",
          f"   input per call                           {rub+per+np.mean(convs):8.0f}",
          f"   output (forced single word)              {CFG.MAX_OUTPUT_TOKENS:8.0f}",
          f"\n   full run: 350 items x {len(P.all_conditions())} conditions = "
          f"{350*len(P.all_conditions()):,} calls",
          f"   estimated total input tokens: "
          f"{350*len(P.all_conditions())*(rub+per+np.mean(convs))/1e6:.1f}M",
          f"\n   COST: none. Both target providers are free-tier. The binding constraint is",
          f"   rate limits and wall-clock, not money. See src/07_run.py --status."]
    ctxt = "\n".join(c)
    with open("results/06_token_profile.txt", "w", encoding="utf-8") as f:
        f.write(ctxt + "\n")
    print(ctxt)
    print("\nwrote results/06_pilot_prompts.txt, results/06_pilot_plan.csv, "
          "results/06_token_profile.txt")


if __name__ == "__main__":
    main()
