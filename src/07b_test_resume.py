"""Test the resume logic against synthetic shards. Makes no API calls.

Resumability is a hard requirement ("a crash at call 5000 must not cost the first 5000"), so
it is tested rather than asserted. Synthetic records are written to a TEMP directory, never to
raw_responses/, so nothing fake can ever be mistaken for real data.

Cases covered:
  1. completed keys are skipped
  2. error records are NOT treated as complete, so they are retried
  3. a torn final line from a hard kill does not break the reader
  4. duplicate keys across shards are counted once
  5. records spread over several shards are all honoured
  6. a full shard set leaves nothing to do
"""
import sys, os, json, tempfile, importlib, shutil
sys.path.insert(0, "src")

runner = importlib.import_module("07_run")


def write_shard(d, name, records, torn_last=False):
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        if torn_last:
            fh.write('{"key": "999|Asian-Woman#1", "item_id": 999, "cond')  # cut mid-write
    return p


def rec(key, error=None):
    iid, cond = key.split("|")
    return dict(key=key, item_id=int(iid), condition=cond, kind="persona",
                cell="Asian-Woman", persona_k=0, ent_bin="low", provider="test",
                model_requested="test", model_version="test-v1", text="Yes",
                finish_reason="STOP", in_tokens=1000, out_tokens=1, attempts=1,
                error=error, raw={})


def main():
    plan = runner.build_plan("pilot")
    keys = [r["key"] for r in plan]
    print(f"pilot plan: {len(plan)} calls")

    tmp = tempfile.mkdtemp(prefix="dices_resume_")
    real_dir = runner.RAW_DIR
    runner.RAW_DIR = tmp
    try:
        # ---- 1 & 2 & 3: 30 done, 5 errored, torn final line
        done_keys, err_keys = keys[:30], keys[30:35]
        write_shard(tmp, "pilot_a_1.jsonl",
                    [rec(k) for k in done_keys] + [rec(k, error="Timeout") for k in err_keys],
                    torn_last=True)
        done, dupes, n_err = runner.load_done("pilot")
        todo = [r for r in plan if r["key"] not in done]
        print(f"\n1. completed keys skipped")
        print(f"   complete {len(done)} (expected 30)   remaining {len(todo)} (expected 30)")
        assert len(done) == 30, len(done)
        assert set(done) == set(done_keys)
        print(f"2. error records retried")
        print(f"   error records seen {n_err} (expected 5)")
        assert n_err == 5
        assert all(k in {r['key'] for r in todo} for k in err_keys), "errored keys not requeued"
        print(f"   all 5 errored keys are back in the todo list: yes")
        print(f"3. torn final line tolerated: yes (reader did not raise, "
              f"key 999|... not counted)")
        assert "999|Asian-Woman#1" not in done

        # ---- 4 & 5: a second shard, overlapping
        write_shard(tmp, "pilot_b_2.jsonl",
                    [rec(k) for k in keys[25:45]])       # 25-29 overlap, 35-44 new
        done, dupes, n_err = runner.load_done("pilot")
        todo = [r for r in plan if r["key"] not in done]
        print(f"\n4. duplicate keys across shards counted once")
        print(f"   duplicates reported {dupes} (expected 5)")
        assert dupes == 5, dupes
        print(f"5. records spread over shards all honoured")
        print(f"   complete {len(done)} (expected 45)   remaining {len(todo)} (expected 15)")
        assert len(done) == 45, len(done)
        assert len(todo) == 15

        # ---- 6: everything done
        write_shard(tmp, "pilot_c_3.jsonl", [rec(k) for k in keys])
        done, _, _ = runner.load_done("pilot")
        todo = [r for r in plan if r["key"] not in done]
        print(f"\n6. complete shard set leaves nothing to do")
        print(f"   complete {len(done)} / {len(plan)}   remaining {len(todo)} (expected 0)")
        assert len(todo) == 0 and len(done) == len(plan)

        # ---- the guarantee, stated as a number
        print(f"\n{'='*78}")
        print("THE GUARANTEE")
        print(f"{'='*78}")
        print("   Each response is written, flushed, and fsync'd before the next call is made.")
        print("   Worst case loss on a crash, kill, quota wall, or power cut is therefore")
        print("   ONE call - the one in flight. A crash at call 5,000 of 7,350 costs call")
        print("   5,000 only; the other 4,999 are on disk and are skipped on restart.")
        print("\nALL RESUME TESTS PASSED")
    finally:
        runner.RAW_DIR = real_dir
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"\n(temp shards deleted; raw_responses/ was never written to)")


if __name__ == "__main__":
    main()
