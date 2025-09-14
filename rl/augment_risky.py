# augment_risky.py
import random, pathlib, csv

DELETE_TBLS = ["users", "orders", "logs", "`Customer-Data`", "`t`"]
UPDATE_TBLS = ["users", "orders", "inventory", "`backup`.`tbl`"]
SETS = ["set name='X'", "set a=1, b=b+1", "set `flag`=true"]

COMMENTS = [
    " -- bulk op\n", " /* audit */ ", " /*+ INDEX(t idx) */ ",
    " # quick\n"
]
SP = [" ", "  ", "\t", "\n", " \n "]

def gen_delete_no_where(n=300):
    out = []
    for _ in range(n):
        t = random.choice(DELETE_TBLS)
        pieces = ["DELETE", random.choice(SP), "FROM", random.choice(SP), t]
        # chèn comment/space ngẫu nhiên
        for _ in range(random.randint(0,2)):
            pieces.insert(random.randint(0, len(pieces)), random.choice(COMMENTS))
        out.append("".join(pieces) + ";")
    return out

def gen_update_no_where(n=300):
    out = []
    for _ in range(n):
        t = random.choice(UPDATE_TBLS)
        s = random.choice(SETS)
        pieces = ["UPDATE", random.choice(SP), t, random.choice(SP), "SET", random.choice(SP), s]
        for _ in range(random.randint(0,2)):
            pieces.insert(random.randint(0, len(pieces)), random.choice(COMMENTS))
        out.append("".join(pieces) + ";")
    return out

def main():
    out_path = pathlib.Path("data/augmented_risky.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for q in gen_delete_no_where() + gen_update_no_where():
        rows.append({"sql": q, "normalized_query": q.lower(), "reward": -3, "risky": 1})
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sql","normalized_query","reward","risky"])
        w.writeheader()
        w.writerows(rows)
    print("Wrote", out_path)

if __name__ == "__main__":
    main()
