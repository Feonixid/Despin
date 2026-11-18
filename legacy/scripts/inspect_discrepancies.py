import json

def main():
    with open('results_v2/benchmark_v2_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = data['records']
    print(f"Total problems: {len(records)}")
    
    greedy_pass_despin_fail = []
    despin_pass_greedy_fail = []

    for r in records:
        pid = r['problem']['id']
        bench = r['problem']['benchmark']
        res = r['results']
        g_corr = res['Greedy (T=0.0)']['correct']
        d_corr = res['DESPIN-v2 (PSRD + ASD)']['correct']

        if g_corr and not d_corr:
            greedy_pass_despin_fail.append((pid, bench, res['Greedy (T=0.0)'], res['DESPIN-v2 (PSRD + ASD)']))
        elif d_corr and not g_corr:
            despin_pass_greedy_fail.append((pid, bench, res['Greedy (T=0.0)'], res['DESPIN-v2 (PSRD + ASD)']))

    print(f"\n--- Greedy PASS, DESPIN FAIL ({len(greedy_pass_despin_fail)}) ---")
    for pid, bench, g, d in greedy_pass_despin_fail:
        print(f"Problem: {pid} ({bench})")
        print(f"  Greedy: Pred={g['pred']} Tokens={g['tokens']}")
        print(f"  DESPIN: Pred={d['pred']} Tokens={d['tokens']} Rollbacks={d['rollbacks']}")

    print(f"\n--- DESPIN PASS, Greedy FAIL ({len(despin_pass_greedy_fail)}) ---")
    for pid, bench, g, d in despin_pass_greedy_fail:
        print(f"Problem: {pid} ({bench})")
        print(f"  Greedy: Pred={g['pred']} Tokens={g['tokens']}")
        print(f"  DESPIN: Pred={d['pred']} Tokens={d['tokens']} Rollbacks={d['rollbacks']}")

if __name__ == '__main__':
    main()

