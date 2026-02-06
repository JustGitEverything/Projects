import pickle
# import tiktoken

# tokenizer = tiktoken.get_encoding("p50k_base")
# print("s", tokenizer.encode("All Citizens:"))
# print("s", tokenizer.encode("\n"))


def stats(t):
    counts = {}
    for ft, st in zip(t[:-1], t[1:]):
        counts[(ft, st)] = counts.get((ft, st), 0) + 1
    return counts


def merge(ft, st, t, nxt):
    new_t = []
    i = 0
    while i < len(t):
        if i < len(t) - 1 and t[i] == ft and t[i + 1] == st:
            new_t.append(nxt)

            i += 2
        else:
            new_t.append(t[i])

            i += 1
    return new_t


def f_merge(ft, st, t, nxt):
    i = 0
    while i < len(t):
        if i < len(t) - 1 and t[i] == ft and t[i + 1] == st:
            t[i] = nxt
            t.pop(i + 1)

        i += 1

    return t


# potential negative i issue when unpacking
def rtn(t, it, us):
    new_t = []

    i = 0
    while i < len(t):
        char = t[i]

        if char < us:
            new_t.append(char)

            i += 1
        else:
            ft, st = it[char]
            t[i - 1] = ft
            t[i] = st
            i -= 1

    return new_t


# cumbersome
def show_tokens(t, it, us):
    new_t = []

    last_highest = -1
    i = 0
    while i < len(t):
        char = t[i]

        if char < us:
            # print("USD", char, us)
            new_t.append(char)

            if i == last_highest:
                new_t.append(-1)

            i += 1
        else:
            # print("HERE 1")
            if i > last_highest:
                # print("HERE")
                new_t.append(-2)

                last_highest = i

            ft, st = it[char]
            t[i - 1] = ft
            t[i] = st

            i -= 1

    # print("new t", new_t)

    decode_colors(new_t, it)


# merges without taking merge hierarchy into account -> wrong
def merge_text(t, ct):
    i = 0
    while i < len(t):
        if i < len(t) - 1:
            try:
                ft = t[i]
                st = t[i + 1]
                hit = ct[(ft, st)]
                # print("hit", hit)

                t.pop(i + 1)
                t[i] = hit

                i = i - 10 if i - 10 >= 0 else 0
            except KeyError:
                i += 1
        else:
            i += 1

    return t


def quick_merge(t, ct):
    for key, i in ct.items():
        if not isinstance(key, str):
            # print(key, i)

            t = f_merge(key[0], key[1], t, i)

    return t


def priority_merge(t, ct):
    """Efficiently applies the stored merge order without scanning the text per merge."""
    import heapq

    if len(t) < 2:
        return t

    pair_ranks = {}
    rank = 0
    for key in ct:
        if isinstance(key, tuple):
            pair_ranks[key] = rank
            rank += 1

    if not pair_ranks:
        return t

    n = len(t)
    prev = [-1] * n
    nxt = [-1] * n
    alive = [True] * n

    for i in range(n):
        if i > 0:
            prev[i] = i - 1
        if i < n - 1:
            nxt[i] = i + 1

    heap = []
    counter = 0

    def push(left):
        nonlocal counter

        right = nxt[left]
        if right == -1 or not alive[left] or not alive[right]:
            return

        pair = (t[left], t[right])
        pair_rank = pair_ranks.get(pair)
        if pair_rank is None:
            return

        heapq.heappush(heap, (pair_rank, counter, left))
        counter += 1

    for i in range(n - 1):
        push(i)

    while heap:
        rank_value, _, left = heapq.heappop(heap)

        if not alive[left]:
            continue

        right = nxt[left]
        if right == -1 or not alive[right]:
            continue

        pair = (t[left], t[right])
        actual_rank = pair_ranks.get(pair)
        if actual_rank is None:
            continue

        if actual_rank != rank_value:
            heapq.heappush(heap, (actual_rank, counter, left))
            counter += 1
            continue

        new_token = ct[pair]
        t[left] = new_token

        next_right = nxt[right]
        nxt[left] = next_right
        if next_right != -1:
            prev[next_right] = left

        alive[right] = False

        prev_left = prev[left]
        if prev_left != -1:
            push(prev_left)

        if nxt[left] != -1:
            push(left)

    result = []
    idx = 0
    while idx != -1 and not alive[idx]:
        idx = nxt[idx]

    while idx != -1:
        result.append(t[idx])
        idx = nxt[idx]

    return result


def ret(t, it):
    i = 0
    while i < len(t):
        cur = it[t[i]]
        if isinstance(cur, str):
            # print("is base", cur)
            # print(it[i])
            i += 1
        else:
            # print("is composed", cur)
            new_elements = ret([cur[0], cur[1]], it)
            # print(new_elements)
            # print("pre", t)
            t.pop(i)
            t[i:i] = new_elements
            # print("post", t)
            i += 1
    return t


# wrong since later merged does not automatically mean better and containing more tokens
def chunk_merge(t, it, us):
    # print("itc", it)
    nit = {}

    for key, q in it.items():
        if key < us:
            nit[key] = [key]  # [it[key]]

            continue

        # print("k", key)
        ft, st = q
        # print(ft, st, it[key])
        nit[key] = nit[ft] + nit[st]
        # print("pos", it[key])

    print("merged itc", nit)

    for key, q in reversed(nit.items()):
        # print("k", key, q)

        for i in range(len(t)):
            if i + len(q) > len(t) or t[i] != q[0]:
                continue

            if t[i: i + len(q)] == q:
                t[i:i + len(q)] = [key]
            # print(t[i])

    return t


def best_merge(t, it, us):
    # print("itc", it)
    nit = {}

    for key, q in it.items():
        if key < us:
            nit[key] = [key]  # [it[key]]

            continue

        # print("k", key)
        # print("q", q)
        ft, st = q
        # print(ft, st, it[key])
        nit[key] = nit[ft] + nit[st]
        # print("pos", it[key])

    quick_dict = {}

    for key, q in nit.items():
        # print("t", q)
        quick_dict[q[0]] = quick_dict[q[0]] + [[key, q]] if q[0] in quick_dict else [[key, q]]

    for key, q in quick_dict.items():
        q.sort(key=lambda k: len(k[1]), reverse=True)
    # print("QD", quick_dict)
    # quick_dict = sorted(quick_dict, key=lambda k: len(quick_dict[k]), reverse=True)
    # print("BEW", quick_dict)

    i = 0
    while i < len(t):
        char = t[i]

        candidates = quick_dict[char]
        # print("c", t[i], candidates)

        for can in candidates:
            seq = can[1]
            if i + len(seq) < len(t) and t[i:i + len(seq)] == seq:
                # print("match", seq)
                t[i:i + len(seq)] = [can[0]]
                break

        i += 1

    return t


from collections import deque


def chunk_merge_priority(t, it, us):
    """
    t  : list[int] - current token sequence
    it : dict[int, tuple[int, int]] - merge table: key -> (left, right)
    us : int - size of the base vocabulary (ids < us are primitive chars/tokens)
    """

    # ---- 1) Build full expansions nit[key] as tuples of base tokens ----
    nit = {}

    def expand(key):
        if key in nit:
            return nit[key]
        if key < us:
            nit[key] = (key,)
        else:
            ft, st = it[key]
            nit[key] = expand(ft) + expand(st)
        return nit[key]

    for key in it:
        expand(key)

    # ---- 2) Build patterns list with explicit priority ----
    # Original code uses `for key,q in reversed(nit.items())`,
    # so later keys in `it` have higher priority.
    merge_order = list(reversed(list(it.keys())))
    rank = {key: i for i, key in enumerate(merge_order)}  # lower i = higher priority

    patterns = []  # (sequence, key, priority)
    for key in merge_order:
        seq = nit[key]
        if len(seq) > 1:           # length-1 doesn't need a merge
            patterns.append((seq, key, rank[key]))

    if not patterns:
        return t

    # ---- 3) Build Aho–Corasick automaton over int sequences ----
    # Each node: {'next': {token: next_state}, 'fail': int, 'out': [(key, prio, length), ...]}
    nodes = [{'next': {}, 'fail': 0, 'out': []}]

    for seq, key, prio in patterns:
        cur = 0
        for token in seq:
            nxt = nodes[cur]['next'].get(token)
            if nxt is None:
                nxt = len(nodes)
                nodes[cur]['next'][token] = nxt
                nodes.append({'next': {}, 'fail': 0, 'out': []})
            cur = nxt
        nodes[cur]['out'].append((key, prio, len(seq)))

    # Build failure links (standard AC construction)
    q = deque()
    for token, nxt in nodes[0]['next'].items():
        nodes[nxt]['fail'] = 0
        q.append(nxt)

    while q:
        v = q.popleft()
        for token, nxt in nodes[v]['next'].items():
            q.append(nxt)
            f = nodes[v]['fail']
            while f != 0 and token not in nodes[f]['next']:
                f = nodes[f]['fail']
            nodes[nxt]['fail'] = nodes[f]['next'].get(token, 0)
            nodes[nxt]['out'].extend(nodes[nodes[nxt]['fail']]['out'])

    # ---- 4) Run automaton over t to find all matches ----
    matches = []  # (start, end, key, prio, length)
    state = 0
    for i, tok in enumerate(t):
        while state != 0 and tok not in nodes[state]['next']:
            state = nodes[state]['fail']
        state = nodes[state]['next'].get(tok, 0)
        if nodes[state]['out']:
            for (key, prio, length) in nodes[state]['out']:
                start = i - length + 1
                end = i + 1
                matches.append((start, end, key, prio, length))

    if not matches:
        return t

    # ---- 5) Resolve overlaps by priority, then by position, then by length ----
    # Lower prio index = higher priority (because of rank above).
    matches.sort(key=lambda m: (m[3], m[0], -m[4]))  # (prio, start, -length)

    n = len(t)
    covered = [False] * (n + 1)
    chosen = []

    for (s, e, key, prio, length) in matches:
        if s < 0 or e > n:
            continue
        if any(covered[i] for i in range(s, e)):
            continue
        # accept this match
        chosen.append((s, e, key))
        for i in range(s, e):
            covered[i] = True

    if not chosen:
        return t

    chosen.sort(key=lambda m: m[0])  # sort by start index

    # ---- 6) Build final token sequence ----
    out = []
    pos = 0
    ci = 0
    while pos < n:
        if ci < len(chosen) and chosen[ci][0] == pos:
            s, e, key = chosen[ci]
            out.append(key)
            pos = e
            ci += 1
        else:
            out.append(t[pos])
            pos += 1

    return out


def light_merge(t, it, us):
    from collections import defaultdict
    # print("itc", it)
    nit = {}

    for key, q in it.items():
        if key < us:
            nit[key] = (key, )

            continue

        ft, st = q
        nit[key] = nit[ft] + nit[st]

    patterns_by_first = defaultdict(list)
    for key, seq in nit.items():
        patterns_by_first[seq[0]].append((key, seq))

    for lst in patterns_by_first.values():
        lst.sort(key=lambda kv: len(kv[1]), reverse=True)

    out = []
    i = 0
    n = len(t)

    while i < n:
        tok = t[i]
        candidates = patterns_by_first.get(tok)

        if not candidates:
            out.append(tok)

            i += 1
            continue

        for key, seq in candidates:
            m = len(seq)
            if i + m > n:
                continue
            ok = True
            for j in range(m):
                if t[i + j] != seq[j]:
                    ok = False
                    break

            if ok:
                match_key = key
                match_len = m
                break

        if match_key is None:
            out.append(tok)
            i += 1
        else:
            out.append(match_key)
            i += match_len
    return out


def karpathy_merge(t, ct):
    while True:
        sts = stats(t)
        pair = min(sts, key=lambda p: ct.get(p, float('inf')))
        if pair not in ct:
            break
        idx = ct[pair]
        t = f_merge(pair[0], pair[1], t, idx)

    return t


def easy_colors(t, it, us, is_first=True, ci=0):
    from colorist import BgBrightColor as b
    from colorist import Color
    END = b.OFF
    COLORS = [b.CYAN, b.MAGENTA, b.GREEN, b.BLUE, b.YELLOW, b.RED]
    c_i = 0
    res = ''

    i = 0
    while i < len(t):
        cur = it[t[i]]
        if isinstance(cur, str):
            # print("is base", cur)
            # print(it[i])
            res += cur if cur != '\n' else ' ' + END + '\n' + COLORS[ci]
            i += 1
        else:
            if is_first:
                c_i = (c_i + 1) % len(COLORS)
                res += Color.BLACK + COLORS[c_i]
                # print("is composed", cur)
                res += easy_colors([cur[0], cur[1]], it, us, is_first=False, ci=c_i)
                res += END
            else:
                res += easy_colors([cur[0], cur[1]], it, us, is_first=False, ci=ci)
            # print(new_elements)
            # print("pre", t)
            # print("post", t)
            i += 1
    return res


# replaced
def decode_colors(t, it):
    from colorist import BgBrightColor as b
    from colorist import Color
    END = b.OFF
    COLORS = [b.CYAN, b.MAGENTA, b.GREEN, b.BLUE, b.YELLOW, b.RED]

    c_i = 0
    last_color = None

    res = ""

    for c in t:
        if c == -1:
            res += END
        elif c < 0:
            last_color = Color.BLACK + COLORS[c_i]
            res += last_color

            c_i += 1
            c_i = c_i % len(COLORS)
        else:
            if it[c] == '\n':
                res += ' ' + END + '\n' + last_color
            else:
                res += Color.BLACK + it[c]

    print(res)


def chunk_decode(t, it, us):
    # print("itc", it)
    nit = {}

    for key, q in it.items():
        if key < us:
            nit[key] = [key]  # [it[key]]

            continue

        # print("k", key)
        ft, st = q
        # print(ft, st, it[key])
        nit[key] = nit[ft] + nit[st]
        # print("pos", it[key])

    # print("merged itc", it)

    i = 0
    while i < len(t):
        token = t[i]
        # print("i", i, token)
        if not token < us:
            # print("pre", t, token)
            t[i:i + 1] = nit[token]
            # print("post", t)

        i += 1

    return t


def load_cti():
    with open('cti.p', 'rb') as fp:
        return pickle.load(fp)


def load_itc():
    with open('itc.p', 'rb') as fp:
        return pickle.load(fp)


def encode(ip, ct):
    return [ct.get(c, 11) for c in ip]


def decode(ip, it):
    return "".join([it[i] for i in ip])


if __name__ == "__main__":
    import time
    import json

    with open(r"""C:\Users\Justin Hohenstein\PycharmProjects\datasets\fineData\shard_00.txt""", 'r', encoding='utf-8') as f:
        text = f.read()

    # chars = sorted(list(set(text)))

    with open(r"""C:\Users\Justin Hohenstein\PycharmProjects\datasets\fineData\training_unique_chars.json""", "r", encoding="utf-8") as fh:
        loaded_chars = json.load(fh)

    print('chars', loaded_chars)

    unit_size = len(loaded_chars)
    # unit_size = 314  # len(chars)

    cti = {c: i for i, c in enumerate(loaded_chars)}
    itc = {i: c for i, c in enumerate(loaded_chars)}

    encode = lambda ip: [cti[c] for c in ip]
    decode = lambda ip: "".join([itc[i] for i in ip])

    print("encode", encode("test"), decode(encode("test")))
    print("C", loaded_chars)
    print("original length:", len(text))

    text = encode(text)[:200000]
    print("new len", len(text))

    start_time = time.time()
    # for 100000 with 20k merges only 5k were actual merges, remaining first token merge
    for i in range(20000):
        c = stats(text)
        most_common = max(c, key=c.get)
        next_merged = len(cti)

        # print("t", text[:50])
        text = merge(most_common[0], most_common[1], text, next_merged)

        cti[most_common] = next_merged
        itc[next_merged] = most_common

        if i % 1000 == 0:
            print("merge", i, "done")
            print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()

    # print("most common:", most_common, most_common[0])
    print("CTI", cti)
    # print("ITC", itc)
    # print("t", text[:50])

    print("new length:", len(text))

    # print("rtn", rtn(text[:500]))
    # text = ret(text[:100], itc, unit_size)
    # print("dec", decode(text[:500]))

    with open('cti.p', 'wb') as f:
        pickle.dump(cti, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open('itc.p', 'wb') as f:
        pickle.dump(itc, f, protocol=pickle.HIGHEST_PROTOCOL)
