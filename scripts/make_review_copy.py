"""PRX の下書きから、日本語のメモと経緯を抜いた英語だけの写しを作る（外の読み手に渡す用）。"""
import re, sys
src, dst, title = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(src).read()
a = s.index("## Abstract"); b = s.index("## 補足（Appendix F、G）の予定")
out = []
for L in s[a:b].split("\n"):
    if L.startswith("#") and "（" in L: L = L[:L.index("（")]
    if L.startswith("（") or L.strip() == "---": continue
    L = re.sub(r"（[^（）]*）", "", L)
    if re.search(r"[぀-ヿ一-鿿]", L): continue
    L = L.replace("../paper/figures/", "figures/")
    out.append(L)
t = f"# {title}\n\n" + re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"
open(dst, "w").write(t)
print(len(t.split()), "words")
