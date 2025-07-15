import os
import re
import urllib.parse
from bs4 import BeautifulSoup
from urllib.parse import unquote

# ======= 1️⃣ 文件夹 & 输出 =======
input_dir = "output"   # 你的 HTML 文件夹
output_file = "output/decoder_all.txt"
def decoder_brige():
    # ======= 2️⃣ 存放所有提取到的 lin =======
    all_lins = []
    
    # ======= 3️⃣ 遍历文件夹下所有 .html 文件 =======
    for filename in os.listdir(input_dir):
        if filename.endswith(".html"):
            file_path = os.path.join(input_dir, filename)
            print(f"\n🔍 正在处理：{file_path}")
            
            with open(file_path, "r", encoding="utf-8") as f:
                html = f.read()
            
            # -------------------------------
            # 先用正则提取 URL 里的 lin=
            # -------------------------------
            matches = re.findall(r'lin=([^"&]+)', html)
            if matches:
                for lin_encoded in matches:
                    lin_decoded = unquote(lin_encoded)
                    all_lins.append(lin_decoded)
                    print(f"✅ [URL] lin: {lin_decoded[:50]}...")
            else:
                print("⚠️ [URL] 没找到 lin")
            
            # -------------------------------
            # 再用 BeautifulSoup 提取 hv_popuplin 的 st||
            # -------------------------------
            soup = BeautifulSoup(html, "html.parser")
            for td in soup.find_all("td", class_="movie"):
                a_tag = td.find("a")
                if a_tag and a_tag.has_attr("onclick"):
                    onclick_str = a_tag["onclick"]
                    param_match = re.search(r"hv_popuplin\('(.+?)'\)", onclick_str)
                    if param_match:
                        param_str = param_match.group(1)
                        decoded_param = unquote(param_str)
                        st_match = re.search(r"st\|\|(.+)", decoded_param)
                        if st_match:
                            st_content = "st||" + st_match.group(1)
                            all_lins.append(st_content)
                            print(f"✅ [Movie] st: {st_content[:50]}...")
                        else:
                            print("⚠️ [Movie] 没找到 st 部分")
                    else:
                        print("⚠️ [Movie] 没匹配到 hv_popuplin")
    
    # ======= 4️⃣ 保存 =======
    with open(output_file, "w", encoding="utf-8") as f_out:
        for lin in all_lins:
            f_out.write(lin + "\n")
    
    print(f"\n✔️ 共提取到 {len(all_lins)} 条 lin/st，已保存到：{output_file}")


def complete_md_st(st_raw):
    """
    给定一个 st|| 串，自动补全第四家手牌，返回新的 st|| 串
    """
    # 先拆出 md 部分
    md_match = re.search(r'md\|[1234](.*?)\|', st_raw)
    if not md_match:
        raise ValueError("未找到 md 部分！")
    md_data = md_match.group(1)

    # 前三家
    hands = md_data.split(',')
    if len(hands) < 3:
        raise ValueError("md 不完整！")

    hand1, hand2, hand3 = hands[:3]

    # 全 52 张牌
    ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    suits = ["S", "H", "D", "C"]
    all_cards = set(suit + rank for suit in suits for rank in ranks)

    # 已用
    used_cards = set()
    for hand in [hand1, hand2, hand3]:
        segments = re.findall(r"[SHDC][^SHDC]*", hand)
        for seg in segments:
            suit = seg[0]
            for rank in seg[1:]:
                used_cards.add(suit + rank)

    # 第四家
    remaining = all_cards - used_cards

    # 按顺序重组
    fourth_hand = ""
    for suit in suits:
        fourth_hand += suit
        cards = [card[1:] for card in remaining if card[0] == suit]
        fourth_hand += "".join(sorted(cards, key=lambda x: ranks.index(x)))

    # 重新拼接新的 md
    new_md = f"md|1{hand1},{hand2},{hand3},{fourth_hand}|"

    # 替换原 md
    new_st = re.sub(r'md\|[1234].*?\|', new_md, st_raw)

    return new_st

def format_md_st():
    input_file = "output/decoder_all.txt"
    output_file = "output/decoder_all.txt"
    
    results = []
    
    with open(input_file, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                new_st = complete_md_st(line)
                results.append(new_st)
                print(f"✅ 已处理：{new_st[:60]}...")
            except Exception as e:
                results.append(line)
    
    with open(output_file, "w", encoding="utf-8") as f_out:
        for item in results:
            f_out.write(item + "\n")
    
    print(f"\n✔️ 全部完成，已保存到 {output_file}")


if __name__ == "__main__":
    decoder_brige() # 解码获得最终表示
    format_md_st() # 统一解码形式方便之后解析