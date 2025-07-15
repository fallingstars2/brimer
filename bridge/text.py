import urllib.parse
from pydantic import BaseModel

lin_decoded = """
st||md|2S9AH2568JD5TQAC6A,S34TH9KD2689C2TJQ,S2578KH34QD7JKC57,SQJ6HAT7D43CK9843|rh||ah|Board 4|sv|b|mb|p|mb|p|mb|p|mb|1N|an|notrump opener. Could have 5M. -- 2-5 !C; 2-5 !D; 2-5 !H; 2-5 !S; 15-|mb|p|mb|2H!|an|Jacoby transfer -- 5+ !S; 11- HCP; 12- total points |mb|p|mb|2S|an|Transfer completed to S -- 2-5 !C; 2-5 !D; 2-5 !H; 2-5 !S; 15-17 HCP;|mb|p|mb|2N|an|Invitational to either game -- 5 !S; 9 HCP |mb|p|mb|p|mb|p|pc|CQ|pc|C5|pc|C4|pc|C6|pc|CJ|pc|C7|pc|C8|pc|CA|pc|D5|pc|D8|pc|DK|pc|D3|pc|DJ|pc|D4|pc|DT|pc|D6|pc|D7|pc|H7|pc|DQ|pc|D9|pc|DA|pc|D2|pc|H3|pc|HT|pc|SA|pc|S3|pc|S2|pc|S6|pc|S9|pc|ST|pc|SK|pc|SJ|pc|S8|pc|SQ|pc|H2|pc|S4|pc|CK|pc|H5|pc|C2|pc|S5|pc|C9|pc|H6|pc|CT|pc|S7|pc|H9|pc|H4|pc|HA|pc|H8|pc|C3|pc|HJ|pc|HK|pc|HQ|
"""
import re
from urllib.parse import unquote
from pydantic import BaseModel

class BridgeTools(BaseModel):
    """
    该方法存放所有关于桥牌类的方法。包括手牌转换，胜负判断，出牌可能确定等方法
    """
    @staticmethod
    def split_hand_string(hand_str: str) -> list[str]: # 手牌转换
        """
        输入: "SQ52HAQ7652DKTC85"
        输出: ['SQ', 'S5', 'S2', 'HA', ...]
        """
        # 用正则找到每一段花色和对应牌
        parts = re.findall(r'([SHDC])([^SHDC]+)', hand_str)
        
        # 拆成单张
        result = []
        for suit, ranks in parts:
            for rank in ranks:
                result.append(suit + rank)
        return result
    
    @staticmethod
    def assign_players_by_cards(chu,n,e,s,w): # 出牌规则化
        """
        根据 BridgeItem 里的 n/e/s/w 四家手牌，给 chu 每张牌标记出牌人。
        返回同样结构，只是把每张牌换成 (player, card)。
        """
        # 把四家牌做成查找表
        player_hands = {
            'n': set(n),
            'e': set(e),
            's': set(s),
            'w': set(w)
        }
        
        result = []
        
        for trick in chu:
            trick_result = []
            for card in trick:
                found = False
                for player, cards in player_hands.items():
                    if card in cards:
                        trick_result.append((player, card))
                        cards.remove(card)  # 用过的牌去掉，避免重复
                        found = True
                        break
                if not found:
                    raise ValueError(f"牌 {card} 没有在任何人的手牌中找到！")
            result.append(trick_result)
        
        return result
    
    @staticmethod
    def get_final_contract(jiao: list[str]) -> (str,str):
        """
        根据叫牌序列，返回最终定约：
        例如：
            ['1H', '1S', 'p', '2H', 'p', '2S', 'p', 'p', 'p'] -> 2S
        若无人叫牌，则返回 'Pass Out'
        """
        # 去掉后缀空格等
        map = ["n","e","s","w"]
        calls = [c.strip().upper() for c in jiao]
        
        # 检查是否至少有一个非 Pass
        non_pass = [c for c in calls if c != 'P' and c != 'PASS']
        if not non_pass:
            return "Pass Out"
        
        # 叫牌结束条件：最后三个 Pass
        if calls[-3:] != ['P', 'P', 'P']:
            raise ValueError("叫牌还没结束（最后三个不是 Pass）")
        n = len(calls)
        # 定约就是最后一个非 Pass
        final_contract = non_pass[-1]
        return (map[n%4],final_contract)
    
    @staticmethod
    def who_win_this_trick(chu: list[tuple[str, str]], contract: str) -> str:
        """
        chu: [('n', 'C5'), ('e', 'CA'), ('s', 'C2'), ('w', 'CJ')]
        contract: '2S'
        返回：该墩赢家的方位，如 'n'
        """
        # 定约花色
        trump_suit = contract[-1].upper()  # 'S', 'H', 'D', 'C', 或 'N' (无将)
        if trump_suit == 'N':
            trump_suit = None
        
        # 牌面大小
        ranks = "23456789TJQKA"
        rank_value = {r: i for i, r in enumerate(ranks, start=2)}
        
        # 本墩首领出的花色
        first_suit = chu[0][1][0]
        
        def card_score(card):
            rank = card[1:]
            return rank_value[rank]
        
        # 分类
        trumps = [(player, card) for player, card in chu if trump_suit and card[0] == trump_suit]
        if trumps:
            # 有主将，出最大的主将赢
            winner = max(trumps, key=lambda x: card_score(x[1]))
        else:
            # 没主将，比首花色最大的牌
            leads = [(player, card) for player, card in chu if card[0] == first_suit]
            winner = max(leads, key=lambda x: card_score(x[1]))
        
        return winner[0]
    
    @staticmethod
    def next_player_and_legal_cards(
            trick: list,     # 形如 [('w', 'D5'), ('n', 'S2'), None, None]
            hand: list[str], # 当前玩家手牌
    ):
        """
        返回：(下一个该谁出, 可出的牌列表)
        """
        if all(h == None for h in hand):
            print("手牌已经出完，可能有错请修改")
            return None, None
        if trick[0] == None: # 如果第一个是None
            print("当前是第一个出牌，为上一墩的获胜者先出")
            return hand,""
        # 顺时针顺序（示例）
        order = ['n', 'e', 's', 'w']
        lead_suit = trick[0][1][0]
        # 已经出的
        played = [p for p in trick if p is not None]
        # 下一个
        if played:
            last = played[-1][0]
            idx = order.index(last)
            next_idx = (idx + 1) % 4
        else:
            # 若一个都没出，就从 trick[0] 推断
            next_idx = 0
        # 下一个该谁出
        next_player = order[next_idx]
        # 可出的牌（如果有首领花色，必须跟牌）
        same_suit_cards = [card for card in hand if card and card.startswith(lead_suit)]
        if same_suit_cards:
            legal = same_suit_cards
        else:
            legal = [card for card in hand if card]
        return next_player, legal

class BridgeItem(BaseModel):
    """
    桥牌对局信息
    """
    name: str | None = None
    n: list[str]
    e: list[str]
    s: list[str]
    w: list[str]
    zhuang: tuple[str,str]  # 庄家方向
    is_ju: str | None = None
    jiao: list[str]
    chu: list  # 出牌顺序，包含分墩
    
    @classmethod
    def from_lin(cls, lin_decoded: str) -> "BridgeItem":
        # ========= 1️⃣ 解析庄家 =========
        sv_match = re.search(r"sv\|([a-z])\|", lin_decoded)
        is_ju = sv_match.group(1) if sv_match else None

        # ========= 2️⃣ 解析手牌 =========
        md_match = re.search(r"md\|([1234])(.*?)\|", lin_decoded)
        if not md_match:
            raise ValueError("未找到 md 部分")
        dealer_num = md_match.group(1)
        md_data = md_match.group(2)

        hands = md_data.split(",")
        if len(hands) < 4:
            raise ValueError("手牌数量不足四家")

        # 按照桥牌约定顺序调整：md 的首位数字是庄家，后面顺时针为 S, W, N, E
        # 庄家的方向 + 对应手牌分配
        dealer_map = {
            "1": ["S", "W", "N", "E"],
            "2": ["W", "N", "E", "S"],
            "3": ["N", "E", "S", "W"],
            "4": ["E", "S", "W", "N"]
        }

        order = dealer_map[dealer_num]
        hand_map = dict(zip(order, hands))

        # ========= 3️⃣ 解析叫牌 =========
        jiao_raw = re.findall(r"mb\|([^|]+)", lin_decoded)
        jiao = jiao_raw

        # ========= 4️⃣ 解析出牌 =========
        pc_raw = re.findall(r"pc\|([^|]+)", lin_decoded)

        # 将出牌顺序按 4 张分墩
        chu = []
        trick = []
        for idx, card in enumerate(pc_raw):
            trick.append(card)
            if len(trick) == 4:
                chu.append(trick)
                trick = []
        if trick:
            chu.append(trick)
        
        n,e,s,w = BridgeTools.split_hand_string(hand_map["N"]),BridgeTools.split_hand_string(hand_map["E"]),BridgeTools.split_hand_string(hand_map["S"]),BridgeTools.split_hand_string(hand_map["W"])
        # ========= 5️⃣ 返回实例 =========
        return cls(
            n=n,
            e=e,
            s=s,
            w=w,
            zhuang=BridgeTools.get_final_contract(jiao),
            is_ju=is_ju,   # 可以根据需要扩展
            jiao=jiao,
            chu = BridgeTools.assign_players_by_cards(chu,n,e,s,w)
        )
    


if __name__ == "__main__":
    item = BridgeItem.from_lin(lin_decoded)
    print(item)
    print(BridgeTools.who_win_this_trick(item.chu[1],item.zhuang[1]))
    c = item.chu[0]
    c[2]=None
    c[3]=None
    print(c)
    print(BridgeTools.next_player_and_legal_cards(c,item.n))
# def lin_decoder(lin_encoded:list[str]):
#     l_items = []
#     for l in lin_encoded:
#         item = {}
#         l_decoded = urllib.parse.unquote(l.strip())
#         fields = l_decoded.split('|')
#         # 3. 准备解析容器
#         metadata = {}
#         calls = []
#         call_explanations = []
#         play_cards = []
#         # 4. 解析字段
#         i = 0
#         while i < len(fields):
#             tag = fields[i]
#             if tag == 'md':
#                 metadata['hands'] = fields[i+1]
#                 i += 2
#             elif tag == 'sv':
#                 metadata['vul'] = fields[i+1]
#                 i += 2
#             elif tag == 'ah':
#                 metadata['board'] = fields[i+1]
#                 i += 2
#             elif tag == 'mb':
#                 calls.append(fields[i+1])
#                 i += 2
#             elif tag == 'an':
#                 call_explanations.append(fields[i+1])
#                 i += 2
#             elif tag == 'pc':
#                 play_cards.append(fields[i+1])
#                 i += 2
#             elif tag == 'mc':
#                 metadata['mc'] = fields[i+1]
#                 i += 2
#             else:
#                 i += 1
#         item["is_ju"]=metadata.get('vul')
#
#         print(fields)
#
#     return l_items
# lin_decoder([lin_encoded])
# # 1. URL解码
# lin_decoded = urllib.parse.unquote(lin_encoded.strip())
#
# # 2. 按 | 分割
# fields = lin_decoded.split('|')
#
# # 3. 准备解析容器
# metadata = {}
# calls = []
# call_explanations = []
# play_cards = []
#
# # 4. 解析字段
# i = 0
# while i < len(fields):
#     tag = fields[i]
#     if tag == 'md':
#         metadata['hands'] = fields[i+1]
#         i += 2
#     elif tag == 'sv':
#         metadata['vul'] = fields[i+1]
#         i += 2
#     elif tag == 'ah':
#         metadata['board'] = fields[i+1]
#         i += 2
#     elif tag == 'mb':
#         calls.append(fields[i+1])
#         i += 2
#     elif tag == 'an':
#         call_explanations.append(fields[i+1])
#         i += 2
#     elif tag == 'pc':
#         play_cards.append(fields[i+1])
#         i += 2
#     elif tag == 'mc':
#         metadata['mc'] = fields[i+1]
#         i += 2
#     else:
#         i += 1
#
# # 5. 打印解析结果
#
# print("==== 基本信息 ====")
# print(f"牌局: {metadata.get('board')}")
# print(f"易位: {metadata.get('vul')} (0=无)")
# print(f"已记录墩数: {metadata.get('mc')}")
#
# print("\n==== 四家手牌 ====")
# hands = metadata['hands']
# # md|3SKT8HAKJT73DK7CT7,S32H85D8653CAQ852,S65HQ42DAQJT42C63,SAQJ974H96D9CKJ94
# dealer = hands[0]
# hands_data = hands[1:].split(',')
# directions = ['N', 'E', 'S', 'W']
# dealer_dir = directions[int(dealer)-1]
# print(f"庄家: {dealer_dir}")
# for d, hand in zip(['N', 'E', 'S', 'W'], hands_data):
#     print(f"{d}: {hand}")
#
# print("\n==== 叫牌顺序 ====")
# for c, a in zip(calls, call_explanations + ['']*(len(calls)-len(call_explanations))):
#     print(f"{c:>4} {a}")
#
# print("\n==== 出牌顺序 ====")
# for idx, pc in enumerate(play_cards, start=1):
#     print(f"{idx:02d}: {pc}")

