from pydantic import BaseModel

lin_decoded = """
st||md|4SA82HK5D743CAQ983,S7HQJT73D2CJT7642,SQJT5H9842DAKQJTC,SK9643HA6D9865CK5|sv|0|ah|Board 14|mb|P|mb|1C|an|Minor suit opening -- 3+ !C; 11-21 HCP; 12-22 total points|mb|P|mb|1D|an|One over one -- 4+ !D; 6+ total points|mb|P|mb|1N|an|3-5 !C; 2-3 !D; 2-3 !H; 2-3 !S; 12-14 HCP|mb|P|mb|2S|an|4+ !D; 12+ total points; stop in !S|mb|P|mb|3D|an|3-5 !C; 3 !D; 2-3 !H; 2-3 !S; 12-14 HCP; forcing to 3N|mb|P|mb|3N|an|4+ !D; 4- !H; 4- !S; 13-20 HCP; stop in !S|mb|P|mb|P|mb|P|pc|HQ|pc|H2|pc|HA|pc|H5|pc|H6|pc|HK|pc|H7|pc|H4|pc|D3|pc|D2|pc|DA|pc|D6|pc|SQ|pc|S3|pc|S2|pc|S7|pc|SJ|pc|S6|pc|S8|pc|C4|pc|S5|pc|S9|pc|SA|pc|C2|pc|D4|pc|C6|pc|DK|pc|D5|pc|DQ|pc|D8|pc|D7|pc|H3|pc|DJ|pc|D9|pc|C3|pc|C7|pc|DT|pc|C5|pc|C8|pc|CT|pc|ST|pc|SK|pc|C9|pc|CJ|pc|S4|pc|CQ|pc|HJ|pc|H8|pc|CK|pc|CA|pc|HT|pc|H9|
"""
import re


class BridgeTools(BaseModel):
    """
    该方法存放所有关于桥牌类的方法。包括手牌转换，胜负判断，出牌可能确定等方法
    """

    @staticmethod
    def split_hand_string(hand_str: str) -> list[str]:  # 手牌转换
        """
        输入: "SQ52HAQ7652DKTC85"
        输出: ['SQ', 'S5', 'S2', 'HA', ...]
        """
        # 用正则找到每一段花色和对应牌
        parts = re.findall(r"([SHDC])([^SHDC]+)", hand_str)

        # 拆成单张
        result = []
        for suit, ranks in parts:
            for rank in ranks:
                result.append(suit + rank)
        return result

    @staticmethod
    def assign_players_by_cards(chu, n, e, s, w):  # 出牌规则化
        """
        根据 BridgeItem 里的 n/e/s/w 四家手牌，给 chu 每张牌标记出牌人。
        返回同样结构，只是把每张牌换成 (player, card)。
        """
        # 把四家牌做成查找表
        player_hands = {"n": set(n), "e": set(e), "s": set(s), "w": set(w)}

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
    def get_final_contract(bids: list[tuple]) -> (str, str):
        # 只保留非 Pass
        valid_bids = [bid for bid in bids if bid[1] != "P"]

        # 最后一个有效叫牌就是定约
        final_bid = valid_bids[-1]  # 例如 ('w', '3N')
        final_level = final_bid[1][0]  # '3'
        final_suit = final_bid[1][1:]  # 'N'

        # 找到第一个叫出这个花色的人
        dealer = None
        for who, call in valid_bids:
            level = call[0]
            suit = call[1:]
            if suit == final_suit:
                dealer = who
                break
        # print((dealer.lower(),final_bid[1]))
        return (dealer.lower(), final_bid[1])

    @staticmethod
    def who_win_this_trick(chu: list[tuple[str, str]], contract: str) -> str:
        """
        chu: [('n', 'C5'), ('e', 'CA'), ('s', 'C2'), ('w', 'CJ')]
        contract: '2S'
        返回：该墩赢家的方位，如 'n'
        """
        # 定约花色
        trump_suit = contract[-1].upper()  # 'S', 'H', 'D', 'C', 或 'N' (无将)
        if trump_suit == "N":
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
        trumps = [
            (player, card)
            for player, card in chu
            if trump_suit and card[0] == trump_suit
        ]
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
        trick: list,  # 形如 [('w', 'D5'), ('n', 'S2'), None, None]
        hand: list[str],  # 当前玩家手牌
    ):
        """
        返回：(下一个该谁出, 可出的牌列表)
        """
        if all(h == None for h in hand):
            print("手牌已经出完，可能有错请修改")
            return None, None
        if trick[0] == None:  # 如果第一个是None
            print("当前是第一个出牌，为上一墩的获胜者先出")
            return hand, ""
        # 顺时针顺序（示例）
        order = ["n", "e", "s", "w"]
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
    zhuang: tuple[str, str]  # 庄家方向
    is_ju: str | None = None
    jiao: list[tuple]
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
            "4": ["E", "S", "W", "N"],
        }

        order = dealer_map[dealer_num]
        hand_map = dict(zip(order, hands, strict=False))

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
        print(len(chu))
        n, e, s, w = (
            BridgeTools.split_hand_string(hand_map["N"]),
            BridgeTools.split_hand_string(hand_map["E"]),
            BridgeTools.split_hand_string(hand_map["S"]),
            BridgeTools.split_hand_string(hand_map["W"]),
        )
        # 下标从 0 开始
        jiao2 = []
        i = 0
        dealer_map0 = {
            "3": "1",
            "1": "1",
            "2": "3",
            "4": "3",
        }  # 非常诡异的庄家和映射，就这样吧，恶心死了前面那个移位
        order_new = dealer_map[dealer_map0.get(dealer_num)]
        while i < len(jiao):
            dir1 = order_new[i % 4].lower()  # 循环 NESW
            bid = jiao[i]
            jiao2.append((dir1, bid))
            i += 1
        # ========= 5️⃣ 返回实例 =========
        return cls(
            n=n,
            e=e,
            s=s,
            w=w,
            zhuang=BridgeTools.get_final_contract(jiao2),
            is_ju=is_ju,  # 可以根据需要扩展
            jiao=jiao2,
            chu =chu,
            # chu=BridgeTools.assign_players_by_cards(chu, n, e, s, w),
        )


if __name__ == "__main__":
    item = BridgeItem.from_lin(lin_decoded)
    print("\n\n", item)
