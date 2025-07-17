import pickle
import re
from wsgiref.util import request_uri
import random
from pydantic import BaseModel
import copy


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
        jiao = [s.replace("!", "") for s in jiao_raw]  # 去掉感叹号

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
            bid = jiao[i].upper() # 全部变为大写
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
            chu=BridgeTools.assign_players_by_cards(chu, n, e, s, w),
        )

def daochu(BridgeItem) -> list[tuple[list,list]]:  # 最终通过该类构造出模型输入所需要的所有的数据
    """
    特殊字符<mask>为不知道的字段，训练时有一个目标预测。<None>为已经没有的字段，不需要预测。<cls>为分类任务的目标，这里有三个（角色，花色，大小）。
    一个BridgeItem应该可以导出52个训练数据（预测每一次出牌，另外在预测的同时增加预测其他人手牌的任务）。
    最终导出为List[list]数据，外层长度为52，内层长度应该为334。保存为pkl数据，方便之后读取。
    
    在后续的处理中加入预测不同任务时的提示词<pre>和<cls>。所以后续的所有索引全部+=1
    
    最终索引0-2是预测，3 4-29 30 是第n的手牌 31 32-57 58 是e的手牌
    59 60-85 86是s的手牌 87 88-113 114 是w的手牌
    
    115-116 117-132是庄家
    133-177 是叫牌
    
    178-180 1轮
    181-183 2轮
    .....
    331-333 13轮
    """
    result = []
    for i in range(52):
        # 每次出牌的预测数据
        data = [
            "<cls>",  # 角色
            "<cls>",  # 花色
            "<cls>",  # 大小
        ]
        data.append("n")
        for j in range(13):
            data.append(BridgeItem.n[j][0])  # 13手牌
            data.append(BridgeItem.n[j][1])  # 13手牌
        data.append("n")
        data.append("e")
        for j in range(13):
            data.append(BridgeItem.e[j][0])  # 13手牌
            data.append(BridgeItem.e[j][1])  # 13手牌
        data.append("e")
        data.append("s")
        for j in range(13):
            data.append(BridgeItem.s[j][0])  # 13手牌
            data.append(BridgeItem.s[j][1])  # 13手牌
        data.append("s")
        data.append("w")
        for j in range(13):
            data.append(BridgeItem.w[j][0])  # 13手牌
            data.append(BridgeItem.w[j][1])  # 13手牌
        data.append("w")
        # 庄家和约定花色
        for j in range(2):
            data.append(BridgeItem.zhuang[0])
        for j in range(8):
            data.append(BridgeItem.zhuang[1][0])
            data.append(BridgeItem.zhuang[1][1])
        # 叫牌阶段
        l_jiao = len(BridgeItem.jiao)
        for j in range(15):
            if j < l_jiao:
                data.append(BridgeItem.jiao[j][0])
                if len(BridgeItem.jiao[j][1]) == 1:
                    if BridgeItem.jiao[j][1] == "D": # 单个的D为dabble,换为X
                        data.append("X")
                        data.append("X")
                    else: # 否则就正常加入
                        data.append(BridgeItem.jiao[j][1])
                        data.append(BridgeItem.jiao[j][1])
                else:
                    data.append(BridgeItem.jiao[j][1][0])
                    data.append(BridgeItem.jiao[j][1][1])
            else:
                data.append("<None>")
                data.append("<None>")
                data.append("<None>") # 表示没有后续叫牌了
        # 出牌阶段
        for j in range(13):
            for z in range(4): # 第j轮第z个
                data.append(BridgeItem.chu[j][z][0])
                data.append(BridgeItem.chu[j][z][1][0])
                data.append(BridgeItem.chu[j][z][1][1])
        masked_data = mask_data(data,i,BridgeItem.zhuang[0]) # 添加蒙版
        for m in masked_data:  # 将所有的处理训练数据加入到最终的返回中
            if m:
                result.append(m)
    # 返回当前的训练数据字典。
    for r in result:
        assert len(r[0]) == 335 and len(r[1]) == 334
    return result

def mask_data(data: list, i, zhuang) -> list[tuple[list, list]] | list[None]:
    """
    该函数接收当前步数：i 和拼凑完成的data，并且还有庄家，完成对data的mask替换处理
    首先找到庄家处理手牌
    """
    result = []
    hush_dict={"n":4,"s":60,"e":32,"w":88}
    if i == 0:  # 表示为预测叫牌
        # 133-177 是叫牌
        jiao = data[133:178] # 开始根据叫牌挨个构造数据集。
        j = 0
        while True:
            data_new = copy.deepcopy(data)
            if j>=15:
                break
            jiao0 = jiao[j*3:(j+1)*3] # 获取当前叫牌
            jiao_r = jiao0[0] # 当前叫牌的角色。
            if jiao_r == "<None>":
                break
            # 将其他人全部隐匿<mask>。
            for key,value in hush_dict.items():
                if key == jiao_r:
                    continue
                data_new[value:value+26] = ["<mask>"] * 26
            # 将庄家全部隐匿<mask> # 115-116 117-132是庄家
            data_new[115:133] = ["<None>"] * 18
            # 将当前的叫牌隐匿
            data_new[133+j*3:136+j*3] = ["<mask>"] * 3
            # 将当前之后的所有全部设置为None
            data_new[136+j*3:] = ["<None>"] * (len(data_new) - (136 + j*3))
            j+=1
            result.append((["<jiao>"]+data_new,data_new))
        return result
    
    
    num = random.random()+0.05 # 不希望被去掉太多
    if abs(1-i/32) >= num:
        return [None]
    # 现在预测出牌
    mash_dict = {"n":"s","s":"n","w":"e","e":"w"}
    ming = mash_dict.get(zhuang) # 得到当前的明手
    
    chu = data[178+i*3:181+i*3] # 当前的出牌
    r = chu[0] # 当前的出牌人
    if ming==r: # 如果出牌人就是角色，则
        ming = zhuang # 则庄家本次充当明手
    
    data_new = copy.deepcopy(data)
    # 首先将除了明手和当前手的所有手牌掩蔽<mask>。
    for key,value in hush_dict.items():
        if key in [ming,r]:
            continue
        else:
            data_new[value:value+26] = ["<mask>"] * 26
    # 其次再将当前的出牌掩蔽    178-180 1轮
    data_new[178+i*3:181+i*3] = ["<mask>"]*3
    # 最后将所有的后续全部变为<None>
    data_new[181+i*3:] = ["<None>"] * (len(data_new) - (181 + i*3))
    
    """
    掩蔽策略： 在每一组的第一个添加一个元素为当前预测的策略，分为预测叫牌<jiao>，出牌<pre>2种。
    分别对应不同的损失函数计算方法，并且引导模型完成不同的任务。
    
    如果当前是第0个，则目标为预测叫牌<jiao>。
    叫牌预测需要选定当前的叫牌人，并且按照顺序做蒙版添加到数据中。
    
    如果当前是第1个及以后的，逐步添加预测对方的剩余手牌<pre>。
    每个都需要添加预测下一个出牌的<pre>。
    如果出牌数量较少时很难预测出对方的手牌数，因此我们将按照一定的分布采样得到当前的样本是否需要加入到其中。
    
    另外在设计损失函数时我们希望能够尽可能精确地将对方手牌解码出来
    """
    result.append((["<pre>"] + data_new, data))
    # for r in result:
    #     print(r[0],"\n",r[1])
    # raise ""
    return result


class Bridges(BaseModel):  # 批量加载桥牌对局
    items: list[BridgeItem]

import os
def load_all_data(lin_path):
    result = []
    j = 1
    with open(lin_path, "r", encoding="utf-8") as f:
        for line in f:
            lin_decoded = line.strip()
            if not lin_decoded:
                continue
            b = BridgeItem.from_lin(lin_decoded)
            try:
                result += daochu(b)
            except Exception as e:
                print(f"第{j}行的数据有问题")
            j+=1
    with open("bridges_dataset.pkl", "wb") as f:
        pickle.dump(result, f)
    print(f"一共保存了 {len(result)} 条数据")
    return True
if __name__ == "__main__":
    load_all_data(lin_path="database/output/decoder_all.txt")
    # lin_decoded = """
    # st||md|4SA82HK5D743CAQ983,S7HQJT73D2CJT7642,SQJT5H9842DAKQJTC,SK9643HA6D9865CK5|sv|0|ah|Board 14|mb|P|mb|1C|an|Minor suit opening -- 3+ !C; 11-21 HCP; 12-22 total points|mb|P|mb|1D|an|One over one -- 4+ !D; 6+ total points|mb|P|mb|1N|an|3-5 !C; 2-3 !D; 2-3 !H; 2-3 !S; 12-14 HCP|mb|P|mb|2S|an|4+ !D; 12+ total points; stop in !S|mb|P|mb|3D|an|3-5 !C; 3 !D; 2-3 !H; 2-3 !S; 12-14 HCP; forcing to 3N|mb|P|mb|3N|an|4+ !D; 4- !H; 4- !S; 13-20 HCP; stop in !S|mb|P|mb|P|mb|P|pc|HQ|pc|H2|pc|HA|pc|H5|pc|H6|pc|HK|pc|H7|pc|H4|pc|D3|pc|D2|pc|DA|pc|D6|pc|SQ|pc|S3|pc|S2|pc|S7|pc|SJ|pc|S6|pc|S8|pc|C4|pc|S5|pc|S9|pc|SA|pc|C2|pc|D4|pc|C6|pc|DK|pc|D5|pc|DQ|pc|D8|pc|D7|pc|H3|pc|DJ|pc|D9|pc|C3|pc|C7|pc|DT|pc|C5|pc|C8|pc|CT|pc|ST|pc|SK|pc|C9|pc|CJ|pc|S4|pc|CQ|pc|HJ|pc|H8|pc|CK|pc|CA|pc|HT|pc|H9|
    # """
    # b = BridgeItem.from_lin(lin_decoded)
    # daochu(b)