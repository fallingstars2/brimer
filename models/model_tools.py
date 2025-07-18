from pydantic import BaseModel


class BrimerTools(BaseModel):
    """
    该类存放所有关于模型推理需要的工具，包括获得嵌入表示，加入位置嵌入等获得模型输出。
    整理合并模型的输出，由于我把每一项拆开来因此需要一定的合并判断。
    判断该条训练数据使用的损失函数计算（因为蒙版预测不需要顺序，因此需要将label尽可能地匹配现有的推理）。
    通过模型结果推理得到最终的输出结果（加入判断出牌可能）。
    """

    @staticmethod
    def get_embeddings(x, embeddings):
        """
        该方法用于通过输入和词嵌入字典得到最终的模型输入表示。
        """
        return embeddings

    @staticmethod
    def fix_label(y_pred, label):
        """
        该方法通过传入预测值和真实label，返回label经过最大匹配后的y_pred和label形式，能够直接做交叉熵损失。
        """
        return y_pred, label

    @staticmethod
    def next_player_and_legal_cards(
        trick: list,  # 形如 [('w', 'D5'), ('n', 'S2'), None, None]
        hand: list[str],  # 当前玩家手牌
    ):
        """
        返回：(下一个该谁出, 可出的牌列表)
        该方法用来限制模型输出为合法的出牌。
        """
        if all(h == None for h in hand):
            print("手牌已经出完，可能有错请修改")
            return None, None
        if trick[0] == None:  # 如果第一个是None
            # print("当前是第一个出牌")
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
