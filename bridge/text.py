import urllib.parse
from pydantic import BaseModel

lin_encoded = """
st%7C%7Cmd%7C4S7HKT8DAKQJ84C985%2CSA52HQ764D976CQJT%2CSQT864H92D32CAK74%2CSKJ93HAJ53DT5C632%7Csv%7CN%7Cah%7CBoard%202%7Cmb%7CP%7Cmb%7C1D%7Can%7CMinor%20suit%20opening%20--%203%2B%20%21D%3B%2011-21%20HCP%3B%2012-22%20total%20points%7Cmb%7CP%7Cmb%7C1S%7Can%7COne%20over%20one%20--%204%2B%20%21S%3B%206%2B%20total%20points%7Cmb%7CP%7Cmb%7C2D%7Can%7COpener%20rebids%20his%20D%20--%203-%20%21S%3B%2011-15%20HCP%3B%20twice%20rebiddable%20%21D%3B%2012-16%20total%20points%7Cmb%7CP%7Cmb%7C3D%7Can%7C2%2B%20%21D%3B%204%2B%20%21S%3B%2010-12%20total%20points%7Cmb%7CP%7Cmb%7CP%7Cmb%7CP%7Cpc%7CCQ%7Cpc%7CCA%7Cpc%7CC2%7Cpc%7CC5%7Cpc%7CH9%7Cpc%7CH5%7Cpc%7CHT%7Cpc%7CHQ%7Cpc%7CCJ%7Cpc%7CCK%7Cpc%7CC6%7Cpc%7CC9%7Cpc%7CH2%7Cpc%7CHA%7Cpc%7CH8%7Cpc%7CH4%7Cpc%7CD5%7Cpc%7CDA%7Cpc%7CD7%7Cpc%7CD2%7Cpc%7CDK%7Cpc%7CD9%7Cpc%7CD3%7Cpc%7CDT%7Cpc%7CDQ%7Cpc%7CD6%7Cpc%7CS4%7Cpc%7CS9%7Cpc%7CDJ%7Cpc%7CS2%7Cpc%7CS6%7Cpc%7CC3%7Cpc%7CD8%7Cpc%7CS5%7Cpc%7CS8%7Cpc%7CS3%7Cpc%7CHK%7Cpc%7CH6%7Cpc%7CST%7Cpc%7CH3%7Cpc%7CD4%7Cpc%7CH7%7Cpc%7CC4%7Cpc%7CHJ%7Cpc%7CC8%7Cpc%7CCT%7Cpc%7CC7%7Cpc%7CSK%7Cpc%7CSA%7Cpc%7CSQ%7Cpc%7CSJ%7Cpc%7CS7%7C
"""
class BridgeItem(BaseModel): # 建立一个桥牌item类，存储一把对局的所有信息
    """
    需要保存的信息有：对局双发和手牌 self.n e s w
    是否有局 self.is_ju:None/ns/ew/all
    叫牌顺序（保存为列表）self.jiao [p,1d,p,...]
    出牌顺序（保存为列表，并且每一墩为一个新的列表，之后添加一个元素为谁获胜，下一个为上一墩获胜的一方出牌）chu [n,[cq,...],s,[...],...]
    """
    name:str | None = None
    n:list[str]
    e:list[str]
    s:list[str]
    w:list[str]
    zhuang:str # 表示本局的庄家
    
    is_ju:str | None = None
    jiao:list[str]
    chu:list # 该属性比较特殊
    
    @staticmethod
    def analysis_chu(self,):
        result = self.chu
        return result
    
def lin_decoder(lin_encoded:list[str]):
    l_items = []
    for l in lin_encoded:
        item = {}
        l_decoded = urllib.parse.unquote(l.strip())
        fields = l_decoded.split('|')
        # 3. 准备解析容器
        metadata = {}
        calls = []
        call_explanations = []
        play_cards = []
        # 4. 解析字段
        i = 0
        while i < len(fields):
            tag = fields[i]
            if tag == 'md':
                metadata['hands'] = fields[i+1]
                i += 2
            elif tag == 'sv':
                metadata['vul'] = fields[i+1]
                i += 2
            elif tag == 'ah':
                metadata['board'] = fields[i+1]
                i += 2
            elif tag == 'mb':
                calls.append(fields[i+1])
                i += 2
            elif tag == 'an':
                call_explanations.append(fields[i+1])
                i += 2
            elif tag == 'pc':
                play_cards.append(fields[i+1])
                i += 2
            elif tag == 'mc':
                metadata['mc'] = fields[i+1]
                i += 2
            else:
                i += 1
        item["is_ju"]=metadata.get('vul')
        
        print(fields)
    
    return l_items
lin_decoder([lin_encoded])
# 1. URL解码
lin_decoded = urllib.parse.unquote(lin_encoded.strip())

# 2. 按 | 分割
fields = lin_decoded.split('|')

# 3. 准备解析容器
metadata = {}
calls = []
call_explanations = []
play_cards = []

# 4. 解析字段
i = 0
while i < len(fields):
    tag = fields[i]
    if tag == 'md':
        metadata['hands'] = fields[i+1]
        i += 2
    elif tag == 'sv':
        metadata['vul'] = fields[i+1]
        i += 2
    elif tag == 'ah':
        metadata['board'] = fields[i+1]
        i += 2
    elif tag == 'mb':
        calls.append(fields[i+1])
        i += 2
    elif tag == 'an':
        call_explanations.append(fields[i+1])
        i += 2
    elif tag == 'pc':
        play_cards.append(fields[i+1])
        i += 2
    elif tag == 'mc':
        metadata['mc'] = fields[i+1]
        i += 2
    else:
        i += 1

# 5. 打印解析结果

print("==== 基本信息 ====")
print(f"牌局: {metadata.get('board')}")
print(f"易位: {metadata.get('vul')} (0=无)")
print(f"已记录墩数: {metadata.get('mc')}")

print("\n==== 四家手牌 ====")
hands = metadata['hands']
# md|3SKT8HAKJT73DK7CT7,S32H85D8653CAQ852,S65HQ42DAQJT42C63,SAQJ974H96D9CKJ94
dealer = hands[0]
hands_data = hands[1:].split(',')
directions = ['N', 'E', 'S', 'W']
dealer_dir = directions[int(dealer)-1]
print(f"庄家: {dealer_dir}")
for d, hand in zip(['N', 'E', 'S', 'W'], hands_data):
    print(f"{d}: {hand}")

print("\n==== 叫牌顺序 ====")
for c, a in zip(calls, call_explanations + ['']*(len(calls)-len(call_explanations))):
    print(f"{c:>4} {a}")

print("\n==== 出牌顺序 ====")
for idx, pc in enumerate(play_cards, start=1):
    print(f"{idx:02d}: {pc}")

