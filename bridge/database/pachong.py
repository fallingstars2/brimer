from playwright.sync_api import sync_playwright, TimeoutError
import time
import os
from urllib.parse import urlparse

# ====== 参数 ======
CHROME_PATH = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
START_URL = "https://www.bridgebase.com/v3/app/lv"
TARGET_PAGES = [
    "https://www.bridgebase.com/myhands/hands.php?tourney=90031-1752545161-&username=olafssonm",
    "https://www.bridgebase.com/myhands/hands.php?tourney=90031-1752545161-&username=Baloo_rus&from_login=0",
]
OUTPUT_DIR = "output"


# 可选：从文件批量加载
def load_target_pages(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]


# ====== 爬虫逻辑 ======
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with sync_playwright() as p:
        # 启动 Chromium（有头模式）
        browser = p.chromium.launch(executable_path=CHROME_PATH, headless=False)
        context = browser.new_context()

        page = context.new_page()
        page.goto(START_URL)

        print("🚩 已打开 BBO 首页，如有验证码请手动验证 ✅")
        input("👉 验证通过后按回车继续：")

        # success_links = []

        for link in TARGET_PAGES:
            print(f"🚀 正在抓取：{link}")

            try:
                page.goto(link, timeout=30000)
                time.sleep(10)
                page.wait_for_load_state("networkidle")

            except TimeoutError:
                print(f"❌ 跳转超时：{link}")
                continue

            # 使用 tourney 参数作为文件名
            filename_base = link.split("tourney=")[-1].split("&")[0]
            safe_filename = (
                filename_base.replace("/", "").replace(":", "").strip() or "index"
            )
            html_filename = os.path.join(OUTPUT_DIR, f"{safe_filename}.html")

            # txt_filename = os.path.join(OUTPUT_DIR, f"{filename_base}.txt")

            # 保存完整 HTML
            with open(html_filename, "w", encoding="utf-8") as f_html:
                f_html.write(page.content())
            print(f"✔️ 已保存 HTML：{html_filename}")

            # # 尝试提取主要内容
            # content = ""
            # for selector in ["article", "main.page", "div.content__default"]:
            #     element = page.query_selector(selector)
            #     if element:
            #         content = element.inner_text()
            #         break
            #
            # if content:
            #     with open(txt_filename, "w", encoding="utf-8") as f_txt:
            #         f_txt.write(content)
            #     print(f"✔️ 已保存纯文本：{txt_filename}")
            #     success_links.append(link)
            # else:
            #     print(f"⚠️ 没找到主要内容标签：{link}")

            time.sleep(2)
        # # 保存成功抓取的链接
        # if success_links:
        #     with open(os.path.join(OUTPUT_DIR, "true_select_http.txt"), "a", encoding="utf-8") as f:
        #         for url in success_links:
        #             f.write(url + "\n")
        #     print(f"✅ 已保存 {len(success_links)} 条有效链接到 true_select_http.txt")
        # else:
        #     print("❌ 没有抓到任何内容！")

        browser.close()


if __name__ == "__main__":
    main()
