import pypdfium2 as pdfium
import cv2
import numpy as np
import os

def analyze_contracts(pdf_path, template_path, rel_stamp_coords):
    """
    契約書を特定し、ハンコ（赤色）があるか調べる関数です。
    
    Args:
        pdf_path (str): 調査するPDFファイルのパス
        template_path (str): 契約書のヘッダーなどを切り取った画像パス
        rel_stamp_coords (tuple): テンプレート左上から見たハンコ枠の位置 (x, y, w, h)
                                  x: 横方向の距離, y: 縦方向の距離, w: 幅, h: 高さ
    """

    # 1. テンプレート画像を読み込みます
    if not os.path.exists(template_path):
        print(f"エラー: テンプレート画像が見つかりません: {template_path}")
        return []
    
    # テンプレートをグレースケールで読み込み
    template = cv2.imread(template_path, 0)
    t_h, t_w = template.shape[:2]

    # 2. PDFを開きます
    if not os.path.exists(pdf_path):
        print(f"エラー: PDFファイルが見つかりません: {pdf_path}")
        return []
        
    try:
        pdf = pdfium.PdfDocument(pdf_path)
    except Exception as e:
        print(f"PDFを開くのに失敗しました...: {e}")
        return []

    print(f"処理を開始します... (全 {len(pdf)} ページ)")
    results = []

    # 3. 1ページずつ順番にチェックします
    for i, page in enumerate(pdf):
        page_num = i + 1
        
        # --- PDFを画像(Bitmap)に変換 ---
        # scale=2.0 は 144dpi相当（画質を良くして文字やハンコを見やすくします）
        bitmap = page.render(scale=2.0, rev_byteorder=False)
        pil_image = bitmap.to_pil()
        img_bgr = np.array(pil_image)
        
        # 色の並びをOpenCV用に変換 (RGB -> BGR)
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # --- A. 契約書かどうかチェック (テンプレートマッチング) ---
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # 一致度 (0.0 ~ 1.0)
        score = max_val

        # しきい値：0.8未満なら「契約書ではない」と判断して次へ
        if score < 0.8:
            continue

        # --- B. ハンコ場所の特定 ---
        # テンプレートが見つかった場所の左上座標
        found_x, found_y = max_loc
        
        # 指定された相対位置を足して、ハンコの場所を計算します
        rx, ry, rw, rh = rel_stamp_coords
        
        s_x = int(found_x + rx)
        s_y = int(found_y + ry)
        s_w = int(rw)
        s_h = int(rh)

        # 画像からはみ出さないように調整
        h_img, w_img = img_bgr.shape[:2]
        s_x = max(0, min(s_x, w_img))
        s_y = max(0, min(s_y, h_img))
        crop_img = img_bgr[s_y:s_y+s_h, s_x:s_x+s_w]

        # --- C. 赤色（ハンコ）の検出 ---
        has_stamp = False
        red_ratio = 0.0
        
        if crop_img.size > 0:
            # HSV色空間に変換
            hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
            
            # 赤色の範囲定義 (2パターン)
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = mask1 + mask2
            
            # 赤いピクセルの割合を計算
            total_pixels = crop_img.shape[0] * crop_img.shape[1]
            red_pixels = cv2.countNonZero(mask)
            
            if total_pixels > 0:
                red_ratio = red_pixels / total_pixels
            
            # 1%以上赤ければ「ハンコあり」とみなす
            if red_ratio > 0.01:
                has_stamp = True

            # (デバッグ用) 切り抜いた画像を保存して確認したい場合はコメントアウトを外してください
            # cv2.imwrite(f"debug_page_{page_num}_crop.jpg", crop_img)

        # 結果をリストに追加
        results.append({
            "page": page_num,
            "score": score,
            "has_stamp": has_stamp,
            "red_ratio": red_ratio
        })

    return results

# ==========================================
#   ここからメインの実行部分です
# ==========================================

if __name__ == "__main__":
    # 1. 調査したいPDFファイル名
    target_pdf = "sample_contract.pdf"
    
    # 2. テンプレート画像（契約書のタイトル部分など）
    # ※ 事前にこの画像を用意しておく必要があります
    template_img = "header_template.png"
    
    # 3. ハンコの位置設定 (x, y, w, h)
    # テンプレート画像の「左上」から見て、右に何px、下に何pxの位置にハンコ枠があるか
    # 幅(w)と高さ(h)も指定します。
    # ※ scale=2.0 で処理するため、実際のピクセル数は大きめにとるのがコツです
    stamp_relative_pos = (50, 200, 150, 150) 

    print("--- 調査を開始します ---")
    
    # 関数実行
    data = analyze_contracts(target_pdf, template_img, stamp_relative_pos)
    
    # 結果表示
    if not data:
        print("契約書は見つかりませんでした。（またはエラーが発生しました）")
    else:
        print(f"\n★ {len(data)} 件の契約書が見つかりました！\n")
        for item in data:
            result_str = "【合格】押印あり" if item["has_stamp"] else "【未処理】押印なし！！"
            print(f"ページ: {item['page']} | 一致度: {item['score']:.2f} | 赤色率: {item['red_ratio']:.2%} -> {result_str}")

    print("\n--- 処理終了 ---")
