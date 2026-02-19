import pypdfium2 as pdfium
import cv2
import numpy as np
import os

def analyze_contracts(pdf_path, template_path, rel_stamp_coords, debug_folder="."):
    """
    契約書を特定し、ハンコ（赤色）があるか調べる関数です。
    デバッグ機能付き。
    """

    if not os.path.exists(template_path):
        print(f"【エラー】テンプレート画像が見つかりません: {template_path}")
        return []
    
    if not os.path.exists(pdf_path):
        print(f"【エラー】PDFファイルが見つかりません: {pdf_path}")
        return []

    # 1. テンプレート画像を読み込み
    template = cv2.imread(template_path, 0)
    if template is None:
        print("【エラー】テンプレート画像の読み込みに失敗しました。")
        return []
    
    # テンプレートの幅と高さを取得（枠を描くために使います）
    t_h, t_w = template.shape[:2]

    try:
        pdf = pdfium.PdfDocument(pdf_path)
    except Exception as e:
        print(f"【エラー】PDFを開けませんでした: {e}")
        return []

    print(f"--- 処理を開始します (全 {len(pdf)} ページ) ---")
    results = []

    # 3. 1ページずつ順番にチェック
    for i, page in enumerate(pdf):
        page_num = i + 1
        
        bitmap = page.render(scale=2.0, rev_byteorder=False)
        pil_image = bitmap.to_pil()
        img_bgr = np.array(pil_image)
        
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # --- A. 契約書判定 ---
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        score = max_val

        if score < 0.8:
            continue

        # --- B. ハンコ場所の特定 ---
        found_x, found_y = max_loc
        rx, ry, rw, rh = rel_stamp_coords
        
        s_x = int(found_x + rx)
        s_y = int(found_y + ry)
        s_w = int(rw)
        s_h = int(rh)

        h_img, w_img = img_bgr.shape[:2]
        s_x = max(0, min(s_x, w_img))
        s_y = max(0, min(s_y, h_img))
        
        crop_img = img_bgr[s_y:s_y+s_h, s_x:s_x+s_w]

        # --- C. 赤色（ハンコ）検出 ---
        has_stamp = False
        red_ratio = 0.0
        
        if crop_img.size > 0:
            hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
            mask = mask1 + mask2
            
            total_pixels = crop_img.shape[0] * crop_img.shape[1]
            red_pixels = cv2.countNonZero(mask)
            
            if total_pixels > 0:
                red_ratio = red_pixels / total_pixels
            
            if red_ratio > 0.01:
                has_stamp = True

        # ==========================================
        #   ★ デバッグ用画像の保存（答え合わせ）
        # ==========================================
        # 1. 見つけたテンプレートの場所に【青枠】を描く (B, G, R) = (255, 0, 0)
        cv2.rectangle(img_bgr, (found_x, found_y), (found_x + t_w, found_y + t_h), (255, 0, 0), 4)
        
        # 2. ハンコを探した場所に【赤枠】を描く (B, G, R) = (0, 0, 255)
        cv2.rectangle(img_bgr, (s_x, s_y), (s_x + s_w, s_y + s_h), (0, 0, 255), 4)
        
        # 3. 画像をファイルとして保存する
        debug_filename = os.path.join(debug_folder, f"debug_page_{page_num}.jpg")
        cv2.imwrite(debug_filename, img_bgr)
        # ==========================================

        results.append({
            "page": page_num,
            "score": score,
            "has_stamp": has_stamp,
            "red_ratio": red_ratio
        })

    return results

# ==========================================
#   実行部分
# ==========================================

if __name__ == "__main__":
    current_folder = os.path.dirname(os.path.abspath(__file__))

    pdf_filename = "sample_contract.pdf"
    target_pdf = os.path.join(current_folder, pdf_filename)
    
    template_filename = "header_template.png"
    template_img = os.path.join(current_folder, template_filename)
    
    # 座標: (右への移動量, 下への移動量, 枠の幅, 枠の高さ)
    stamp_relative_pos = (50, 200, 150, 150) 

    if os.path.exists(target_pdf) and os.path.exists(template_img):
        # 実行
        data = analyze_contracts(target_pdf, template_img, stamp_relative_pos, current_folder)
        
        if not data:
            print("\n契約書は見つかりませんでした。")
        else:
            print(f"\n★ {len(data)} 件の契約書が見つかりました！\n")
            for item in data:
                result_str = "【合格】押印あり" if item["has_stamp"] else "【未処理】押印なし！！"
                print(f"ページ: {item['page']} | 一致度: {item['score']:.2f} | 赤色率: {item['red_ratio']:.2%} -> {result_str}")
            
            print("\n【確認】プログラムと同じフォルダに『debug_page_〇〇.jpg』という画像が保存されました！")
            print("青枠が『タイトルを見つけた場所』、赤枠が『ハンコを探した場所』です。座標が合っているか確認してみてくださいね。")
    else:
        print("\n【注意】ファイルが見つかりません。")

    print("\n--- 処理終了 ---")
