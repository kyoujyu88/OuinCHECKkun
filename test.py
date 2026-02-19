import pypdfium2 as pdfium
import cv2
import numpy as np
import os

def analyze_contracts(pdf_path, template_path, rel_stamp_coords, debug_folder="."):
    """
    契約書を特定し、ハンコ（赤色）があるか調べる関数です。
    画像の大きさが違っても自動で合わせて探します！
    """

    if not os.path.exists(template_path):
        print(f"【エラー】テンプレート画像が見つからないみたいです…: {template_path}")
        return []
    
    if not os.path.exists(pdf_path):
        print(f"【エラー】PDFファイルが見つからないみたいです…: {pdf_path}")
        return []

    # 1. テンプレート画像を読み込み
    template = cv2.imread(template_path, 0)
    if template is None:
        print("【エラー】テンプレート画像の読み込みに失敗してしまいました…")
        return []
    
    try:
        pdf = pdfium.PdfDocument(pdf_path)
    except Exception as e:
        print(f"【エラー】PDFを開けませんでした…: {e}")
        return []

    print(f"--- 処理を開始します！ (全 {len(pdf)} ページです) ---")
    results = []

    # 3. 1ページずつ順番にチェック
    for i, page in enumerate(pdf):
        page_num = i + 1
        
        bitmap = page.render(scale=2.0, rev_byteorder=False)
        pil_image = bitmap.to_pil()
        img_bgr = np.array(pil_image)
        
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # ==========================================
        #   A. 契約書判定 (マルチスケールで探します！)
        # ==========================================
        best_score = -1
        best_loc = None
        best_w, best_h = 0, 0
        
        t_h_orig, t_w_orig = template.shape[:2]
        
        # 0.5倍から3.0倍まで、少しずつ大きさを変えながら探します
        for scale in np.linspace(0.5, 3.0, 40):
            t_w = int(t_w_orig * scale)
            t_h = int(t_h_orig * scale)
            
            # はみ出してしまう場合や、小さすぎる場合は飛ばします
            if t_w > img_gray.shape[1] or t_h > img_gray.shape[0] or t_w == 0 or t_h == 0:
                continue
                
            resized_template = cv2.resize(template, (t_w, t_h))
            res = cv2.matchTemplate(img_gray, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            # 一番似ているところをメモしておきます
            if max_val > best_score:
                best_score = max_val
                best_loc = max_loc
                best_w = t_w
                best_h = t_h

        score = best_score
        
        # 一致度が 0.7 未満なら、契約書じゃないと判断して次に進みます
        if score < 0.7:
            continue

        # 一番ピッタリだった場所と大きさをセットします
        found_x, found_y = best_loc
        t_w, t_h = best_w, best_h

        # ==========================================
        #   B. ハンコ場所の特定
        # ==========================================
        rx, ry, rw, rh = rel_stamp_coords
        
        s_x = int(found_x + rx)
        s_y = int(found_y + ry)
        s_w = int(rw)
        s_h = int(rh)

        h_img, w_img = img_bgr.shape[:2]
        s_x = max(0, min(s_x, w_img))
        s_y = max(0, min(s_y, h_img))
        
        crop_img = img_bgr[s_y:s_y+s_h, s_x:s_x+s_w]

        # ==========================================
        #   C. 赤色（ハンコ）検出
        # ==========================================
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

        # --- デバッグ用画像の保存 ---
        # 見つけた文字の場所に【青枠】を描きます
        cv2.rectangle(img_bgr, (found_x, found_y), (found_x + t_w, found_y + t_h), (255, 0, 0), 4)
        
        # ハンコを探す場所に【赤枠】を描きます
        cv2.rectangle(img_bgr, (s_x, s_y), (s_x + s_w, s_y + s_h), (0, 0, 255), 4)
        
        debug_filename = os.path.join(debug_folder, f"debug_page_{page_num}.jpg")
        cv2.imwrite(debug_filename, img_bgr)

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

    # 実際のファイル名に合わせて書き換えてくださいね
    pdf_filename = "sample_contract.pdf"
    target_pdf = os.path.join(current_folder, pdf_filename)
    
    template_filename = "header_template.png"
    template_img = os.path.join(current_folder, template_filename)
    
    # ハンコを探す枠の位置です (右へ, 下へ, 幅, 高さ)
    stamp_relative_pos = (50, 200, 150, 150) 

    if os.path.exists(target_pdf) and os.path.exists(template_img):
        data = analyze_contracts(target_pdf, template_img, stamp_relative_pos, current_folder)
        
        if not data:
            print("\n契約書は見つかりませんでした…。（文字が違うか、もっと低い一致度かもしれません）")
        else:
            print(f"\n★ {len(data)} 件の書類が見つかりました！ です！\n")
            for item in data:
                result_str = "【合格】押印あり" if item["has_stamp"] else "【未処理】押印なし！！"
                print(f"ページ: {item['page']} | 一致度: {item['score']:.2f} | 赤色率: {item['red_ratio']:.2%} -> {result_str}")
            
            print("\n【確認】プログラムと同じフォルダに『debug_page_〇〇.jpg』という画像が保存されました！")
            print("青枠と赤枠が、思っている場所にきているか確認してみてくださいね。")
    else:
        print("\n【注意】ファイルが見つからないみたいです…。")
        print("ファイル名が間違っていないか、もう一度確認をお願いします。")

    print("\n--- 処理終了です ---")
