import pypdfium2 as pdfium
import cv2
import numpy as np
import os

def analyze_contracts(pdf_path, template_path, rel_stamp_coords):
    """
    契約書を特定し、ハンコ（赤色）があるか調べる関数です。
    """

    # --- ファイルが存在するか、念のためここでも確認します ---
    if not os.path.exists(template_path):
        print(f"【エラー】テンプレート画像が見つかりません: {template_path}")
        return []
    
    if not os.path.exists(pdf_path):
        print(f"【エラー】PDFファイルが見つかりません: {pdf_path}")
        return []

    # 1. テンプレート画像を読み込み
    template = cv2.imread(template_path, 0)
    if template is None:
        print("【エラー】テンプレート画像の読み込みに失敗しました。画像ファイルか確認してください。")
        return []

    # 2. PDFを開きます
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
        
        # scale=2.0 で高画質化 (144dpi相当)
        bitmap = page.render(scale=2.0, rev_byteorder=False)
        pil_image = bitmap.to_pil()
        img_bgr = np.array(pil_image)
        
        # 色変換 (RGB -> BGR)
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # --- A. 契約書判定 (テンプレートマッチング) ---
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        score = max_val

        # 一致度が 0.8 未満ならスキップ
        if score < 0.8:
            continue

        # --- B. ハンコ場所の特定 ---
        found_x, found_y = max_loc
        rx, ry, rw, rh = rel_stamp_coords
        
        s_x = int(found_x + rx)
        s_y = int(found_y + ry)
        s_w = int(rw)
        s_h = int(rh)

        # 画像範囲チェック
        h_img, w_img = img_bgr.shape[:2]
        s_x = max(0, min(s_x, w_img))
        s_y = max(0, min(s_y, h_img))
        
        crop_img = img_bgr[s_y:s_y+s_h, s_x:s_x+s_w]

        # --- C. 赤色（ハンコ）検出 ---
        has_stamp = False
        red_ratio = 0.0
        
        if crop_img.size > 0:
            hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
            
            # 赤色の範囲 (2パターン)
            mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
            mask = mask1 + mask2
            
            total_pixels = crop_img.shape[0] * crop_img.shape[1]
            red_pixels = cv2.countNonZero(mask)
            
            if total_pixels > 0:
                red_ratio = red_pixels / total_pixels
            
            # 1%以上で「ハンコあり」
            if red_ratio > 0.01:
                has_stamp = True

        results.append({
            "page": page_num,
            "score": score,
            "has_stamp": has_stamp,
            "red_ratio": red_ratio
        })

    return results

# ==========================================
#   ここから実行部分です（ここを修正しました）
# ==========================================

if __name__ == "__main__":
    # ★重要：このファイルの場所（フォルダ）を自動で取得します
    # これがあれば、どこで実行しても「隣にあるファイル」をちゃんと探せます
    current_folder = os.path.dirname(os.path.abspath(__file__))

    # 1. 調査したいPDFファイル名
    # os.path.join を使って、フォルダとファイル名を正しくつなぎます
    pdf_filename = "sample_contract.pdf"  # ← ここは実際のファイル名に変えてください
    target_pdf = os.path.join(current_folder, pdf_filename)
    
    # 2. テンプレート画像名
    template_filename = "header_template.png"  # ← ここも実際のファイル名に
    template_img = os.path.join(current_folder, template_filename)
    
    # パスの確認（実行時に表示されます）
    print(f"調査対象PDF: {target_pdf}")
    print(f"テンプレート: {template_img}")

    # 3. ハンコの位置設定 (x, y, w, h)
    # テンプレート画像の左上からの距離です
    # ※うまくいかない時は、ここの数値を調整してみてください
    stamp_relative_pos = (50, 200, 150, 150) 

    # 関数実行
    if os.path.exists(target_pdf) and os.path.exists(template_img):
        data = analyze_contracts(target_pdf, template_img, stamp_relative_pos)
        
        # 結果表示
        if not data:
            print("\n契約書は見つかりませんでした。（座標がずれているか、一致度が低いかもしれません）")
        else:
            print(f"\n★ {len(data)} 件の契約書が見つかりました！ です！\n")
            for item in data:
                result_str = "【合格】押印あり" if item["has_stamp"] else "【未処理】押印なし！！"
                print(f"ページ: {item['page']} | 一致度: {item['score']:.2f} | 赤色率: {item['red_ratio']:.2%} -> {result_str}")
    else:
        print("\n【注意】ファイルが見つかりません。")
        print("ファイル名が合っているか、拡張子(.pngや.pdf)が正しいか確認してみてください。")

    print("\n--- 処理終了 ---")
