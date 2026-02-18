import pypdfium2 as pdfium
import cv2
import numpy as np

def check_contract_ultimate(pdf_path, template_path, base_stamp_roi):
    """
    pdf_path: 調査するPDF
    template_path: 契約書のヘッダー画像（テンプレート）
    base_stamp_roi: テンプレート基準のハンコ位置 (x1, y1, x2, y2)
    """

    # 1. テンプレート画像を読み込み
    template = cv2.imread(template_path, 0)
    if template is None:
        print("テンプレート画像が見つかりません")
        return

    # PDFを開く
    try:
        pdf = pdfium.PdfDocument(pdf_path)
    except Exception as e:
        print(f"PDFが開けませんでした: {e}")
        return

    results = []

    # 全ページループ
    for i in range(len(pdf)):
        page = pdf[i]
        
        # --- ここがポイント: PDFを画像(Bitmap)としてレンダリング ---
        # scale=2 で 144dpi 相当（標準的な画質）になります
        # rev_byteorder=False で OpenCV向けの BGR 配列として取得
        bitmap = page.render(scale=2, rev_byteorder=False)
        
        # バッファからNumPy配列を作成 (OpenCV形式)
        img_bgr = np.asanyarray(bitmap.buffer, dtype=np.uint8)
        
        # pypdfium2の仕様によっては (H, W, 4) のBGRAで返ることがあるため確認
        if img_bgr.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

        # ----------------------------------------------------
        # これ以降は今までの画像処理ロジックと全く同じです
        # ----------------------------------------------------

        # グレースケール変換（テンプレート用）
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 2. テンプレートマッチング
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # 一致度が低いならスキップ
        if max_val < 0.75: # 少し緩めに設定
            continue 

        # 3. 位置ズレ補正
        found_x, found_y = max_loc
        
        # 相対座標でハンコエリアを計算
        rel_x1, rel_y1, rel_x2, rel_y2 = base_stamp_roi
        s_x1, s_y1 = int(found_x + rel_x1), int(found_y + rel_y1)
        s_x2, s_y2 = int(found_x + rel_x2), int(found_y + rel_y2)

        # 画像範囲内に収める
        h, w = img_bgr.shape[:2]
        s_x1, s_y1 = max(0, s_x1), max(0, s_y1)
        s_x2, s_y2 = min(w, s_x2), min(h, s_y2)

        # ハンコエリア切り出し
        crop_img = img_bgr[s_y1:s_y2, s_x1:s_x2]

        if crop_img.size == 0:
            stamp_detected = False
        else:
            # 4. 赤色検出 (HSV)
            hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
            mask = mask1 + mask2
            
            # 赤色ピクセル率
            red_ratio = cv2.countNonZero(mask) / (crop_img.shape[0] * crop_img.shape[1])
            stamp_detected = red_ratio > 0.01

        results.append({
            "page": i + 1,
            "match_score": f"{max_val:.2f}",
            "stamp": "あり" if stamp_detected else "なし"
        })

    return results

# --- 実行 ---
# print(check_contract_ultimate("scan.pdf", "header_template.png", (500, 800, 700, 1000)))
