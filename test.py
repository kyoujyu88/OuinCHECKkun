import pypdfium2 as pdfium
import cv2
import numpy as np
import os

def analyze_contracts_akaze(pdf_path, template_path, rel_stamp_coords, debug_folder="."):
    """
    特徴点マッチング（AKAZE）を使って、大きさが違っても書類を探し出します！
    """

    if not os.path.exists(template_path):
        print(f"【エラー】テンプレート画像が見つかりません…: {template_path}")
        return []
    
    if not os.path.exists(pdf_path):
        print(f"【エラー】PDFファイルが見つかりません…: {pdf_path}")
        return []

    # 1. テンプレート画像をグレースケールで読み込み
    template = cv2.imread(template_path, 0)
    if template is None:
        print("【エラー】テンプレート画像の読み込みに失敗しました…")
        return []
    
    # --- ★ AKAZE（特徴点を見つけるAIのようなもの）を準備します ---
    akaze = cv2.AKAZE_create()
    # テンプレート画像から「星（特徴点）」を見つけます
    kp_temp, des_temp = akaze.detectAndCompute(template, None)
    
    if des_temp is None or len(kp_temp) < 10:
        print("【エラー】テンプレート画像がシンプルすぎて、特徴（星）が見つけられませんでした…もう少し文字が多い部分を切り取ってみてください！")
        return []

    # 特徴を比較する係（マッチャー）を準備します
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    try:
        pdf = pdfium.PdfDocument(pdf_path)
    except Exception as e:
        print(f"【エラー】PDFを開けませんでした…: {e}")
        return []

    print(f"--- 星座を探すように処理を開始します！ (全 {len(pdf)} ページ) ---")
    results = []

    for i, page in enumerate(pdf):
        page_num = i + 1
        
        bitmap = page.render(scale=2.0, rev_byteorder=False)
        pil_image = bitmap.to_pil()
        img_bgr = np.array(pil_image)
        
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # ページ全体の画像からも「星（特徴点）」を見つけます
        kp_img, des_img = akaze.detectAndCompute(img_gray, None)

        score = 0
        has_stamp = False
        red_ratio = 0.0
        is_found = False

        # ページが真っ白じゃなければ、比較を始めます
        if des_img is not None and len(des_img) >= 2:
            # 似ている星のペアを探します
            matches = bf.knnMatch(des_temp, des_img, k=2)
            
            # 間違ったペアを弾いて、本当に正しいペア（good_matches）だけを残します
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            
            score = len(good_matches)
            
            # ★ 合格ライン：正しいペアが「10個以上」見つかれば契約書とみなします
            MIN_MATCH_COUNT = 10
            
            if score >= MIN_MATCH_COUNT:
                is_found = True
                
                # 星のペアの位置から、画像がどう歪んでいるか（どう拡大縮小されているか）を計算します
                src_pts = np.float32([ kp_temp[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp_img[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
                
                # M というのが「歪みの魔法の計算式」です
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if M is not None:
                    # --- 1. タイトル部分の青枠を描きます ---
                    h_temp, w_temp = template.shape
                    # テンプレートの四隅の座標
                    pts = np.float32([ [0,0],[0,h_temp-1],[w_temp-1,h_temp-1],[w_temp-1,0] ]).reshape(-1,1,2)
                    # 魔法の計算式で、実際のページ上の座標に変換します
                    dst = cv2.perspectiveTransform(pts, M)
                    img_bgr = cv2.polylines(img_bgr, [np.int32(dst)], True, (255,0,0), 3, cv2.LINE_AA)

                    # --- 2. ハンコを探す赤枠を計算します ---
                    rx, ry, rw, rh = rel_stamp_coords
                    # ハンコ枠の四隅の座標
                    stamp_pts = np.float32([ [rx, ry], [rx, ry+rh], [rx+rw, ry+rh], [rx+rw, ry] ]).reshape(-1,1,2)
                    # こちらも魔法の計算式で変換！ 大きさが違っても自動で合わせてくれます！
                    stamp_dst = cv2.perspectiveTransform(stamp_pts, M)
                    img_bgr = cv2.polylines(img_bgr, [np.int32(stamp_dst)], True, (0,0,255), 3, cv2.LINE_AA)
                    
                    # 赤枠の中を切り抜きます
                    stamp_rect = cv2.boundingRect(np.int32(stamp_dst))
                    sx, sy, sw, sh = stamp_rect
                    h_img, w_img = img_bgr.shape[:2]
                    sx, sy = max(0, sx), max(0, sy)
                    crop_img = img_bgr[sy:min(sy+sh, h_img), sx:min(sx+sw, w_img)]
                    
                    # --- 3. 赤色判定（今までと同じです） ---
                    if crop_img.size > 0:
                        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
                        mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
                        mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
                        stamp_mask = mask1 + mask2
                        total_pixels = crop_img.shape[0] * crop_img.shape[1]
                        red_pixels = cv2.countNonZero(stamp_mask)
                        if total_pixels > 0:
                            red_ratio = red_pixels / total_pixels
                        if red_ratio > 0.01:
                            has_stamp = True

        # デバッグ用画像の保存
        debug_filename = os.path.join(debug_folder, f"debug_page_{page_num}.jpg")
        cv2.imwrite(debug_filename, img_bgr)

        # 契約書が見つかった時だけ結果に残します
        if is_found:
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

    pdf_filename = "sample_contract.pdf"  # ← ここを直してくださいね
    target_pdf = os.path.join(current_folder, pdf_filename)
    
    template_filename = "header_template.png"  # ← ここも直してくださいね
    template_img = os.path.join(current_folder, template_filename)
    
    # テンプレート画像の左上からのハンコ枠の位置 (右へ, 下へ, 幅, 高さ)
    stamp_relative_pos = (50, 200, 150, 150) 

    if os.path.exists(target_pdf) and os.path.exists(template_img):
        data = analyze_contracts_akaze(target_pdf, template_img, stamp_relative_pos, current_folder)
        
        if not data:
            print("\n書類が見つかりませんでした…。（点数が10点に届かなかったみたいです）")
        else:
            print(f"\n★ {len(data)} 件の書類が見つかりました！ です！\n")
            for item in data:
                result_str = "【合格】押印あり" if item["has_stamp"] else "【未処理】押印なし！！"
                print(f"ページ: {item['page']} | 特徴の一致点数: {item['score']}点 | 赤色率: {item['red_ratio']:.2%} -> {result_str}")
            
            print("\n【確認】『debug_page_〇〇.jpg』を確認してみてくださいね。")
    else:
        print("\n【注意】ファイルが見つからないみたいです…。")

    print("\n--- 処理終了です ---")
